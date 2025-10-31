#!/usr/bin/env python3
"""
Создаёт цельный контур внешних стен с отверстиями окон и дверей.
Надежная версия с булевыми операциями.
"""

import bpy
import bmesh
import os
import json
import random
from mathutils import Vector
import sys

# ---------------------------------
# Константы
# ---------------------------------
SCALE_FACTOR = 0.01        # 1 px = 1 см
WALL_THICKNESS_PX = 22.0   # толщина стены в пикселях
WALL_HEIGHT = 3.0          # высота стены в метрах
MERGE_DISTANCE = 0.005     # допуск для Merge by Distance (5 мм)
CUT_MARGIN = 0.10          # запас для гарантии пересечения
OPENING_WIDTH_MULTIPLIER = 1.85  # масштаб ширины проёмов
WINDOW_FRAME_WIDTH = 0.05  # ширина рамки окна по периметру (м)

WINDOW_BOTTOM = 0.65
WINDOW_TOP = 2.45
DOOR_BOTTOM = 0.10
DOOR_TOP = 2.45

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "wall_coordinates_inverted.json")
OUTPUT_OBJ = os.path.join(SCRIPT_DIR, "precise_building_outline_with_openings.obj")
OBJ_HEIGHT_SOURCE_PATH = os.path.join(SCRIPT_DIR, "wall_coordinates_inverted_3d.obj")
OUTPUT_NORMALS_JSON = os.path.join(SCRIPT_DIR, "mesh_normals.json")

def _derive_defaults_from_json(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    # Извлекаем только первую часть до первого подчеркивания
    prefix = base_name.split('_')[0]
    return {
        'heights_obj': os.path.join(base_dir, f"{prefix}_wall_coordinates_inverted_3d.obj"),
        'out_obj': os.path.join(base_dir, f"{prefix}_outline_with_openings.obj"),
    }

def _apply_cli_overrides(argv):
    """Парсит CLI аргументы: --json, --heights-obj, --out.
    Возвращает кортеж (json_path, heights_obj, out_obj, normals_json). --json обязателен.
    """
    json_path = None
    heights_obj = None
    out_obj = None
    for i, a in enumerate(argv):
        if a == '--json' and i + 1 < len(argv):
            json_path = argv[i + 1]
        elif a.startswith('--json='):
            json_path = a.split('=', 1)[1]
        elif a == '--heights-obj' and i + 1 < len(argv):
            heights_obj = argv[i + 1]
        elif a.startswith('--heights-obj='):
            heights_obj = a.split('=', 1)[1]
        elif a == '--out' and i + 1 < len(argv):
            out_obj = argv[i + 1]
        elif a.startswith('--out='):
            out_obj = a.split('=', 1)[1]

    if not json_path:
        print("Ошибка: не указан путь к JSON. Использование: --json <path> [--heights-obj <path>] [--out <path>]")
        sys.exit(1)
    if not os.path.exists(json_path):
        print(f"Ошибка: JSON файл не найден: {json_path}")
        sys.exit(1)

    derived = _derive_defaults_from_json(json_path)
    heights_obj = heights_obj or derived['heights_obj']
    out_obj = out_obj or derived['out_obj']
    return json_path, heights_obj, out_obj

# Параметры фундамента 
FOUNDATION_Z_OFFSET = 0.0   # верх фундамента на уровне низа здания
FOUNDATION_THICKNESS = 0.75 # толщина фундамента в метрах
EXPORT_LABELS_IN_OBJ = True   # при True метки-конвертируются в mesh и попадают в OBJ
LABEL_Z_OFFSET = 0.05         # поднять метки на 5 см над стеной
DEBUG_WALL_TO_VISUALIZE = None   # номер стены для визуализации её граней (None, чтобы выключить)

# ---------------------------------
# Вспомогательные функции
# ---------------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    for coll in list(bpy.context.scene.collection.children):
        bpy.context.scene.collection.children.unlink(coll)


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_opening_heights_from_obj(path):
    if not os.path.exists(path):
        print(f"    ⚠️  OBJ с высотами не найден: {path}")
        return {}
    below_heights = {}
    above_heights = {}
    current_opening = None
    current_type = None

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("o "):
                current_opening = None
                current_type = None
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[1]
                    if name.startswith("Fill_Below_"):
                        current_type = "below"
                        current_opening = name[len("Fill_Below_"):]
                    elif name.startswith("Fill_Above_"):
                        current_type = "above"
                        current_opening = name[len("Fill_Above_"):]
                continue

            if line.startswith("v ") and current_opening and current_type in {"below", "above"}:
                _, xs, ys, zs = line.split()[:4]
                y = float(ys)
                if current_type == "below":
                    below_heights.setdefault(current_opening, []).append(y)
                else:
                    above_heights.setdefault(current_opening, []).append(y)

    heights = {}
    all_keys = set(below_heights) | set(above_heights)
    for opening_id in all_keys:
        bottom = max(below_heights.get(opening_id, [0.0])) if opening_id in below_heights else None
        top = min(above_heights.get(opening_id, [WALL_HEIGHT])) if opening_id in above_heights else None
        if bottom is not None and top is not None:
            heights[opening_id] = (bottom, top)

    print(f"    Высоты из OBJ: {len(heights)} проёмов")
    return heights


def get_or_create_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def get_outline_junctions_and_segments(data):
    vertices = data["building_outline"]["vertices"]
    outline_ids = {v["junction_id"] for v in vertices}

    segments_with_bbox = []

    for group in ("wall_segments_from_openings", "wall_segments_from_junctions"):
        for seg in data.get(group, []):
            if seg["start_junction_id"] in outline_ids and seg["end_junction_id"] in outline_ids:
                segments_with_bbox.append(seg)

    segment_dict = {}
    for seg in segments_with_bbox:
        j1 = seg["start_junction_id"]
        j2 = seg["end_junction_id"]
        segment_dict[(j1, j2)] = seg
        segment_dict[(j2, j1)] = seg

    edges_without_bbox = []
    for idx in range(len(vertices) - 1):
        v1 = vertices[idx]
        v2 = vertices[idx + 1]
        j1, j2 = v1["junction_id"], v2["junction_id"]
        if (j1, j2) not in segment_dict:
            edges_without_bbox.append(
                {
                    "start_junction_id": j1,
                    "end_junction_id": j2,
                    "start_vertex": v1,
                    "end_vertex": v2,
                }
            )

    print(f"    Вершин контура: {len(vertices)}")
    print(f"    Сегментов с bbox: {len(segments_with_bbox)}")
    print(f"    Ребер без bbox: {len(edges_without_bbox)}")

    return vertices, segments_with_bbox, edges_without_bbox

def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    if n % 2:
        return s[m]
    return 0.5 * (s[m - 1] + s[m])

def build_outline_runs(outline_vertices, scale_factor=SCALE_FACTOR):
    """Строит участки стены только от угла (corner=1) до угла (corner=1).

    Игнорирует промежуточные узлы (corner=0) — они не режут стену.
    Для каждого участка вычисляет:
      - axis: 'X' (вертикальная) или 'Y' (горизонтальная)
      - plane: координата плоскости (x или y, м)
      - span_min/span_max: диапазон вдоль продольной оси (м)
      - start_junction_id/end_junction_id: ограничители участка
    """
    if not outline_vertices:
        return []

    verts = [
        {
            'x': float(v['x']) * scale_factor,
            'y': float(v['y']) * scale_factor,
            'corner': int(v.get('corner', 0)) == 1,
            'junction_id': v.get('junction_id')
        }
        for v in outline_vertices
    ]

    corner_idx = [i for i, v in enumerate(verts) if v['corner']]
    if len(corner_idx) < 2:
        # нет двух углов — вернём пустой список, чтобы не создавать ложные стены
        return []

    runs = []
    n = len(verts)
    for k in range(len(corner_idx)):
        i0 = corner_idx[k]
        i1 = corner_idx[(k + 1) % len(corner_idx)]

        seg = []
        i = i0
        while True:
            seg.append(verts[i])
            if i == i1:
                break
            i = (i + 1) % n

        xs = [p['x'] for p in seg]
        ys = [p['y'] for p in seg]
        dx = (max(xs) - min(xs)) if xs else 0.0
        dy = (max(ys) - min(ys)) if ys else 0.0

        if dx >= dy:
            axis = 'Y'
            plane = _median(ys)
            smin, smax = (min(xs), max(xs)) if xs else (0.0, 0.0)
        else:
            axis = 'X'
            plane = _median(xs)
            smin, smax = (min(ys), max(ys)) if ys else (0.0, 0.0)

        runs.append({
            'axis': axis,
            'plane': plane,
            'span_min': smin,
            'span_max': smax,
            'start_junction_id': seg[0].get('junction_id'),
            'end_junction_id': seg[-1].get('junction_id'),
            'source': 'corner_run'
        })

    print(f"    Контур (угол→угол): {len(runs)} участков")
    for i, r in enumerate(runs[:20]):
        print(f"      RunC #{i}: ось {r['axis']}, плоскость={r['plane']:.3f}, span=({r['span_min']:.3f}..{r['span_max']:.3f}), J{r['start_junction_id']}→J{r['end_junction_id']}")
    if len(runs) > 20:
        print(f"      ... ещё {len(runs) - 20} участков")
    return runs


def build_detailed_outline_runs(outline_vertices, segments_with_bbox, edges_without_bbox, scale_factor=SCALE_FACTOR):
    """Создает участки контура (runs) по фактическим сегментам, чтобы ограничивать стены по длине.

    Каждый run содержит:
      - axis: 'X' (вертикальная стена, нормали ±X) или 'Y' (горизонтальная, нормали ±Y)
      - plane: координата плоскости стены (x или y в метрах)
      - span_min/span_max: продольный диапазон по оси протяженности (в метрах)
      - start_junction_id/end_junction_id: идентификаторы ограничивающих узлов
    """
    runs = []

    def add_run(axis, plane, smin, smax, meta=None):
        if smin > smax:
            smin, smax = smax, smin
        entry = {'axis': axis, 'plane': plane, 'span_min': smin, 'span_max': smax}
        if meta:
            entry.update(meta)
        runs.append(entry)

    # 1) Сегменты с bbox (из openings и junctions)
    for seg in segments_with_bbox:
        bbox = seg.get('bbox') or {}
        ori = seg.get('orientation') or bbox.get('orientation')
        x = float(bbox.get('x', 0.0)) * scale_factor
        y = float(bbox.get('y', 0.0)) * scale_factor
        w = float(bbox.get('width', 0.0)) * scale_factor
        h = float(bbox.get('height', 0.0)) * scale_factor

        if ori == 'vertical':
            axis = 'X'
            plane = x + 0.5 * w
            smin, smax = y, y + h
        else:
            axis = 'Y'
            plane = y + 0.5 * h
            smin, smax = x, x + w

        add_run(axis, plane, smin, smax, meta={
            'segment_id': seg.get('segment_id'),
            'start_junction_id': seg.get('start_junction_id'),
            'end_junction_id': seg.get('end_junction_id'),
            'source': 'bbox',
        })

    # 2) Рёбра контура без bbox (между соседними вершинами)
    for edge in edges_without_bbox:
        v1 = edge.get('start_vertex') or {}
        v2 = edge.get('end_vertex') or {}
        x1 = float(v1.get('x', 0.0)) * scale_factor
        y1 = float(v1.get('y', 0.0)) * scale_factor
        x2 = float(v2.get('x', 0.0)) * scale_factor
        y2 = float(v2.get('y', 0.0)) * scale_factor

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx >= dy:
            axis = 'Y'
            plane = 0.5 * (y1 + y2)
            smin, smax = min(x1, x2), max(x1, x2)
        else:
            axis = 'X'
            plane = 0.5 * (x1 + x2)
            smin, smax = min(y1, y2), max(y1, y2)

        add_run(axis, plane, smin, smax, meta={
            'start_junction_id': edge.get('start_junction_id'),
            'end_junction_id': edge.get('end_junction_id'),
            'source': 'edge',
        })

    # Лёгкая сортировка для стабильного вывода
    runs.sort(key=lambda r: (r['axis'], round(r['plane'], 4), round(r['span_min'], 4)))
    print(f"    Детализированных участков (по сегментам): {len(runs)}")
    for i, r in enumerate(runs[:20]):
        print(f"      RunD #{i}: ось {r['axis']}, плоскость={r['plane']:.3f}, span=({r['span_min']:.3f}..{r['span_max']:.3f}), src={r.get('source')}")
    if len(runs) > 20:
        print(f"      ... ещё {len(runs) - 20} участков")
    return runs


def create_rectangles(data):
    """Создаёт прямоугольники стен по угловым вершинам building_outline, используя bbox-подсказки.

    Механизм:
      - Идём от corner=1 к corner=1 по порядку обхода контура
      - Плоскость стены берём из bbox (центр по поперечной оси), иначе средняя по вершинам
      - Длину берём от ближайших к вершинам краёв bbox по продольной оси, иначе сами вершины
      - Спец-правка для внутреннего правого поворота в J9
    """

    def extract_corner_vertices(building_outline, scale_factor=SCALE_FACTOR):
        out = []
        if not building_outline or 'vertices' not in building_outline:
            return out
        for v in building_outline['vertices']:
            if int(v.get('corner', 0)) == 1:
                out.append({
                    'x': float(v['x']) * scale_factor,
                    'y': float(v['y']) * scale_factor,
                    'junction_id': v.get('junction_id'),
                })
        return out

    def _bbox_entries_from_data(data):
        entries = []
        def push_seg(seg):
            bbox = seg.get('bbox') or {}
            if not bbox:
                return
            entries.append({
                'kind': 'segment',
                'orientation': bbox.get('orientation') or seg.get('orientation'),
                'x': float(bbox.get('x', 0.0)),
                'y': float(bbox.get('y', 0.0)),
                'width': float(bbox.get('width', 0.0)),
                'height': float(bbox.get('height', 0.0)),
                'start_junction_id': seg.get('start_junction_id'),
                'end_junction_id': seg.get('end_junction_id'),
            })
        for group in ('wall_segments_from_openings', 'wall_segments_from_junctions'):
            for seg in data.get(group, []) or []:
                push_seg(seg)
        for op in data.get('openings', []) or []:
            bbox = op.get('bbox') or {}
            if not bbox:
                continue
            entries.append({
                'kind': 'opening',
                'orientation': op.get('orientation') or bbox.get('orientation'),
                'x': float(bbox.get('x', 0.0)),
                'y': float(bbox.get('y', 0.0)),
                'width': float(bbox.get('width', 0.0)),
                'height': float(bbox.get('height', 0.0)),
                'edge_junctions': [ej.get('junction_id') for ej in (op.get('edge_junctions') or []) if 'junction_id' in ej],
            })
        return entries

    def _entries_touching_junction(entries, junction_id):
        touched = []
        for e in entries:
            if e['kind'] == 'segment':
                if e.get('start_junction_id') == junction_id or e.get('end_junction_id') == junction_id:
                    touched.append(e)
            else:
                ejs = e.get('edge_junctions') or []
                if junction_id in ejs:
                    touched.append(e)
        return touched

    def _plane_from_entry(entry):
        ori = (entry.get('orientation') or '').lower()
        x = float(entry.get('x', 0.0)) * SCALE_FACTOR
        y = float(entry.get('y', 0.0)) * SCALE_FACTOR
        w = float(entry.get('width', 0.0)) * SCALE_FACTOR
        h = float(entry.get('height', 0.0)) * SCALE_FACTOR
        if ori == 'vertical':
            return 'vertical', x + 0.5 * w
        else:
            return 'horizontal', y + 0.5 * h

    def _s_extreme_near_vertex(entry, vx_m, vy_m):
        ori = (entry.get('orientation') or '').lower()
        x0 = float(entry.get('x', 0.0)) * SCALE_FACTOR
        y0 = float(entry.get('y', 0.0)) * SCALE_FACTOR
        w = float(entry.get('width', 0.0)) * SCALE_FACTOR
        h = float(entry.get('height', 0.0)) * SCALE_FACTOR
        if ori == 'vertical':
            cand = [y0, y0 + h]
            val = min(cand, key=lambda yy: abs(yy - vy_m))
            return 'vertical', val
        else:
            cand = [x0, x0 + w]
            val = min(cand, key=lambda xx: abs(xx - vx_m))
            return 'horizontal', val

    def collect_corner_guides(corner_vertices, data):
        entries = _bbox_entries_from_data(data)
        guides = {}
        for v in corner_vertices:
            j = v.get('junction_id')
            vx, vy = float(v['x']), float(v['y'])
            planes_h, planes_v = [], []
            exts_h, exts_v = [], []
            for e in _entries_touching_junction(entries, j):
                ax, plane = _plane_from_entry(e)
                if ax == 'horizontal':
                    planes_h.append(float(plane))
                else:
                    planes_v.append(float(plane))
                ax2, sval = _s_extreme_near_vertex(e, vx, vy)
                if ax2 == 'horizontal':
                    exts_h.append(float(sval))
                else:
                    exts_v.append(float(sval))
            guides[j] = {
                'horizontal_planes': planes_h,
                'vertical_planes': planes_v,
                'horizontal_extremes': exts_h,
                'vertical_extremes': exts_v,
            }
        return guides

    def pick_plane(hints, fallback):
        return (sum(hints) / float(len(hints))) if hints else float(fallback)

    def pick_extreme(hints, fallback):
        return (min(hints, key=lambda v: abs(v - fallback))) if hints else float(fallback)

    def build_wall_rectangles_from_corners(corner_vertices, guides,
                                           wall_thickness_px=WALL_THICKNESS_PX,
                                           scale_factor=SCALE_FACTOR):
        rects = []
        n = len(corner_vertices)
        if n < 2:
            return rects
        half_th = float(wall_thickness_px) * float(scale_factor) / 2.0

        j_to_idx = {v.get('junction_id'): idx for idx, v in enumerate(corner_vertices)}

        def turn_at(jid: int):
            if jid not in j_to_idx:
                return 'straight'
            idx = j_to_idx[jid]
            prev_v = corner_vertices[(idx - 1) % n]
            cur_v = corner_vertices[idx]
            next_v = corner_vertices[(idx + 1) % n]
            v1x = cur_v['x'] - prev_v['x']
            v1y = cur_v['y'] - prev_v['y']
            v2x = next_v['x'] - cur_v['x']
            v2y = next_v['y'] - cur_v['y']
            cross = v1x * v2y - v1y * v2x
            if cross > 1e-9:
                return 'left'
            if cross < -1e-9:
                return 'right'
            return 'straight'

        for i in range(n):
            v1 = corner_vertices[i]
            v2 = corner_vertices[(i + 1) % n]
            x1, y1 = float(v1['x']), float(v1['y'])
            x2, y2 = float(v2['x']), float(v2['y'])
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            j1, j2 = v1.get('junction_id'), v2.get('junction_id')
            g1 = guides.get(j1, {})
            g2 = guides.get(j2, {})

            if dx >= dy:
                y_plane_1 = pick_plane(g1.get('horizontal_planes', []), (y1 + y2) * 0.5)
                y_plane_2 = pick_plane(g2.get('horizontal_planes', []), (y1 + y2) * 0.5)
                y_plane = 0.5 * (y_plane_1 + y_plane_2) if abs(y_plane_1 - y_plane_2) < half_th else 0.5 * (y1 + y2)
                x_edge_1 = pick_extreme(g1.get('horizontal_extremes', []), x1)
                x_edge_2 = pick_extreme(g2.get('horizontal_extremes', []), x2)
                # Спец-правка J9: горизонтальный, который заканчивается в J9
                if (j2 == 9) and (turn_at(9) == 'right'):
                    vplane = pick_plane(g2.get('vertical_planes', []), (x1 + x2) * 0.5)
                    x_edge_2 = vplane - half_th
                x_min, x_max = (x_edge_1, x_edge_2) if x_edge_1 <= x_edge_2 else (x_edge_2, x_edge_1)
                corners = [
                    (x_min, y_plane - half_th),
                    (x_max, y_plane - half_th),
                    (x_max, y_plane + half_th),
                    (x_min, y_plane + half_th),
                ]
                rects.append({
                    'corners': corners,
                    'id': f'wall_corner_{j1}_to_{j2}',
                    'start_junction_id': j1,
                    'end_junction_id': j2,
                    'orientation': 'horizontal',
                })
            else:
                x_plane_1 = pick_plane(g1.get('vertical_planes', []), (x1 + x2) * 0.5)
                x_plane_2 = pick_plane(g2.get('vertical_planes', []), (x1 + x2) * 0.5)
                x_plane = 0.5 * (x_plane_1 + x_plane_2) if abs(x_plane_1 - x_plane_2) < half_th else 0.5 * (x1 + x2)
                y_edge_1 = pick_extreme(g1.get('vertical_extremes', []), y1)
                y_edge_2 = pick_extreme(g2.get('vertical_extremes', []), y2)
                # Спец-правка J9: вертикальный, который начинается в J9
                if (j1 == 9) and (turn_at(9) == 'right'):
                    yplane = pick_plane(g1.get('horizontal_planes', []), (y1 + y2) * 0.5)
                    y_edge_1 = yplane + half_th
                y_min, y_max = (y_edge_1, y_edge_2) if y_edge_1 <= y_edge_2 else (y_edge_2, y_edge_1)
                corners = [
                    (x_plane - half_th, y_min),
                    (x_plane + half_th, y_min),
                    (x_plane + half_th, y_max),
                    (x_plane - half_th, y_max),
                ]
                rects.append({
                    'corners': corners,
                    'id': f'wall_corner_{j1}_to_{j2}',
                    'start_junction_id': j1,
                    'end_junction_id': j2,
                    'orientation': 'vertical',
                })
        return rects

    # Данные для построения
    if ("building_outline" not in data) or (not data["building_outline"]) or ("vertices" not in data["building_outline"]) or (not data["building_outline"]["vertices"]):
        print("    ⚠️ Нет корректного building_outline — прямоугольники не созданы")
        return []
    corners = extract_corner_vertices(data['building_outline'])
    guides = collect_corner_guides(corners, data)
    rectangles = build_wall_rectangles_from_corners(corners, guides)
    print(f"    Всего прямоугольников стен: {len(rectangles)} (углы+bbox)")
    return rectangles


def create_wall_mesh(rectangles, wall_height=WALL_HEIGHT):
    vertices = []
    faces = []
    offset = 0

    for rect in rectangles:
        lower = [(x, y, 0.0) for (x, y) in rect["corners"]]
        upper = [(x, y, wall_height) for (x, y) in rect["corners"]]
        vertices.extend(lower + upper)

        faces.append([offset + 0, offset + 1, offset + 2, offset + 3])       # bottom
        faces.append([offset + 4, offset + 5, offset + 6, offset + 7])       # top

        for i in range(4):
            j = (i + 1) % 4
            faces.append(
                [
                    offset + i,
                    offset + j,
                    offset + 4 + j,
                    offset + 4 + i,
                ]
            )

        offset += 8

    mesh = bpy.data.meshes.new("OutlineWallMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Outline_Walls", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = obj

    print(f"    Mesh создан: {len(vertices)} вершин, {len(faces)} граней")
    return obj


def create_foundation(foundation_data, z_offset=FOUNDATION_Z_OFFSET, thickness=FOUNDATION_THICKNESS, scale_factor=SCALE_FACTOR):
    """Создает 3D меш фундамента из данных JSON (как в merge_with_remesh_fine.py)."""
    if not foundation_data or 'vertices' not in foundation_data:
        print("    ⚠️  Данные фундамента не найдены в JSON")
        return None

    vertices_2d = foundation_data['vertices']

    mesh = bpy.data.meshes.new(name="Foundation_Mesh")
    foundation_obj = bpy.data.objects.new("Foundation", mesh)
    bpy.context.collection.objects.link(foundation_obj)

    vertices = []
    for v in vertices_2d:
        vertices.append((v['x'] * scale_factor, v['y'] * scale_factor, z_offset))
    for v in vertices_2d:
        vertices.append((v['x'] * scale_factor, v['y'] * scale_factor, z_offset - thickness))

    num_verts = len(vertices_2d)
    faces = []
    top_face = list(range(num_verts))
    faces.append(top_face)
    bottom_face = list(range(num_verts, 2 * num_verts))
    bottom_face.reverse()
    faces.append(bottom_face)
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        face = [i, next_i, next_i + num_verts, i + num_verts]
        faces.append(face)

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    bpy.context.view_layer.objects.active = foundation_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Материал тёмно-серый
    mat = bpy.data.materials.new(name="Foundation_Material_Dark_Gray")
    mat.use_nodes = True
    try:
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
    except Exception:
        pass
    foundation_obj.data.materials.append(mat)

    print(f"    Создан фундамент: {len(vertices)} вершин, {len(faces)} граней")
    print(f"    Z: {z_offset}м до {z_offset - thickness}м, толщина: {thickness}м")

    return foundation_obj


def merge_duplicate_vertices(obj, distance):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=distance)
    bpy.ops.object.mode_set(mode='OBJECT')


def recalc_normals(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def collect_outline_openings(data, outline_ids):
    openings = []
    for opening in data.get("openings", []):
        edge_junctions = opening.get("edge_junctions", [])
        if edge_junctions and all(ej["junction_id"] in outline_ids for ej in edge_junctions):
            openings.append(opening)
    return openings


def opening_bounds(opening):
    if opening.get("type") == "door":
        return DOOR_BOTTOM, DOOR_TOP
    return WINDOW_BOTTOM, WINDOW_TOP


def create_opening_cutters(openings, opening_heights):
    entries = []
    debug_collection = get_or_create_collection("OPENINGS_DEBUG")

    for opening in openings:
        bbox = opening.get("bbox")
        if not bbox:
            continue

        orientation = opening.get("orientation", "horizontal")
        
        # Для гарантии пересечения делаем вырезатели больше
        if orientation == "vertical":
            # Вертикальные проемы - глубина задаёт ширину отверстия вдоль стены
            width = WALL_THICKNESS_PX * SCALE_FACTOR * 3.0  # Толщина стены с запасом
            depth = bbox["height"] * SCALE_FACTOR * OPENING_WIDTH_MULTIPLIER
        else:
            # Горизонтальные проемы - используем ширину проема
            width = bbox["width"] * SCALE_FACTOR * OPENING_WIDTH_MULTIPLIER
            depth = WALL_THICKNESS_PX * SCALE_FACTOR * 3.0  # Толщина стены с запасом


        x_center = (bbox["x"] + bbox["width"] / 2.0) * SCALE_FACTOR
        y_center = (bbox["y"] + bbox["height"] / 2.0) * SCALE_FACTOR

        # Получаем высоты проема
        bottom_top = opening_heights.get(opening["id"]) if opening_heights else None
        if bottom_top:
            bottom, top = bottom_top
            # Добавляем запас по высоте
            height = (top - bottom) * 2  # +20% к высоте
            z_center = (bottom + top) / 2.0
        else:
            bottom, top = opening_bounds(opening)
            height = (top - bottom) * 1.2  # +20% к высоте
            z_center = (bottom + top) / 2.0

        print(f"    Проем '{opening.get('id', 'unknown')}':")
        print(f"      Ориентация: {orientation}")
        print(f"      Размеры вырезателя: {width:.3f} x {depth:.3f} x {height:.3f}")
        print(f"      Позиция: ({x_center:.3f}, {y_center:.3f}, {z_center:.3f})")

        # СОЗДАЕМ ВЫРЕЗАТЕЛЬ КАК ОБЪЕКТ-КУБ
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        cutter_obj = bpy.context.active_object
        cutter_obj.name = f"Opening_Cutter_{opening.get('id', 'unknown')}"
        cutter_obj.location = (x_center, y_center, z_center)
        cutter_obj.scale = (width/2, depth/2, height/2)
        
        # Перемещаем в коллекцию отладки
        debug_collection.objects.link(cutter_obj)
        bpy.context.scene.collection.objects.unlink(cutter_obj)

        entries.append({
            "opening": opening,
            "cutter": cutter_obj,
        })

    print(f"    Создано вырезателей: {len(entries)}")
    return entries

def create_opening_fills(openings, opening_heights,
                         wall_thickness_px=WALL_THICKNESS_PX,
                         scale_factor=SCALE_FACTOR,
                         doors_near_windows=None):
    """Создаёт заполняющие объекты в проёмах: окна — голубые, двери — тёмно-коричневые.

    Геометрия совпадает с размерами вырезателей (cutters).
    Объекты складываются в коллекцию "OPENING_FILLS".
    """
    fills_collection = get_or_create_collection("OPENING_FILLS")

    # Материалы
    mat_window = get_or_create_material("Window_Fill_Blue", (0.45, 0.75, 1.0, 1.0))
    mat_door = get_or_create_material("Door_Fill_Dark_Brown", (0.20, 0.12, 0.05, 1.0))

    thickness_m = float(wall_thickness_px) * float(scale_factor)
    created = []

    for opening in openings:
        bbox = opening.get("bbox")
        if not bbox:
            continue

        otype = (opening.get("type") or "").lower()
        orientation = opening.get("orientation", "horizontal")

        # Центр в XY
        x_center = (bbox["x"] + bbox["width"] / 2.0) * scale_factor
        y_center = (bbox["y"] + bbox["height"] / 2.0) * scale_factor

        # Высоты Z (или совместимый аппрокс.)
        bottom_top = opening_heights.get(opening["id"]) if opening_heights else None
        if bottom_top:
            bottom, top = bottom_top
        else:
            bottom, top = opening_bounds(opening)

        # XY размеры и высота — точно как у вырезателей
        if bottom_top:
            # логика cutters: высота = (top-bottom) * 2.0
            height = (float(top) - float(bottom)) * 2.0
            z_center = (float(bottom) + float(top)) * 0.5
        else:
            # логика cutters: высота = (top-bottom) * 1.2
            height = (float(top) - float(bottom)) * 1.2
            z_center = (float(bottom) + float(top)) * 0.5

        if orientation == "vertical":
            # Делаем толщину в 2 раза меньше (было 3.0)
            width = thickness_m * 1.5
            depth = float(bbox["height"]) * scale_factor * OPENING_WIDTH_MULTIPLIER
        else:
            width = float(bbox["width"]) * scale_factor * OPENING_WIDTH_MULTIPLIER
            # Делаем толщину в 2 раза меньше (было 3.0)
            depth = thickness_m * 1.5

        # Создаём заполнитель (без рамки)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, y_center, z_center))
        fill_obj = bpy.context.active_object
        fill_obj.name = ("Window_Fill_" if otype == "window" else "Door_Fill_") + str(opening.get('id', 'unknown'))
        fill_obj.scale = (width / 2.0, depth / 2.0, height / 2.0)

        # Материал по типу
        fill_obj.data.materials.clear()
        if otype == "window":
            fill_obj.data.materials.append(mat_window)
        else:
            # Если дверь близко к окну (< 0.5 толщины стены) — красим как окно
            near = bool(doors_near_windows) and (opening.get('id') in doors_near_windows)
            fill_obj.data.materials.append(mat_window if near else mat_door)

        # Перенос в коллекцию: сначала убрать из всех текущих
        for c in list(fill_obj.users_collection):
            c.objects.unlink(fill_obj)
        fills_collection.objects.link(fill_obj)

        created.append(fill_obj)

        # Рамки по периметру окон отключены по запросу

    print(f"    Заполнители проёмов созданы: {len(created)}")
    return created


def apply_boolean_operations(wall_obj, entries):
    """Применяем булевы операции через модификаторы"""
    if not entries:
        print("    Проёмов нет — булевы операции пропущены.")
        return

    print("    Применение булевых операций...")
    
    # Убедимся, что стена имеет хорошую геометрию
    bpy.context.view_layer.objects.active = wall_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    successful_cuts = 0
    
    for i, entry in enumerate(entries):
        cutter = entry["cutter"]
        print(f"      Вырез {i+1}/{len(entries)}: {cutter.name}")
        
        # Активируем стену
        bpy.context.view_layer.objects.active = wall_obj
        wall_obj.select_set(True)
        
        # Добавляем модификатор Boolean
        mod = wall_obj.modifiers.new(name=f"Opening_{i}", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.solver = 'EXACT'  # Используем EXACT solver для надежности
        mod.object = cutter

        # Пытаемся применить модификатор
        try:
            # Сначала показываем модификатор в режиме просмотра
            mod.show_viewport = True
            mod.show_render = True
            
            # Затем применяем
            bpy.ops.object.modifier_apply(modifier=mod.name)
            successful_cuts += 1
            print(f"        ✅ Булева операция успешна")
            
        except Exception as e:
            print(f"        ❌ Ошибка булевой операции: {e}")
            # Удаляем проблемный модификатор
            wall_obj.modifiers.remove(mod)

        # Удаляем вырезатель
        bpy.data.objects.remove(cutter, do_unlink=True)

    print(f"    Успешно применено {successful_cuts} из {len(entries)} булевых операций")


def cleanup_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


# -------------------------------
# Идентификация стен и метки
# -------------------------------
def identify_and_number_walls_from_mesh(
    obj,
    position_tolerance=0.1,
    normal_threshold=0.9,
    min_face_count=4,
    min_wall_length=WALL_THICKNESS_PX * SCALE_FACTOR,
    merge_position_eps=WALL_THICKNESS_PX * SCALE_FACTOR * 1.1,
    outline_runs=None,
):
    """Группирует грани в стены и присваивает номера.

    Принципы:
    - Граним задаём доминантную ось по нормали (±X или ±Y), позицию берём по поперечной оси.
    - Сначала объединяем группы с близкой позицией (игнорируя знак нормали),
      затем фильтруем кластеры по общему количеству граней и длине стены.
    - Длину измеряем по оси протяженности: для оси 'X' длина по Y; для оси 'Y' длина по X.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # integer слой для номера стены
    wall_num_layer = bm.faces.layers.int.get("wall_number")
    if wall_num_layer is None:
        wall_num_layer = bm.faces.layers.int.new("wall_number")

    for f in bm.faces:
        f[wall_num_layer] = -1

    wall_groups = {}

    faces_with_dom = 0
    faces_without_dom = 0

    face_meta = []
    for face in bm.faces:
        n = face.normal.copy()
        n.normalize()
        nx, ny = n.x, n.y

        axis = None
        sign = 0
        if abs(nx) > normal_threshold:
            axis = 'X'
            sign = 1 if nx > 0 else -1
        elif abs(ny) > normal_threshold:
            axis = 'Y'
            sign = 1 if ny > 0 else -1

        if axis is None:
            faces_without_dom += 1
            continue
        faces_with_dom += 1

        c = face.calc_center_median()
        plane_pos = c.x if axis == 'X' else c.y
        along_pos = c.y if axis == 'X' else c.x

        face_meta.append({
            'face': face,
            'axis': axis,
            'sign': sign,
            'plane_pos': plane_pos,
            'along_pos': along_pos,
        })

    if outline_runs:
        # Привязка по детальным участкам: ограничиваем по плоскости и длине
        plane_tol = WALL_THICKNESS_PX * SCALE_FACTOR * 1.25
        span_eps = 0.02  # 2 см запас по длине
        for idx, run in enumerate(outline_runs):
            axis = run['axis']
            plane = run['plane']
            smin = run['span_min'] - span_eps
            smax = run['span_max'] + span_eps

            buckets = {+1: [], -1: []}
            for fm in face_meta:
                if fm['axis'] != axis:
                    continue
                if abs(fm['plane_pos'] - plane) > plane_tol:
                    continue
                if smin <= fm['along_pos'] <= smax:
                    buckets[fm['sign']].append(fm['face'])

            for sign, faces in buckets.items():
                if faces:
                    wall_groups[(axis, sign, plane, idx)] = faces
    else:
        # Старая схема: группируем по близкой плоскости
        for fm in face_meta:
            axis = fm['axis']
            sign = fm['sign']
            position = fm['plane_pos']
            group_key = None
            for key in wall_groups.keys():
                key_axis, key_sign, key_pos = key
                if key_axis == axis and key_sign == sign and abs(position - key_pos) < position_tolerance:
                    group_key = key
                    break
            if group_key is None:
                group_key = (axis, sign, position)
                wall_groups[group_key] = []
            wall_groups[group_key].append(fm['face'])

    # 1) Объединяем группы по близкой позиции (игнорируя знак нормали)
    merged = []
    if outline_runs:
        # Группы уже разбиты по участкам — не объединяем между run'ами
        for key, faces in wall_groups.items():
            axis, sign, position, run_idx = key
            merged.append({
                'axis': axis,
                'position': position,
                'faces': list(faces),
                'sign_counts': {sign: len(faces)},
                'run_idx': run_idx,
            })
    else:
        for (axis, sign, position), faces in wall_groups.items():
            # Найдём кластер с той же осью и близкой позицией
            cluster_idx = None
            for i, cluster in enumerate(merged):
                if cluster['axis'] != axis:
                    continue
                if abs(position - cluster['position']) < merge_position_eps:
                    cluster_idx = i
                    break
            if cluster_idx is None:
                merged.append({
                    'axis': axis,
                    'position': position,
                    'faces': list(faces),
                    'sign_counts': {sign: len(faces)}
                })
            else:
                cluster = merged[cluster_idx]
                cluster['faces'].extend(faces)
                cluster['sign_counts'][sign] = cluster['sign_counts'].get(sign, 0) + len(faces)
                # Уточняем позицию как среднее
                cluster['position'] = (cluster['position'] + position) / 2.0

    # 2) Фильтруем кластеры по количеству граней и длине стены
    filtered_clusters = []
    for cluster in merged:
        faces = cluster['faces']
        # При привязке к runs допускаем минимум 1 грань (в прямоугольнике обычно 1 грань/знак)
        min_faces_needed = 1 if ('run_idx' in cluster and outline_runs) else min_face_count
        if len(faces) < min_faces_needed:
            continue

        # Длина стены: по центрам граней (обычный режим) или по самому run (режим runs)
        if 'run_idx' in cluster and outline_runs:
            try:
                r = outline_runs[cluster['run_idx']]
                span = float(r['span_max']) - float(r['span_min'])
            except Exception:
                # fallback — по центрам граней
                if cluster['axis'] == 'X':
                    vals = [f.calc_center_median().y for f in faces]
                else:
                    vals = [f.calc_center_median().x for f in faces]
                if not vals:
                    continue
                span = max(vals) - min(vals)
        else:
            if cluster['axis'] == 'X':
                vals = [f.calc_center_median().y for f in faces]
            else:  # 'Y'
                vals = [f.calc_center_median().x for f in faces]
            if not vals:
                continue
            span = max(vals) - min(vals)

        if span < min_wall_length:
            continue
        cluster['span'] = span
        filtered_clusters.append(cluster)

    print(f"    Кластеризация: групп={len(wall_groups)}, кластеров={len(merged)}, после фильтра={len(filtered_clusters)}")

    # 3) Нумерация финальных стен
    wall_info = {}
    wall_number = 0
    for cluster in filtered_clusters:
        axis = cluster['axis']
        faces = cluster['faces']
        # Робастное определение плоскости (позиции) и центра стены
        face_centers = [f.calc_center_median() for f in faces]
        xs = [c.x for c in face_centers]
        ys = [c.y for c in face_centers]
        zs = [c.z for c in face_centers]

        def _median(vals):
            s = sorted(vals)
            n = len(s)
            if n == 0:
                return 0.0
            mid = n // 2
            if n % 2 == 1:
                return s[mid]
            return 0.5 * (s[mid-1] + s[mid])

        def _percentile(vals, q):
            """Квантиль q (0..1) по упорядоченному списку."""
            if not vals:
                return 0.0
            s = sorted(vals)
            n = len(s)
            if n == 1:
                return s[0]
            i = q * (n - 1)
            lo = int(i)
            hi = min(lo + 1, n - 1)
            frac = i - lo
            return s[lo] * (1.0 - frac) + s[hi] * frac

        if axis == 'X':
            # Плоскость x ≈ const, длина по Y
            plane = _median(xs)
            # Усеченные границы (10..90 перцентиль) по длине, чтобы игнорировать выбросы
            if ys:
                y_lo = _percentile(ys, 0.10)
                y_hi = _percentile(ys, 0.90)
            else:
                y_lo = y_hi = 0.0
            cx, cy = plane, 0.5 * (y_lo + y_hi)
        else:  # 'Y'
            # Плоскость y ≈ const, длина по X
            plane = _median(ys)
            if xs:
                x_lo = _percentile(xs, 0.10)
                x_hi = _percentile(xs, 0.90)
            else:
                x_lo = x_hi = 0.0
            cx, cy = 0.5 * (x_lo + x_hi), plane

        cz = sum(zs) / len(zs) if zs else WALL_HEIGHT * 0.5
        position = plane
        # Выбираем направление по знаку с большим числом граней
        sign = 1
        if cluster['sign_counts']:
            sign = max(cluster['sign_counts'].items(), key=lambda kv: kv[1])[0]

        for f in faces:
            f[wall_num_layer] = wall_number

        direction = (sign, 0) if axis == 'X' else (0, sign)

        wall_info[wall_number] = {
            'direction': direction,
            'position': position,
            'axis': axis,
            'face_count': len(faces),
            'center': (cx, cy, cz),
            'sign': '+' if sign > 0 else '-',
        }
        # Если кластер соответствует детальному run — добавим служебные поля
        if 'run_idx' in cluster:
            ri = int(cluster['run_idx'])
            wall_info[wall_number]['run_idx'] = ri
            try:
                r = outline_runs[ri]
                wall_info[wall_number]['start_junction_id'] = r.get('start_junction_id')
                wall_info[wall_number]['end_junction_id'] = r.get('end_junction_id')
            except Exception:
                pass
        wall_number += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"    Стены: {len(wall_info)} (с доминантной нормалью граней: {faces_with_dom}, без: {faces_without_dom})")
    for wn, info in wall_info.items():
        extra = ""
        if 'run_idx' in info:
            sj = info.get('start_junction_id')
            ej = info.get('end_junction_id')
            extra = f", run={info['run_idx']} (J{sj}→J{ej})"
        print(f"      Стена #{wn}: ось {info['axis']}{info['sign']}, позиция={info['position']:.2f}м, граней={info['face_count']}{extra}")
    return wall_info


def create_red_material():
    name = "RedLabelMaterial"
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    # diffuse для экспорта
    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.5
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.5
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def create_wall_number_labels(wall_info, red_material, label_size=0.5):
    labels = []
    print(f"    Создание меток стен: {len(wall_info)}")
    # Все метки складываем в отдельную коллекцию
    labels_collection = get_or_create_collection("WALL_LABELS")
    for wall_num, info in wall_info.items():
        cx, cy, cz = info['center']
        text_curve = bpy.data.curves.new(name=f"WallLabel_{wall_num}", type='FONT')
        text_curve.body = str(wall_num)
        text_curve.size = label_size
        text_curve.align_x = 'CENTER'
        text_curve.align_y = 'CENTER'
        text_obj = bpy.data.objects.new(f"Wall_Number_{wall_num}", text_curve)
        # Линкуем метку в специальную коллекцию
        labels_collection.objects.link(text_obj)
        # Размещаем метку над стеной: чуть выше верхней кромки стены
        text_obj.location = (cx, cy, WALL_HEIGHT + LABEL_Z_OFFSET)
        if text_obj.data.materials:
            text_obj.data.materials[0] = red_material
        else:
            text_obj.data.materials.append(red_material)
        labels.append(text_obj)
    print(f"    ✅ Меток создано: {len(labels)}")
    return labels


def convert_text_labels_to_mesh(labels):
    """Конвертирует текстовые метки в mesh, чтобы они попали в OBJ экспорт."""
    converted = []
    for obj in labels:
        if obj.type == 'FONT':
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            try:
                bpy.ops.object.convert(target='MESH', keep_original=False)
                converted.append(obj)
            except Exception as e:
                print(f"    ⚠️  Не удалось конвертировать метку {obj.name} в mesh: {e}")
        else:
            converted.append(obj)
    print(f"    Меток для экспорта (mesh): {len(converted)}")
    return converted


def save_wall_normals_to_json(wall_info, path=OUTPUT_NORMALS_JSON):
    """Сохраняет нормали и информацию о стенах в JSON файл."""
    payload = {}
    for wall_num, info in wall_info.items():
        payload[str(wall_num)] = {
            'direction': [float(info['direction'][0]), float(info['direction'][1]), 0.0],
            'axis': info['axis'],
            'sign': info['sign'],
            'position': float(info['position']),
            'center': [float(c) for c in info['center']],
            'face_count': int(info['face_count'])
        }
        # Служебные поля, если присутствуют
        if 'run_idx' in info:
            payload[str(wall_num)]['run_idx'] = int(info['run_idx'])
        if 'start_junction_id' in info:
            payload[str(wall_num)]['start_junction_id'] = info['start_junction_id']
        if 'end_junction_id' in info:
            payload[str(wall_num)]['end_junction_id'] = info['end_junction_id']
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"    ✅ Нормали стен сохранены: {path}")
    except Exception as e:
        print(f"    ❌ Ошибка записи нормалей в JSON: {e}")


def compute_doors_near_windows(openings,
                               scale_factor=SCALE_FACTOR,
                               wall_thickness_px=WALL_THICKNESS_PX):
    """Возвращает множество id дверей, которые ближе половины толщины стены к окну.

    Критерий:
      - одинаковая ориентация (vertical/horizontal)
      - близость по плоскости стены (|plane_d - plane_w| <= толщина стены)
      - зазор вдоль оси протяженности <= 0.5 * толщина стены
    """
    thickness_m = float(wall_thickness_px) * float(scale_factor)
    half_thick = 0.5 * thickness_m
    plane_tol = thickness_m

    def to_entry(op):
        bbox = op.get('bbox') or {}
        ori = op.get('orientation', 'horizontal')
        x, y = float(bbox.get('x', 0.0)) * scale_factor, float(bbox.get('y', 0.0)) * scale_factor
        w, h = float(bbox.get('width', 0.0)) * scale_factor, float(bbox.get('height', 0.0)) * scale_factor
        if ori == 'vertical':
            plane = x + 0.5 * w
            smin, smax = y, y + h
        else:
            plane = y + 0.5 * h
            smin, smax = x, x + w
        return {
            'id': op.get('id'),
            'type': (op.get('type') or '').lower(),
            'orientation': ori,
            'plane': plane,
            'smin': min(smin, smax),
            'smax': max(smin, smax),
        }

    entries = [to_entry(o) for o in openings if o.get('bbox')]
    doors = [e for e in entries if e['type'] == 'door']
    windows = [e for e in entries if e['type'] == 'window']

    def interval_gap(a_min, a_max, b_min, b_max):
        if a_max < b_min:
            return b_min - a_max
        if b_max < a_min:
            return a_min - b_max
        return 0.0

    near = set()
    for d in doors:
        for w in windows:
            if d['orientation'] != w['orientation']:
                continue
            if abs(d['plane'] - w['plane']) > plane_tol:
                continue
            gap = interval_gap(d['smin'], d['smax'], w['smin'], w['smax'])
            if gap <= half_thick + 1e-6:
                near.add(d['id'])
                break
    print(f"    Близкие к окнам двери (<= 0.5 толщины): {len(near)}")
    return near


def create_highlight_material(name="WallFacesHighlight", color=(0.1, 0.9, 0.2, 1.0)):
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
        # обновим цвет
        mat.diffuse_color = color
        if mat.use_nodes and 'Principled BSDF' in mat.node_tree.nodes:
            mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.diffuse_color = color
    try:
        mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
    except Exception:
        pass
    return mat


def get_or_create_material(name, color=(1.0, 1.0, 1.0, 1.0)):
    """Создаёт или возвращает материал с заданным именем и цветом."""
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
    mat.diffuse_color = color
    try:
        if mat.use_nodes:
            nodes = mat.node_tree.nodes
            # Найдем Principled BSDF или создадим
            bsdf = nodes.get('Principled BSDF')
            if bsdf is None:
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                out = nodes.get('Material Output') or nodes.new(type='ShaderNodeOutputMaterial')
                mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
            bsdf.inputs['Base Color'].default_value = color
    except Exception:
        pass
    return mat


def compute_outline_centroid(outline_vertices):
    """Вычисляет центроид по вершинам контура (в метрах)."""
    if not outline_vertices:
        return 0.0, 0.0
    xs = [float(v['x']) * SCALE_FACTOR for v in outline_vertices]
    ys = [float(v['y']) * SCALE_FACTOR for v in outline_vertices]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return cx, cy


def ensure_brick_texture_image(texture_path=None,
                               width=1024,
                               height=1024,
                               brick_w=54,
                               brick_h=24,
                               mortar=10):
    """Генерирует тайловую текстуру кирпича и сохраняет как JPEG, если файл отсутствует.

    Подбирает размеры так, чтобы изображение было тайловым по краям:
    - Горизонтальный период: brick_w + mortar (по ширине)
    - Вертикальный период (2 ряда): 2 * (brick_h + mortar)
    """
    if texture_path is None:
        texture_path = os.path.join(SCRIPT_DIR, "brick_texture.jpg")

    try:
        if os.path.exists(texture_path):
            return texture_path

        # подгоняем размеры до кратности периодам
        period_x = max(8, int(brick_w + mortar))
        period_y = max(8, int(2 * (brick_h + mortar)))
        width_adj = (width // period_x) * period_x
        height_adj = (height // period_y) * period_y
        if width_adj < period_x:
            width_adj = period_x
        if height_adj < period_y:
            height_adj = period_y

        img = bpy.data.images.new("BrickTextureGen", width=width_adj, height=height_adj)
        pixels = [0.0] * (width_adj * height_adj * 4)

        half_brick = brick_w // 2

        def hash01(ix, iy):
            # детерминированный «рандом» по кирпичу
            v = (ix * 928371 + iy * 523847 + 713) % 1000
            return v / 1000.0

        for y in range(height_adj):
            row_period = brick_h + mortar
            row_idx = (y // row_period) % 2  # чёт/нечет ряд
            y_in = y % row_period
            mortar_y = (y_in >= brick_h)
            for x in range(width_adj):
                x_shift = half_brick if row_idx == 1 else 0
                x_in = (x + x_shift) % (brick_w + mortar)
                mortar_x = (x_in >= brick_w)

                if mortar_x or mortar_y:
                    r, g, b = 0.82, 0.82, 0.80  # цвет шва
                else:
                    # индекс текущего кирпича в сетке
                    bx = ((x + x_shift) // (brick_w + mortar))
                    by = (y // (brick_h + mortar))
                    t = hash01(int(bx), int(by))
                    # плавная вариация оттенка
                    base1 = (0.63, 0.27, 0.18)
                    base2 = (0.55, 0.22, 0.12)
                    k = 0.35 * (t - 0.5)
                    r = max(0.0, min(1.0, (base1[0] * (1 - t) + base2[0] * t) + k))
                    g = max(0.0, min(1.0, (base1[1] * (1 - t) + base2[1] * t) + k * 0.5))
                    b = max(0.0, min(1.0, (base1[2] * (1 - t) + base2[2] * t) + k * 0.25))

                idx = (y * width_adj + x) * 4
                pixels[idx + 0] = r
                pixels[idx + 1] = g
                pixels[idx + 2] = b
                pixels[idx + 3] = 1.0

        img.pixels = pixels
        img.filepath_raw = texture_path
        img.file_format = 'JPEG'
        img.save()
        print(f"    ✅ Сгенерирована текстура кирпича: {texture_path} ({width_adj}x{height_adj})")
        return texture_path
    except Exception as e:
        print(f"    ⚠️  Не удалось сгенерировать brick_texture.jpg: {e}")
        return None


def get_or_create_brick_material(name="OuterWall_Brick_Procedural"):
    """Создает или возвращает процедурный материал кирпича для внешних стен.

    Использует ShaderNodeTexBrick + лёгкий bump и объектные координаты для
    бесшовного наложения без UV.
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    # Попытка использовать сгенерированную текстуру-изображение
    tex_path = ensure_brick_texture_image(os.path.join(SCRIPT_DIR, "brick_texture.jpg"))
    if tex_path and os.path.exists(tex_path):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Roughness'].default_value = 0.75
        bsdf.inputs['Specular'].default_value = 0.25

        texcoord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)

        img = nodes.new(type='ShaderNodeTexImage')
        try:
            image = bpy.data.images.load(tex_path)
        except Exception:
            image = None
        if image:
            img.image = image
        img.interpolation = 'Smart'
        img.projection = 'BOX'
        img.projection_blend = 0.05

        # Лёгкий bump из монохрома текстуры
        to_gray = nodes.new(type='ShaderNodeRGBToBW')
        bump = nodes.new(type='ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.1
        bump.inputs['Distance'].default_value = 1.0

        links.new(texcoord.outputs['Object'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], img.inputs['Vector'])
        links.new(img.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(img.outputs['Color'], to_gray.inputs['Color'])
        links.new(to_gray.outputs['Val'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        return mat

    # Fallback: процедурный кирпич
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.8
    bsdf.inputs['Specular'].default_value = 0.2

    texcoord = nodes.new(type='ShaderNodeTexCoord')
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (2.0, 2.0, 2.0)

    brick = nodes.new(type='ShaderNodeTexBrick')
    brick.inputs['Scale'].default_value = 3.0
    brick.inputs['Mortar Size'].default_value = 0.015
    brick.inputs['Mortar Smooth'].default_value = 0.60
    brick.inputs['Bias'].default_value = 0.0
    brick.inputs['Brick Width'].default_value = 0.28
    brick.inputs['Row Height'].default_value = 0.07
    brick.inputs['Color1'].default_value = (0.63, 0.27, 0.18, 1.0)
    brick.inputs['Color2'].default_value = (0.55, 0.22, 0.12, 1.0)
    brick.inputs['Mortar'].default_value = (0.82, 0.82, 0.80, 1.0)

    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 10.0
    noise.inputs['Detail'].default_value = 2.0
    noise.inputs['Roughness'].default_value = 0.5

    mix = nodes.new(type='ShaderNodeMixRGB')
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Fac'].default_value = 0.2

    invert = nodes.new(type='ShaderNodeInvert')
    bump = nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.12
    bump.inputs['Distance'].default_value = 0.8

    links.new(texcoord.outputs['Object'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], brick.inputs['Vector'])
    links.new(brick.outputs['Color'], mix.inputs['Color1'])
    links.new(noise.outputs['Color'], mix.inputs['Color2'])
    links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(brick.outputs['Fac'], invert.inputs['Color'])
    links.new(invert.outputs['Color'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def assign_wall_materials(obj, outline_vertices,
                          outer_color=(1.0, 1.0, 0.0, 1.0),  # не используется для кирпича
                          inner_color=(1.0, 1.0, 1.0, 1.0)):  # белый
    """Назначает материал фасадам: внешние — жёлтые, внутренние — белые.

    Робастный критерий: делаем небольшой шаг от центра грани вдоль её 2D-нормали
    и проверяем, остаётся ли точка внутри полигона внешнего контура здания.
      - если выходит за контур → грань внешняя (жёлтая)
      - иначе → внутренняя (белая)

    Верх/низ (нормали по Z) получают внутренний материал.
    """
    if obj is None or obj.type != 'MESH':
        return

    # Подготовим 2D-полигон контура
    outline_poly = [(float(v['x']) * SCALE_FACTOR, float(v['y']) * SCALE_FACTOR) for v in outline_vertices]
    if len(outline_poly) < 3:
        print("    ⚠️  Недостаточно вершин контура для определения внешних граней")
        return

    def point_in_polygon(pt, poly):
        # Алгоритм ray casting (луч вправо)
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x < xinters:
                    inside = not inside
        return inside

    # Материалы: внешний — кирпич, внутренний — белый
    outer_mat = get_or_create_brick_material("OuterWall_Brick_Procedural")
    inner_mat = get_or_create_material("InnerWall_White", inner_color)

    mats = obj.data.materials
    # Убедимся, что оба материала есть в слотах
    name_to_index = {m.name: i for i, m in enumerate(mats)}
    if outer_mat.name not in name_to_index:
        mats.append(outer_mat)
        name_to_index[outer_mat.name] = len(mats) - 1
    if inner_mat.name not in name_to_index:
        mats.append(inner_mat)
        name_to_index[inner_mat.name] = len(mats) - 1

    outer_idx = name_to_index[outer_mat.name]
    inner_idx = name_to_index[inner_mat.name]

    # Присваиваем материал полигонам
    eps = max(WALL_THICKNESS_PX * SCALE_FACTOR * 0.45, 0.05)  # ≈ половина толщины, но не < 5см
    # Запомним, какие грани внешние, чтобы проставить тег 'external'
    poly_is_external = [False] * len(obj.data.polygons)
    for poly in obj.data.polygons:
        n = poly.normal
        # Пропустим верх/низ (нормаль почти по Z)
        if abs(n.z) > 0.5:
            poly.material_index = inner_idx
            continue
        center = obj.matrix_world @ poly.center if obj.parent else poly.center
        cx, cy = float(center.x), float(center.y)
        nx, ny = float(n.x), float(n.y)
        # нормируем 2D-нормаль
        L = (nx * nx + ny * ny) ** 0.5
        if L < 1e-6:
            poly.material_index = inner_idx
            continue
        nx /= L
        ny /= L
        # точка вне вдоль нормали
        px_out = (cx + nx * eps, cy + ny * eps)
        is_outside = not point_in_polygon(px_out, outline_poly)
        poly.material_index = outer_idx if is_outside else inner_idx
        poly_is_external[poly.index] = bool(is_outside)

    # Проставляем тег 'external' на уровне граней (face layer)
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        ext_layer = bm.faces.layers.int.get("external")
        if ext_layer is None:
            ext_layer = bm.faces.layers.int.new("external")
        for f in bm.faces:
            flag = 1 if (f.index < len(poly_is_external) and poly_is_external[f.index]) else 0
            f[ext_layer] = flag
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        tagged = sum(1 for v in poly_is_external if v)
        print(f"    ✅ Материалы назначены и тег 'external' проставлен: {tagged} граней внешние")
    except Exception as e:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        print(f"    ⚠️  Не удалось проставить тег 'external' на гранях: {e}")


def visualize_wall_faces(src_obj, wall_number, collection_name="WALLS_DEBUG"):
    """Создает отдельный объект только с гранями указанной стены (по wall_number)."""
    # Дублируем объект и его меш
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.name = f"Wall_Faces_{wall_number}"
    bpy.context.scene.collection.objects.link(obj)

    # Оставляем только нужные грани в дубликате
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    layer = bm.faces.layers.int.get("wall_number")
    if layer is None:
        bpy.ops.object.mode_set(mode='OBJECT')
        print("    ⚠️  Визуализация: слой 'wall_number' не найден")
        return None

    # Выбираем все грани, КРОМЕ нужного номера — их удалим
    for f in bm.faces:
        f.select = (f[layer] != wall_number)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.delete(type='FACE')

    # Очистка
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Применяем яркий материал
    hl = create_highlight_material()
    obj.data.materials.clear()
    obj.data.materials.append(hl)

    # Перемещаем в debug-коллекцию
    debug_coll = get_or_create_collection(collection_name)
    # убрать из всех коллекций и добавить в debug
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    debug_coll.objects.link(obj)

    print(f"    ✅ Визуализация: создан объект {obj.name} в коллекции {collection_name}")
    return obj


def import_obj_to_collection(obj_path,
                             target_collection="FLOOR3D_IMPORTED",
                             wall_obj=None):
    """Импортирует OBJ и переносит объекты в отдельную коллекцию.

    Новый фильтр: импортируем только MESH-объекты, чья XY-середина НЕ лежит на основании
    меша внешних стен (`Outline_Walls`).
    - light/camera/empty и прочие не-MESH исключаются всегда
    - если `wall_obj` не передан, ищем объект по имени "Outline_Walls"; если не нашли —
      пропускаем фильтрацию по основанию и переносим все MESH (осторожно)
    """
    if not os.path.exists(obj_path):
        print(f"    ⚠️ OBJ не найден: {obj_path}")
        return []

    before = set(o.name for o in bpy.data.objects)

    # Импорт (Blender 4.x и 3.x)
    try:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='NEGATIVE_Z', up_axis='Y')
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y')

    imported = [o for o in bpy.data.objects if o.name not in before]
    print(f"    Импортировано объектов из OBJ: {len(imported)}")

    # Подготовим список XY-полигонов основания Outline_Walls
    def get_wall_base_polys(wobj):
        base_polys = []
        if wobj is None or wobj.type != 'MESH':
            return base_polys
        mesh = wobj.data
        mw = wobj.matrix_world
        for poly in mesh.polygons:
            # Берём нижние/верхние фаски; фильтруем около Z=0 для основания
            if abs(poly.normal.z) < 0.5:
                continue
            zs = [ (mw @ mesh.vertices[i].co).z for i in poly.vertices ]
            min_z = min(zs)
            if abs(min_z - 0.0) > 1e-3:
                continue
            pts = [mw @ mesh.vertices[i].co for i in poly.vertices]
            base_polys.append([(float(p.x), float(p.y)) for p in pts])
        return base_polys

    def pt_in_poly(pt, poly):
        # ray casting в XY
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x < xinters:
                    inside = not inside
        return inside

    def get_xy_center(obj):
        # берём центр AABB в мировых координатах
        bb = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        xs = [p.x for p in bb]
        ys = [p.y for p in bb]
        return (float((min(xs)+max(xs))/2.0), float((min(ys)+max(ys))/2.0))

    wref = wall_obj or bpy.data.objects.get("Outline_Walls")
    base_polys = get_wall_base_polys(wref)
    if base_polys:
        print(f"    Базовых полигонов основания: {len(base_polys)}")
    else:
        print("    ⚠️ Основание Outline_Walls не найдено — фильтр по основанию отключён")

    def keep(obj):
        # Только меши (никакого света/камер/эмпти)
        if obj.type != 'MESH':
            return False
        # Исключаем по имени только Outline_Merged (без учёта регистра)
        name = obj.name.lower()
        if 'outline_merged' in name:
            return False
        if not base_polys:
            return True
        cx, cy = get_xy_center(obj)
        # Если центр попадает в любой полигон основания — не импортируем
        for poly in base_polys:
            if pt_in_poly((cx, cy), poly):
                return False
        return True

    to_collect = [o for o in imported if keep(o)]
    to_delete = [o for o in imported if o not in to_collect]
    print(f"    В коллекцию уйдёт: {len(to_collect)} (удалено: {len(to_delete)})")

    coll = get_or_create_collection(target_collection)
    for obj in to_collect:
        # убрать из всех коллекций
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        # поместить в целевую
        coll.objects.link(obj)

    # Полностью удалить исключённые объекты из сцены/данных
    for obj in to_delete:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    print(f"    ✅ Объекты перенесены в коллекцию: {target_collection}")
    return to_collect


def export_obj(objs, output_path):
    """Экспортирует один объект или список объектов в OBJ."""
    bpy.ops.object.select_all(action='DESELECT')

    if isinstance(objs, list):
        to_export = [o for o in objs if o is not None]
        for o in to_export:
            o.select_set(True)
        if to_export:
            bpy.context.view_layer.objects.active = to_export[0]
    else:
        objs.select_set(True)
        bpy.context.view_layer.objects.active = objs

    try:
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=True,
            export_materials=True,
            export_normals=True,
            export_uv=True,
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=output_path,
            use_selection=True,
            use_materials=True,
            use_normals=True,
            use_uvs=True,
        )

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"    Экспортирован OBJ: {output_path} ({size / 1024:.1f} KB)")
    else:
        print("    ⚠️  Экспорт OBJ не создал файл.")


def main():
    print("=" * 70)
    print("СОЗДАНИЕ КОНТУРА С ПРОЁМАМИ - НАДЕЖНАЯ ВЕРСИЯ")
    print("=" * 70)
    print(f"JSON: {JSON_PATH}")
    print()

    clear_scene()
    data = load_json_data(JSON_PATH)
    # Жесткая проверка наличия building_outline
    if ("building_outline" not in data) or (not data["building_outline"]) or ("vertices" not in data["building_outline"]) or (not data["building_outline"]["vertices"]):
        raise RuntimeError("В JSON отсутствует корректный 'building_outline' (нет vertices)")

    vertices, segments_with_bbox, edges_without_bbox = get_outline_junctions_and_segments(data)
    # Строим основной контур стен по угловым вершинам + bbox-подсказкам
    rectangles = create_rectangles(data)

    wall_obj = create_wall_mesh(rectangles, wall_height=WALL_HEIGHT)
    merge_duplicate_vertices(wall_obj, MERGE_DISTANCE)
    recalc_normals(wall_obj)

    # Идентификация стен + нумерация и метки (до создания проёмов и фундамента)
    print("\n[WALLS] Идентификация, нумерация и метки (до проёмов)")
    # Участки только от угла к углу (corner=1 → corner=1)
    outline_runs = build_outline_runs(vertices)
    wall_info = identify_and_number_walls_from_mesh(
        wall_obj,
        position_tolerance=0.15,
        normal_threshold=0.7,
        outline_runs=outline_runs,
    )
    print(f"    Найдено стен: {len(wall_info)}")
    for wn, info in wall_info.items():
        nx, ny = info['direction']
        print(f"    Стена #{wn}: центр={info['center']}, нормаль=( {nx:.1f}, {ny:.1f}, 0.0 )")

    red_mat = create_red_material()
    labels = create_wall_number_labels(wall_info, red_mat, label_size=0.5)

    # Сохраняем нормали стен в отдельный JSON
    save_wall_normals_to_json(wall_info, OUTPUT_NORMALS_JSON)

    # Визуализация граней конкретной стены (для отладки)
    if isinstance(DEBUG_WALL_TO_VISUALIZE, int):
        try:
            visualize_wall_faces(wall_obj, DEBUG_WALL_TO_VISUALIZE, collection_name="WALLS_DEBUG")
        except Exception as e:
            print(f"    ⚠️  Не удалось визуализировать стену {DEBUG_WALL_TO_VISUALIZE}: {e}")

    outline_ids = {v["junction_id"] for v in vertices}
    openings = collect_outline_openings(data, outline_ids)
    print(f"    Проёмов на контуре: {len(openings)}")

    opening_heights = load_opening_heights_from_obj(OBJ_HEIGHT_SOURCE_PATH)

    opening_entries = create_opening_cutters(openings, opening_heights)
    
    # Применяем булевы операции
    apply_boolean_operations(wall_obj, opening_entries)

    # Заполняем проёмы новыми объектами (окна — голубые, двери — тёмно-коричневые)
    try:
        # Определим двери, близкие к окнам (<= 0.5 толщины стены)
        doors_near = compute_doors_near_windows(openings)
        create_opening_fills(openings, opening_heights, doors_near_windows=doors_near)
    except Exception as e:
        print(f"    ⚠️ Не удалось создать заполнители проёмов: {e}")

    # Финальная очистка
    merge_duplicate_vertices(wall_obj, 0.001)
    recalc_normals(wall_obj)
    cleanup_mesh(wall_obj)

    # Назначаем материалы стенам: внешний контур — жёлтый, внутренние — белый
    try:
        assign_wall_materials(wall_obj, vertices)
    except Exception as e:
        print(f"    ⚠️  Не удалось назначить материалы стенам: {e}")


    # Импорт 3D-объектов из OBJ и сбор в отдельную коллекцию:
    # берём только те объекты, чья XY-середина НЕ лежит на основании Outline_Walls
    try:
        import_obj_to_collection(OBJ_HEIGHT_SOURCE_PATH, target_collection="FLOOR3D_IMPORTED",
                                 wall_obj=wall_obj)
    except Exception as e:
        print(f"    ⚠️ Ошибка импорта OBJ в коллекцию: {e}")

    # Создаем фундамент, если есть данные
    foundation_obj = None
    if 'foundation' in data:
        print("\n[FOUNDATION] Создание фундамента из JSON")
        foundation_obj = create_foundation(
            data['foundation'],
            z_offset=FOUNDATION_Z_OFFSET,
            thickness=FOUNDATION_THICKNESS,
            scale_factor=SCALE_FACTOR,
        )

    print()
    print("[EXPORT] Экспорт OBJ")
    objects_to_export = [wall_obj]
    if EXPORT_LABELS_IN_OBJ:
        # Конвертируем метки в mesh и добавим в экспорт
        try:
            label_meshes = convert_text_labels_to_mesh(labels)
            objects_to_export.extend(label_meshes)
        except Exception as e:
            print(f"    ⚠️  Ошибка при подготовке меток к экспорту: {e}")
    if foundation_obj is not None:
        objects_to_export.append(foundation_obj)
    export_obj(objects_to_export, OUTPUT_OBJ)

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print(f"Стена: {wall_obj.name}")
    print(f"Вершин: {len(wall_obj.data.vertices)}")
    print(f"Граней: {len(wall_obj.data.polygons)}")
    if foundation_obj is not None:
        print(f"Фундамент: {foundation_obj.name} ✅")
    else:
        print(f"Фундамент: отсутствует")
    print("Перекрытие: отключено")
    
    # Проверяем результат
    if len(wall_obj.data.vertices) > 50:  # Должно быть достаточно вершин после вырезания
        print("✅ Проемы должны быть видны в стенах")
    else:
        print("❌ Возможно, проемы не создались - проверьте позиции вырезателей")


if __name__ == "__main__":
    # Проверяем, передан ли параметр --no-auto-run
    auto_run = "--no-auto-run" not in sys.argv

    if auto_run:
        # Применяем CLI переопределения путей
        jp, hp, op = _apply_cli_overrides(sys.argv)
        # Переопределяем глобальные пути
        JSON_PATH = jp
        OBJ_HEIGHT_SOURCE_PATH = hp
        OUTPUT_OBJ = op
        # Нормали: используем тот же префикс, что и у выходного OBJ
        out_dir = os.path.dirname(os.path.abspath(OUTPUT_OBJ))
        out_base = os.path.splitext(os.path.basename(OUTPUT_OBJ))[0]
        # Если имя заканчивается на _outline_with_openings, отсечём этот суффикс
        prefix_base = out_base[:-len("_outline_with_openings")] if out_base.endswith("_outline_with_openings") else out_base
        OUTPUT_NORMALS_JSON = os.path.join(out_dir, f"{prefix_base}_mesh_normals.json")

        print("=" * 70)
        print("АВТОМАТИЧЕСКИЙ ЗАПУСК СОЗДАНИЯ КОНТУРА С ПРОЁМАМИ")
        print("=" * 70)

        try:
            main()
            print("=" * 70)
            print("АВТОМАТИЧЕСКИЙ ЗАПУСК УСПЕШНО ЗАВЕРШЕН")
            print("=" * 70)
        except Exception as e:
            print(f"Ошибка при выполнении: {e}")
            print("=" * 70)
            print("АВТОМАТИЧЕСКИЙ ЗАПУСК ЗАВЕРШИЛСЯ С ОШИБКОЙ")
            print("=" * 70)
            sys.exit(1)
    else:
        print("Скрипт загружен без автоматического запуска (--no-auto-run)")
