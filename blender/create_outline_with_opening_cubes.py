#!/usr/bin/env python3
"""
Создаёт стены внешнего контура и кубы проёмов (окна/двери),
окрашенные в красный цвет. Булевы вырезы НЕ выполняет.

Минимальная версия, без экспорта, меток, материалов для стен и прочего.
Запускается внутри Blender: blender -b -P this_script.py -- --json <path>
"""

import bpy
import os
import json
import sys

# ---------------------------------
# Константы
# ---------------------------------
SCALE_FACTOR = 0.01        # 1 px = 1 см
WALL_THICKNESS_PX = 22.0   # толщина стены в пикселях
WALL_HEIGHT = 3.0          # высота стены (м)
OPENING_WIDTH_MULTIPLIER = 1.95  # масштаб ширины проёмов (как в исходном скрипте)

WINDOW_BOTTOM = 0.65
WINDOW_TOP = 2.45
DOOR_BOTTOM = 0.10
DOOR_TOP = 2.45

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------
# CLI и пути
# ---------------------------------
def _derive_defaults_from_json(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    prefix = base_name.split('_')[0]
    return {
        'heights_obj': os.path.join(base_dir, f"{prefix}_wall_coordinates_inverted_3d.obj"),
    }


def _apply_cli_overrides(argv):
    """Парсит CLI аргументы: --json, --heights-obj.
    Возвращает кортеж (json_path, heights_obj). --json обязателен.
    """
    json_path = None
    heights_obj = None
    for i, a in enumerate(argv):
        if a == '--json' and i + 1 < len(argv):
            json_path = argv[i + 1]
        elif a.startswith('--json='):
            json_path = a.split('=', 1)[1]
        elif a == '--heights-obj' and i + 1 < len(argv):
            heights_obj = argv[i + 1]
        elif a.startswith('--heights-obj='):
            heights_obj = a.split('=', 1)[1]

    if not json_path:
        print("Ошибка: не указан путь к JSON. Использование: --json <path> [--heights-obj <path>]")
        sys.exit(1)
    if not os.path.exists(json_path):
        print(f"Ошибка: JSON файл не найден: {json_path}")
        sys.exit(1)

    derived = _derive_defaults_from_json(json_path)
    heights_obj = heights_obj or derived['heights_obj']
    return json_path, heights_obj


# ---------------------------------
# Утилиты/данные
# ---------------------------------
def get_or_create_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_opening_heights_from_obj(path):
    """Читает высоты для Fill_Below_/Fill_Above_ из OBJ (если он существует).
    Возвращает словарь {opening_id: (bottom, top)}.
    """
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


# ---------------------------------
# Построение стен (прямоугольники)
# ---------------------------------

def _filter_overlapping_rectangles(rectangles, data):
    """
    Фильтрует прямоугольники стен, которые ПОЛНОСТЬЮ покрываются
    сегментами от проемов (openings) на той же линии.

    Проблема: между угловыми вершинами (например, J2 и J5) создается
    прямоугольник стены, но в той же области уже есть 2 сегмента от window_1
    (левый и правый). Эти 3 прямоугольника ПЕРЕКРЫВАЮТСЯ, создавая несколько
    слоев стен, и Boolean вырезает только один.

    Решение: если прямоугольник между углами ГОРИЗОНТАЛЬНЫЙ, и в той же
    горизонтальной полосе есть сегменты от openings, которые ПОЛНОСТЬЮ
    покрывают длину этого прямоугольника, удаляем его.
    """

    def rect_bounds(rect):
        """Возвращает (x_min, x_max, y_min, y_max) для прямоугольника"""
        corners = rect['corners']
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs), max(xs), min(ys), max(ys))

    # Соберем все ГОРИЗОНТАЛЬНЫЕ сегменты от openings
    horizontal_opening_segments = []
    for seg in data.get('wall_segments_from_openings', []):
        if seg.get('orientation') != 'horizontal':
            continue

        bbox = seg.get('bbox')
        if not bbox:
            continue

        x = float(bbox.get('x', 0)) * SCALE_FACTOR
        y = float(bbox.get('y', 0)) * SCALE_FACTOR
        w = float(bbox.get('width', 0)) * SCALE_FACTOR
        h = float(bbox.get('height', 0)) * SCALE_FACTOR

        horizontal_opening_segments.append({
            'x_min': x,
            'x_max': x + w,
            'y_min': y,
            'y_max': y + h,
            'opening_id': seg.get('opening_id')
        })

    # Фильтруем ТОЛЬКО горизонтальные прямоугольники между углами
    filtered = []
    removed_count = 0

    for rect in rectangles:
        # Проверяем только горизонтальные прямоугольники между углами
        if rect.get('orientation') != 'horizontal':
            filtered.append(rect)
            continue

        rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect_bounds(rect)

        # Найдем все opening сегменты в той же горизонтальной полосе
        y_tolerance = 0.05  # 5 см
        same_line_segments = []

        for seg in horizontal_opening_segments:
            # Проверим, что сегмент в той же горизонтальной полосе
            seg_y_center = (seg['y_min'] + seg['y_max']) / 2
            rect_y_center = (rect_y_min + rect_y_max) / 2

            if abs(seg_y_center - rect_y_center) < y_tolerance:
                same_line_segments.append(seg)

        if not same_line_segments:
            # Нет opening сегментов на той же линии - оставляем прямоугольник
            filtered.append(rect)
            continue

        # Проверим, покрывают ли opening сегменты всю длину прямоугольника
        # Соберем все X-интервалы от opening сегментов
        x_intervals = [(s['x_min'], s['x_max']) for s in same_line_segments]
        x_intervals.sort()

        # Объединим перекрывающиеся интервалы
        merged_intervals = []
        for start, end in x_intervals:
            if merged_intervals and start <= merged_intervals[-1][1]:
                # Расширим последний интервал
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
            else:
                merged_intervals.append((start, end))

        # Проверим, покрывают ли объединенные интервалы весь прямоугольник
        total_coverage = sum(end - start for start, end in merged_intervals)
        rect_length = rect_x_max - rect_x_min
        coverage_ratio = total_coverage / rect_length if rect_length > 0 else 0

        if coverage_ratio > 0.90:  # 90% покрытия = дубликат
            removed_count += 1
            opening_ids = set(s['opening_id'] for s in same_line_segments)
            print(f"    ⚠️  Удален дублирующийся прямоугольник: {rect.get('id')}")
            print(f"       (покрыт на {coverage_ratio*100:.1f}% сегментами от {', '.join(opening_ids)})")
        else:
            filtered.append(rect)

    if removed_count > 0:
        print(f"    ✓ Отфильтровано {removed_count} дублирующихся горизонтальных прямоугольников")

    return filtered


def create_rectangles(data):
    """Создаёт прямоугольники стен по угловым вершинам building_outline, используя bbox-подсказки."""

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

    # Фильтрация дублирующихся прямоугольников
    # Проблема: между углами J2 и J5 создается прямоугольник, но в той же области
    # уже есть сегменты от window_1. Это создает перекрывающиеся слои стен.
    filtered_rectangles = _filter_overlapping_rectangles(rectangles, data)

    print(f"    Всего прямоугольников стен: {len(filtered_rectangles)} (из {len(rectangles)}, отфильтровано {len(rectangles) - len(filtered_rectangles)})")
    return filtered_rectangles


def create_wall_mesh(rectangles, wall_height=WALL_HEIGHT):
    vertices = []
    faces = []
    offset = 0
    for rect in rectangles:
        lower = [(x, y, 0.0) for (x, y) in rect["corners"]]
        upper = [(x, y, wall_height) for (x, y) in rect["corners"]]
        vertices.extend(lower + upper)
        faces.append([offset + 0, offset + 1, offset + 2, offset + 3])  # bottom
        faces.append([offset + 4, offset + 5, offset + 6, offset + 7])  # top
        for i in range(4):
            j = (i + 1) % 4
            faces.append([offset + i, offset + j, offset + 4 + j, offset + 4 + i])
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


# ---------------------------------
# Проёмы
# ---------------------------------
def collect_outline_openings(data, outline_ids):
    openings = []
    for opening in data.get("openings", []):
        edge_junctions = opening.get("edge_junctions", [])
        if edge_junctions and all(ej["junction_id"] in outline_ids for ej in edge_junctions):
            openings.append(opening)
    return openings


def opening_bounds(opening):
    if (opening.get("type") or "").lower() == "door":
        return DOOR_BOTTOM, DOOR_TOP
    return WINDOW_BOTTOM, WINDOW_TOP


def create_red_material():
    name = "OpeningCube_Red"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
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


def create_opening_cubes(openings, opening_heights,
                         wall_thickness_px=WALL_THICKNESS_PX,
                         scale_factor=SCALE_FACTOR):
    """Создаёт для каждого проёма куб (как вырезатель), окрашенный в красный цвет.
    Кубы складываются в коллекцию OPENINGS_DEBUG. Возврат: список объектов.
    """
    created = []
    debug_collection = get_or_create_collection("OPENINGS_DEBUG")
    red_mat = create_red_material()

    for opening in openings:
        bbox = opening.get("bbox")
        if not bbox:
            continue

        orientation = opening.get("orientation", "horizontal")

        # Габариты по XY — МАКСИМАЛЬНЫЙ запас для гарантированного прорезания
        # Убираем все ограничители, делаем куб ОЧЕНЬ большим
        thickness_m = float(wall_thickness_px) * float(scale_factor)
        if orientation == "vertical":
            width = thickness_m * 10.0  # Увеличено до 10x для ГАРАНТИИ Boolean
            depth = float(bbox["height"]) * scale_factor * OPENING_WIDTH_MULTIPLIER
        else:
            width = float(bbox["width"]) * scale_factor * OPENING_WIDTH_MULTIPLIER
            depth = thickness_m * 10.0  # Увеличено до 10x для ГАРАНТИИ Boolean

        # Центр в XY
        x_center = (bbox["x"] + bbox["width"] / 2.0) * scale_factor
        y_center = (bbox["y"] + bbox["height"] / 2.0) * scale_factor

        # Высоты Z: если есть в OBJ — делаем х2 (как у cutters), иначе 1.2x от дефолта
        bottom_top = opening_heights.get(opening.get("id")) if opening_heights else None
        if bottom_top:
            bottom, top = bottom_top
            height = (float(top) - float(bottom)) * 2.0
            z_center = (float(bottom) + float(top)) * 0.5
        else:
            bottom, top = opening_bounds(opening)
            height = (float(top) - float(bottom)) * 1.2
            z_center = (float(bottom) + float(top)) * 0.5

        # Создаём куб
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, y_center, z_center))
        cube_obj = bpy.context.active_object
        cube_obj.name = f"Opening_Cube_{opening.get('id', 'unknown')}"
        cube_obj.scale = (width / 2.0, depth / 2.0, height / 2.0)

        # Назначаем материал
        cube_obj.data.materials.clear()
        cube_obj.data.materials.append(red_mat)

        # Перносим в коллекцию OPENINGS_DEBUG (и убираем из корневой сцены)
        debug_collection.objects.link(cube_obj)
        try:
            bpy.context.scene.collection.objects.unlink(cube_obj)
        except Exception:
            pass

        created.append(cube_obj)

    print(f"    Создано кубов проёмов: {len(created)}")
    return created


# ---------------------------------
# Основной сценарий
# ---------------------------------
def main(json_path, heights_obj_path):
    print("=" * 70)
    print("СОЗДАНИЕ СТЕН И КРАСНЫХ КУБОВ ПРОЁМОВ (без вырезания)")
    print("=" * 70)

    data = load_json_data(json_path)

    # Стены
    rectangles = create_rectangles(data)
    wall_obj = create_wall_mesh(rectangles, wall_height=WALL_HEIGHT)

    # Проёмы на внешнем контуре
    vertices = data.get("building_outline", {}).get("vertices", [])
    outline_ids = {v["junction_id"] for v in vertices}
    openings = collect_outline_openings(data, outline_ids)
    print(f"    Проёмов на контуре: {len(openings)}")

    # Высоты
    opening_heights = load_opening_heights_from_obj(heights_obj_path)

    # Кубы
    create_opening_cubes(openings, opening_heights)

    print("— Готово. Стены созданы, кубы добавлены, вырезов нет.")
    print(f"Стена: {wall_obj.name}")


if __name__ == "__main__":
    # Автозапуск: ожидаем --json (обязателен)
    jp, hp = _apply_cli_overrides(sys.argv)
    main(jp, hp)

