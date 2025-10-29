#!/usr/bin/env python3
"""
Создаёт цельный контур внешних стен с отверстиями окон и дверей.
Надежная версия с булевыми операциями.
"""

import bpy
import os
import json

# ---------------------------------
# Константы
# ---------------------------------
SCALE_FACTOR = 0.01        # 1 px = 1 см
WALL_THICKNESS_PX = 22.0   # толщина стены в пикселях
WALL_HEIGHT = 3.0          # высота стены в метрах
MERGE_DISTANCE = 0.005     # допуск для Merge by Distance (5 мм)
CUT_MARGIN = 0.05          # запас для гарантии пересечения
OPENING_WIDTH_MULTIPLIER = 1.95  # масштаб ширины проёмов

WINDOW_BOTTOM = 0.65
WINDOW_TOP = 2.45
DOOR_BOTTOM = 0.10
DOOR_TOP = 2.45

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "wall_coordinates_inverted.json")
OUTPUT_OBJ = os.path.join(SCRIPT_DIR, "precise_building_outline_with_openings.obj")
OBJ_HEIGHT_SOURCE_PATH = os.path.join(SCRIPT_DIR, "wall_coordinates_inverted_3d.obj")

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


def create_rectangle_from_bbox(bbox, scale_factor=SCALE_FACTOR):
    x, y = bbox["x"], bbox["y"]
    w, h = bbox["width"], bbox["height"]
    return [
        (x * scale_factor, y * scale_factor),
        ((x + w) * scale_factor, y * scale_factor),
        ((x + w) * scale_factor, (y + h) * scale_factor),
        (x * scale_factor, (y + h) * scale_factor),
    ]


def create_rectangle_from_edge(edge, wall_thickness_px=WALL_THICKNESS_PX, scale_factor=SCALE_FACTOR):
    v1 = edge["start_vertex"]
    v2 = edge["end_vertex"]
    x1, y1 = v1["x"], v1["y"]
    x2, y2 = v2["x"], v2["y"]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    half = wall_thickness_px / 2.0

    if dx > dy:
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_mid = (y1 + y2) / 2.0
        corners = [
            (x_min, y_mid - half),
            (x_max, y_mid - half),
            (x_max, y_mid + half),
            (x_min, y_mid + half),
        ]
    else:
        y_min, y_max = min(y1, y2), max(y1, y2)
        x_mid = (x1 + x2) / 2.0
        corners = [
            (x_mid - half, y_min),
            (x_mid + half, y_min),
            (x_mid + half, y_max),
            (x_mid - half, y_max),
        ]

    return [(px * scale_factor, py * scale_factor) for (px, py) in corners]


def create_rectangles(segments_with_bbox, edges_without_bbox, junctions):
    rectangles = []

    for seg in segments_with_bbox:
        rectangles.append(
            {
                "corners": create_rectangle_from_bbox(seg["bbox"]),
                "id": seg.get("segment_id", "segment"),
            }
        )

    for edge in edges_without_bbox:
        rectangles.append(
            {
                "corners": create_rectangle_from_edge(edge),
                "id": f"edge_{edge['start_junction_id']}_{edge['end_junction_id']}",
            }
        )

    print(f"    Всего прямоугольников стен: {len(rectangles)}")
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


def export_obj(obj, output_path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=True,
            export_materials=False,
            export_normals=True,
            export_uv=False,
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=output_path,
            use_selection=True,
            use_materials=False,
            use_normals=True,
            use_uvs=False,
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

    vertices, segments_with_bbox, edges_without_bbox = get_outline_junctions_and_segments(data)
    rectangles = create_rectangles(segments_with_bbox, edges_without_bbox, data["junctions"])

    wall_obj = create_wall_mesh(rectangles, wall_height=WALL_HEIGHT)
    merge_duplicate_vertices(wall_obj, MERGE_DISTANCE)
    recalc_normals(wall_obj)

    outline_ids = {v["junction_id"] for v in vertices}
    openings = collect_outline_openings(data, outline_ids)
    print(f"    Проёмов на контуре: {len(openings)}")

    opening_heights = load_opening_heights_from_obj(OBJ_HEIGHT_SOURCE_PATH)

    opening_entries = create_opening_cutters(openings, opening_heights)
    
    # Применяем булевы операции
    apply_boolean_operations(wall_obj, opening_entries)

    # Финальная очистка
    merge_duplicate_vertices(wall_obj, 0.001)
    recalc_normals(wall_obj)
    cleanup_mesh(wall_obj)

    print()
    print("[EXPORT] Экспорт OBJ")
    export_obj(wall_obj, OUTPUT_OBJ)

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print(f"Стена: {wall_obj.name}")
    print(f"Вершин: {len(wall_obj.data.vertices)}")
    print(f"Граней: {len(wall_obj.data.polygons)}")
    
    # Проверяем результат
    if len(wall_obj.data.vertices) > 50:  # Должно быть достаточно вершин после вырезания
        print("✅ Проемы должны быть видны в стенах")
    else:
        print("❌ Возможно, проемы не создались - проверьте позиции вырезателей")


if __name__ == "__main__":
    main()
