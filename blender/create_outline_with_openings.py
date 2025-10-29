#!/usr/bin/env python3
"""
Создаёт цельный контур внешних стен с отверстиями окон и дверей.

Базируется на логике create_precise_outline_copy.py:
1. Формирует прямоугольные сегменты контура из JSON, объединяет их в один меш.
2. Экструдирует стены на высоту здания.
3. Создаёт вырезы для окон/дверей, при этом первые два окна обрабатываются разными методами:
   - первое окно (boolean-модификатор),
   - второе окно (bmesh.boolean),
   остальные — стандартным boolean после объединения.
Одновременно создаются wireframe-очертания проёмов для визуальной проверки расположения.

Запуск:
    blender --background --python create_outline_with_openings.py
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
CUT_MARGIN = 0.05          # дополнительный запас для булевых вырезов

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
    # удаляем пустые коллекции (кроме master)
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


def create_debug_outline(opening, dims, collection, color=(1.0, 0.2, 0.2, 1.0)):
    opening_id = opening.get("id", "unknown")
    bottom, top = dims["bottom"], dims["top"]
    width = dims["wire_width"]
    depth = dims["wire_depth"]
    x_center = dims["x_center"]
    y_center = dims["y_center"]
    orientation = dims["orientation"]

    if orientation == "horizontal":
        verts = [
            (x_center - width / 2.0, y_center + depth / 2.0 + 0.005, bottom),
            (x_center + width / 2.0, y_center + depth / 2.0 + 0.005, bottom),
            (x_center + width / 2.0, y_center + depth / 2.0 + 0.005, top),
            (x_center - width / 2.0, y_center + depth / 2.0 + 0.005, top),
        ]
    else:
        verts = [
            (x_center + width / 2.0 + 0.005, y_center - depth / 2.0, bottom),
            (x_center + width / 2.0 + 0.005, y_center + depth / 2.0, bottom),
            (x_center + width / 2.0 + 0.005, y_center + depth / 2.0, top),
            (x_center + width / 2.0 + 0.005, y_center - depth / 2.0, top),
        ]

    mesh = bpy.data.meshes.new(f"Opening_Debug_Mesh_{opening_id}")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()

    obj = bpy.data.objects.new(f"Opening_Debug_{opening_id}", mesh)
    obj.display_type = 'WIRE'
    obj.show_in_front = True
    obj.hide_render = True

    collection.objects.link(obj)

    mat_name = "OpeningDebugMaterial"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(mat_name)
        mat.diffuse_color = color
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    return obj


def create_opening_entries(openings, opening_heights):
    entries = []
    debug_collection = get_or_create_collection("OPENINGS_DEBUG")

    for opening in openings:
        bbox = opening.get("bbox")
        if not bbox:
            continue

        orientation = opening.get("orientation", "horizontal")
        x_center = (bbox["x"] + bbox["width"] / 2.0) * SCALE_FACTOR
        y_center = (bbox["y"] + bbox["height"] / 2.0) * SCALE_FACTOR

        wall_thickness_m = WALL_THICKNESS_PX * SCALE_FACTOR
        if orientation == "vertical":
            base_width = wall_thickness_m
            base_depth = max(bbox["height"] * SCALE_FACTOR, wall_thickness_m)
            size_x = wall_thickness_m * 2.0
            size_y = max(bbox["height"] * SCALE_FACTOR * 1.5, wall_thickness_m * 2.0)
        else:
            base_width = max(bbox["width"] * SCALE_FACTOR, wall_thickness_m)
            base_depth = wall_thickness_m
            size_x = max(bbox["width"] * SCALE_FACTOR * 1.5, wall_thickness_m * 2.0)
            size_y = wall_thickness_m * 2.0

        bottom_top = opening_heights.get(opening["id"]) if opening_heights else None
        if bottom_top:
            bottom, top = bottom_top
        else:
            bottom, top = opening_bounds(opening)

        height = max(top - bottom, 0.1)
        z_center = bottom + height / 2.0

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, y_center, z_center))
        cutter = bpy.context.active_object
        cutter.name = f"Opening_{opening.get('id', 'unknown')}"

        cutter.scale.x = base_width #size_x / 2.0 + CUT_MARGIN * 2
        cutter.scale.y = base_depth #size_y / 2.0 + CUT_MARGIN * 2
        cutter.scale.z = height / 2.0 + CUT_MARGIN

        actual_width = cutter.scale.x * 2.0
        actual_depth = cutter.scale.y * 2.0
        actual_height = cutter.scale.z * 2.0

        debug_mesh = bpy.data.meshes.new(f"{cutter.name}_WireMesh")
        debug_mesh.from_pydata(
            [
                (x_center - base_width / 2.0, y_center - base_depth / 2.0, bottom),
                (x_center + base_width / 2.0, y_center - base_depth / 2.0, bottom),
                (x_center + base_width / 2.0, y_center + base_depth / 2.0, bottom),
                (x_center - base_width / 2.0, y_center + base_depth / 2.0, bottom),
                (x_center - base_width / 2.0, y_center - base_depth / 2.0, top),
                (x_center + base_width / 2.0, y_center - base_depth / 2.0, top),
                (x_center + base_width / 2.0, y_center + base_depth / 2.0, top),
                (x_center - base_width / 2.0, y_center + base_depth / 2.0, top),
            ],
            [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ],
            []
        )
        debug_mesh.update()

        debug_copy = bpy.data.objects.new(f"{cutter.name}_Wire", debug_mesh)
        debug_copy.display_type = 'WIRE'
        debug_copy.show_in_front = True
        debug_copy.hide_render = True
        debug_collection.objects.link(debug_copy)

        dims = {
            "wire_width": base_width,
            "wire_depth": base_depth,
            "x_center": x_center,
            "y_center": y_center,
            "bottom": bottom,
            "top": top,
            "orientation": orientation,
            "bbox": bbox,
            "cutter_width": actual_width,
            "cutter_depth": actual_depth,
            "cutter_height": actual_height,
        }

        outline_obj = create_debug_outline(opening, dims, debug_collection)

        entries.append({
            "opening": opening,
            "cutter": cutter,
            "debug_mesh": outline_obj,
            "wire_cutter": debug_copy,
            "dims": dims,
        })

    print(f"    Подготовлено проёмов: {len(entries)}")
    return entries


def apply_boolean_modifier(target_obj, cutter_obj, name_suffix, solver='EXACT'):
    bpy.ops.object.select_all(action='DESELECT')
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    mod = target_obj.modifiers.new(name=f"Opening_{name_suffix}", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = solver
    mod.object = cutter_obj

    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter_obj, do_unlink=True)


def apply_boolean_batch(target_obj, cutter_objects, solver='EXACT'):
    if not cutter_objects:
        return

    bpy.ops.object.select_all(action='DESELECT')
    for cutter in cutter_objects:
        cutter.select_set(True)

    bpy.context.view_layer.objects.active = cutter_objects[0]
    bpy.ops.object.join()
    combined = bpy.context.active_object
    combined.name = "Openings_Cutter_Batch"

    apply_boolean_modifier(target_obj, combined, "Batch", solver=solver)


def apply_openings(wall_obj, entries):
    if not entries:
        print("    Проёмов нет — булевы операции пропущены.")
        return

    print("    Окна/двери:")

    first = entries[0]
    print(f"      • {first['opening']['id']} → Boolean (solver=EXACT)")
    apply_boolean_modifier(wall_obj, first["cutter"], first["opening"]["id"], solver='EXACT')

    remaining_cutters = []

    if len(entries) > 1:
        second = entries[1]
        print(f"      • {second['opening']['id']} → Boolean (solver=FAST)")
        apply_boolean_modifier(wall_obj, second["cutter"], second["opening"]["id"], solver='FAST')

    for entry in entries[2:]:
        remaining_cutters.append(entry["cutter"])

    if remaining_cutters:
        print(f"      • Остальные ({len(remaining_cutters)}) → Batch boolean (solver=EXACT)")
        apply_boolean_batch(wall_obj, remaining_cutters, solver='EXACT')


def cleanup_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
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
    print("СОЗДАНИЕ КОНТУРА С ПРОЁМАМИ")
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

    opening_entries = create_opening_entries(openings, opening_heights)
    apply_openings(wall_obj, opening_entries)

    merge_duplicate_vertices(wall_obj, 1e-5)
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


if __name__ == "__main__":
    main()
