#!/usr/bin/env python3
"""
ТЕСТОВЫЙ скрипт (упрощённая геометрия стен):
  1) Стены по внешнему контуру строго «угол→угол» (без bbox-подсказок и спец‑правок)
  2) Красные кубы проёмов
  3) Булевы вырезы (Difference, EXACT) с проверкой пересечения
  4) Верификация сквозности (INTERSECT до/после)

Запуск:
  blender -b -P blender/build_openings_all_in_one_simplified.py -- \
    --json blender/2_wall_coordinates_inverted.json \
    [--heights-obj blender/2_wall_coordinates_inverted_3d.obj] \
    [--delete-cutters] [--clear]
"""

import bpy
import sys
import os
import importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Параметры (в синхроне с другими скриптами)
SCALE_FACTOR = 0.01        # 1 px = 1 см
WALL_THICKNESS_PX = 22.0   # толщина стены в пикселях
WALL_HEIGHT = 3.0          # высота стены (м)


# ----------------------------
# CLI
# ----------------------------
def _parse_args(argv):
    json_path = None
    heights_obj = None
    delete_cutters = False
    clear_scene = False
    for i, a in enumerate(argv):
        if a == '--json' and i + 1 < len(argv):
            json_path = argv[i + 1]
        elif a.startswith('--json='):
            json_path = a.split('=', 1)[1]
        elif a == '--heights-obj' and i + 1 < len(argv):
            heights_obj = argv[i + 1]
        elif a.startswith('--heights-obj='):
            heights_obj = a.split('=', 1)[1]
        elif a == '--delete-cutters':
            delete_cutters = True
        elif a == '--clear':
            clear_scene = True
    if not json_path:
        print("Ошибка: не указан --json путь к данным")
        sys.exit(1)
    return json_path, heights_obj, delete_cutters, clear_scene


def _derive_defaults_from_json(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    prefix = base_name.split('_')[0]
    return {
        'heights_obj': os.path.join(base_dir, f"{prefix}_wall_coordinates_inverted_3d.obj"),
    }


def _ensure_object_mode():
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass


def _clear_scene():
    _ensure_object_mode()
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    for coll in list(bpy.context.scene.collection.children):
        bpy.context.scene.collection.children.unlink(coll)


def _import_builder():
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    return importlib.import_module('create_outline_with_opening_cubes')


# ----------------------------
# Построение стен (угол→угол)
# ----------------------------
def _extract_corners_from_outline(data):
    out = []
    bo = data.get('building_outline') or {}
    verts = bo.get('vertices') or []
    for v in verts:
        try:
            if int(v.get('corner', 0)) == 1:
                out.append({
                    'x': float(v['x']) * SCALE_FACTOR,
                    'y': float(v['y']) * SCALE_FACTOR,
                    'junction_id': v.get('junction_id')
                })
        except Exception:
            pass
    return out


def build_rectangles_simplified(data):
    """Формирует полосы стен только по вершинам corner=1 без любых bbox‑подсказок."""
    corners = _extract_corners_from_outline(data)
    rects = []
    n = len(corners)
    if n < 2:
        print("    ⚠️ Недостаточно угловых вершин для стен")
        return rects
    half_th = float(WALL_THICKNESS_PX) * float(SCALE_FACTOR) / 2.0

    for i in range(n):
        v1 = corners[i]
        v2 = corners[(i + 1) % n]
        x1, y1 = float(v1['x']), float(v1['y'])
        x2, y2 = float(v2['x']), float(v2['y'])
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx >= dy:
            # Горизонтальная полоса: плоскость по середине Y, длина по X
            y_plane = 0.5 * (y1 + y2)
            # Продлеваем ТОЛЬКО на конце (в сторону v2) на половину толщины
            dir_sign = 1.0 if x2 >= x1 else -1.0
            x_start = x1
            x_end = x2 + dir_sign * half_th
            x_min, x_max = (x_start, x_end) if x_start <= x_end else (x_end, x_start)
            corners_xy = [
                (x_min, y_plane - half_th),
                (x_max, y_plane - half_th),
                (x_max, y_plane + half_th),
                (x_min, y_plane + half_th),
            ]
        else:
            # Вертикальная полоса: плоскость по середине X, длина по Y
            x_plane = 0.5 * (x1 + x2)
            # Продлеваем ТОЛЬКО на конце (в сторону v2) на половину толщины
            dir_sign = 1.0 if y2 >= y1 else -1.0
            y_start = y1
            y_end = y2 + dir_sign * half_th
            y_min, y_max = (y_start, y_end) if y_start <= y_end else (y_end, y_start)
            corners_xy = [
                (x_plane - half_th, y_min),
                (x_plane + half_th, y_min),
                (x_plane + half_th, y_max),
                (x_plane - half_th, y_max),
            ]

        rects.append({'corners': corners_xy})

    print(f"    Всего прямоугольников стен (упрощ.): {len(rects)}")
    return rects


def create_wall_mesh(rectangles, wall_height=WALL_HEIGHT):
    vertices = []
    faces = []
    offset = 0
    for rect in rectangles:
        lower = [(x, y, 0.0) for (x, y) in rect["corners"]]
        upper = [(x, y, wall_height) for (x, y) in rect["corners"]]
        vertices.extend(lower + upper)
        faces.append([offset + 0, offset + 1, offset + 2, offset + 3])
        faces.append([offset + 4, offset + 5, offset + 6, offset + 7])
        for i in range(4):
            j = (i + 1) % 4
            faces.append([offset + i, offset + j, offset + 4 + j, offset + 4 + i])
        offset += 8
    mesh = bpy.data.meshes.new("OutlineWallMesh_Simplified")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Outline_Walls_Simplified", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = obj
    print(f"    Mesh создан: {len(vertices)} вершин, {len(faces)} граней")
    return obj


def _collect_outline_openings(data):
    """Берём только те проёмы, чьи edge_junctions лежат на внешнем контуре."""
    vertices = (data.get("building_outline") or {}).get("vertices") or []
    outline_ids = {v.get("junction_id") for v in vertices if "junction_id" in v}
    openings = []
    for opening in data.get("openings", []):
        ejs = opening.get("edge_junctions", [])
        if ejs and all((ej.get('junction_id') in outline_ids) for ej in ejs):
            openings.append(opening)
    print(f"    Проёмов на контуре: {len(openings)}")
    return openings


# ----------------------------
# Булевы вырезы и проверка
# ----------------------------
def _duplicate_object(obj, name_suffix="_Copy"):
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = f"{obj.name}{name_suffix}"
    bpy.context.scene.collection.objects.link(dup)
    return dup


def _apply_boolean_intersect(target_obj, cutter_obj):
    _ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    mod = target_obj.modifiers.new(name="ProbeIntersect", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.solver = 'EXACT'
    mod.object = cutter_obj
    mod.show_viewport = True
    mod.show_render = True
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return True
    except Exception as e:
        print(f"  ⚠️  Ошибка INTERSECT: {e}")
        try:
            target_obj.modifiers.remove(mod)
        except Exception:
            pass
        return False


def _ensure_cutter_overlaps_wall(wall_obj, cutter_obj, max_attempts=3, grow_factor=1.10):
    """Гарантирует реальное пересечение куба со стеной, при необходимости расширяет куб по XY."""
    for attempt in range(max_attempts + 1):
        probe = _duplicate_object(wall_obj, name_suffix=f"_OverlapProbe_{cutter_obj.name}_{attempt}")
        ok = _apply_boolean_intersect(probe, cutter_obj)
        poly = len(probe.data.polygons) if ok else 0
        try:
            bpy.data.objects.remove(probe, do_unlink=True)
        except Exception:
            pass
        if ok and poly > 0:
            return True
        if attempt == max_attempts:
            break
        try:
            sx, sy, sz = cutter_obj.scale
            cutter_obj.scale = (sx * grow_factor, sy * grow_factor, sz)
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = cutter_obj
            cutter_obj.select_set(True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        except Exception:
            pass
    return False


def apply_boolean_cuts(wall_obj, cutters, delete_cutters=False):
    _ensure_object_mode()

    # Подготовка нормалей
    try:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = wall_obj
        wall_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        _ensure_object_mode()

    success = 0
    for i, cutter in enumerate(cutters):
        print(f"[Cut] {i+1}/{len(cutters)}: {cutter.name}")
        # Применяем scale к кубу и убеждаемся в пересечении со стеной
        try:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = cutter
            cutter.select_set(True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        except Exception:
            pass
        if not _ensure_cutter_overlaps_wall(wall_obj, cutter):
            print("  ❌ Куб не пересекает стену даже после расширения — пропуск")
            continue

        bpy.ops.object.select_all(action='DESELECT')
        wall_obj.select_set(True)
        bpy.context.view_layer.objects.active = wall_obj

        mod = wall_obj.modifiers.new(name=f"Opening_{i}", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.solver = 'EXACT'
        mod.object = cutter
        mod.show_viewport = True
        mod.show_render = True
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
            success += 1
        except Exception as e:
            print(f"  ❌ Ошибка применения Difference: {e}")
            try:
                wall_obj.modifiers.remove(mod)
            except Exception:
                pass

        if delete_cutters:
            try:
                bpy.data.objects.remove(cutter, do_unlink=True)
            except Exception:
                pass

    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        _ensure_object_mode()

    print(f"Применено вырезов: {success}/{len(cutters)}")


def verify_openings_through(wall_obj_after, wall_obj_before, cutters):
    ok, fail = [], []
    for i, cutter in enumerate(cutters):
        print(f"[Verify] {i+1}/{len(cutters)}: {cutter.name}")
        # До выреза — должен быть контакт
        probe_pre = _duplicate_object(wall_obj_before, name_suffix=f"_ProbePre_{i}")
        if not _apply_boolean_intersect(probe_pre, cutter):
            print("  ⚠️ INTERSECT на исходной стене не выполнился")
            try:
                bpy.data.objects.remove(probe_pre, do_unlink=True)
            except Exception:
                pass
            fail.append(cutter.name)
            continue
        pre_poly = len(probe_pre.data.polygons)
        try:
            bpy.data.objects.remove(probe_pre, do_unlink=True)
        except Exception:
            pass
        if pre_poly == 0:
            print("  ❌ Куб не пересекал стену ДО выреза")
            fail.append(cutter.name)
            continue

        # После выреза — не должно остаться геометрии внутри куба
        probe_post = _duplicate_object(wall_obj_after, name_suffix=f"_ProbePost_{i}")
        if not _apply_boolean_intersect(probe_post, cutter):
            try:
                bpy.data.objects.remove(probe_post, do_unlink=True)
            except Exception:
                pass
            fail.append(cutter.name)
            continue
        post_poly = len(probe_post.data.polygons)
        try:
            bpy.data.objects.remove(probe_post, do_unlink=True)
        except Exception:
            pass
        if post_poly == 0:
            print("  ✅ Сквозной вырез подтверждён")
            ok.append(cutter.name)
        else:
            print(f"  ❌ Осталась геометрия после выреза (полигонов: {post_poly})")
            fail.append(cutter.name)
    return ok, fail


# ----------------------------
# Основной сценарий
# ----------------------------
def main():
    json_path, heights_obj, delete_cutters, clear_scene = _parse_args(sys.argv)
    if not heights_obj:
        heights_obj = _derive_defaults_from_json(json_path)['heights_obj']

    print("=" * 70)
    print("УПРОЩЁННЫЙ: СТЕНЫ УГОЛ→УГОЛ, КУБЫ, ВЫРЕЗЫ, ПРОВЕРКА")
    print("=" * 70)
    print(f"JSON: {json_path}")
    print(f"OBJ высот: {heights_obj}")
    print(f"Очистка сцены: {clear_scene}")
    print(f"Удалять кубы после выреза: {delete_cutters}")

    if clear_scene:
        _clear_scene()

    # Загружаем данные и готовим геометрию
    build = _import_builder()
    data = build.load_json_data(json_path)
    rectangles = build_rectangles_simplified(data)
    wall_obj = create_wall_mesh(rectangles, wall_height=WALL_HEIGHT)
    # Небольшая очистка перед булевыми
    try:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = wall_obj
        wall_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    openings = _collect_outline_openings(data)
    opening_heights = build.load_opening_heights_from_obj(heights_obj)
    cutters = build.create_opening_cubes(openings, opening_heights)

    print(f"Создан объект стен: {wall_obj.name}; кубов: {len(cutters)}")

    wall_before = _duplicate_object(wall_obj, name_suffix="_BeforeCuts")
    apply_boolean_cuts(wall_obj, cutters, delete_cutters=delete_cutters)

    if delete_cutters:
        print("Кубы удалены — проверка сквозности пропущена.")
        return

    ok, fail = verify_openings_through(wall_obj, wall_before, cutters)
    try:
        bpy.data.objects.remove(wall_before, do_unlink=True)
    except Exception:
        pass

    print()
    print("РЕЗЮМЕ ПРОВЕРКИ (упрощ.)")
    print(f"  Успешно: {len(ok)}")
    print(f"  Проблемы: {len(fail)}")
    if fail:
        for name in fail:
            print(f"    - {name}")
    print("Готово.")


if __name__ == "__main__":
    main()
