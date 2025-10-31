#!/usr/bin/env python3
"""
Один скрипт, который:
  1) Строит стены внешнего контура
  2) Создаёт красные кубы проёмов
  3) Делает вырезы по кубам (Boolean Difference, EXACT)
  4) Проверяет, что каждый проём вырезан насквозь

Минимальная реализация без меток/материалов стен/экспорта. Кубы остаются в сцене.

Запуск:
  blender -b -P blender/build_openings_all_in_one.py -- \
    --json blender/2_wall_coordinates_inverted.json \
    [--heights-obj blender/2_wall_coordinates_inverted_3d.obj] \
    [--delete-cutters] [--clear]

Где:
  --json          путь к JSON с данными (обязательно)
  --heights-obj   OBJ с Fill_Below_/Fill_Above_ для уточнения высот (опционально, по умолчанию подбирается)
  --delete-cutters  удалить кубы после вырезания (по умолчанию оставить для наглядности)
  --clear         очистить сцену (удалить все объекты) перед построением
"""

import bpy
import sys
import os
import importlib

from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WALL_NAME = "Outline_Walls"
CUTTERS_COLLECTION = "OPENINGS_DEBUG"
CUTTER_NAME_PREFIX = "Opening_Cube_"


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
# Построение
# ----------------------------
def build_walls_and_cubes(json_path, heights_obj):
    build = _import_builder()
    data = build.load_json_data(json_path)
    rectangles = build.create_rectangles(data)
    wall_obj = build.create_wall_mesh(rectangles, wall_height=build.WALL_HEIGHT)
    # Быстрая очистка меша перед булевыми операциями (улучшает устойчивость):
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
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    # Список проёмов только по внешнему контуру
    vertices = data.get("building_outline", {}).get("vertices", [])
    outline_ids = {v["junction_id"] for v in vertices}
    openings = build.collect_outline_openings(data, outline_ids)
    opening_heights = build.load_opening_heights_from_obj(heights_obj)
    cubes = build.create_opening_cubes(openings, opening_heights)
    # Диагностика размеров кубов
    thickness_m = float(build.WALL_THICKNESS_PX) * float(build.SCALE_FACTOR)
    for i, c in enumerate(cubes):
        sx, sy, sz = c.scale
        print(f"  [Cube] #{i} {c.name} scale=({sx:.3f},{sy:.3f},{sz:.3f}) thickness={thickness_m:.3f}")
    return wall_obj, cubes


def apply_boolean_cuts(wall_obj, cutters, delete_cutters=False):
    _ensure_object_mode()

    # Убедимся, что все коллекции видимы в view_layer
    for collection in bpy.data.collections:
        if collection.name in bpy.context.scene.collection.children:
            for layer_collection in bpy.context.view_layer.layer_collection.children:
                if layer_collection.collection == collection:
                    layer_collection.exclude = False

    # Нормали перед началом
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
        # Применим трансформации масштабов к кубу для избежания артефактов
        try:
            _ensure_object_mode()
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = cutter
            cutter.select_set(True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            print(f"  ✓ Трансформации применены к {cutter.name}")
        except Exception as e:
            print(f"  ⚠️ Не удалось применить трансформации к {cutter.name}: {e}")
        # Гарантируем реальное пересечение кубом стены (иначе Difference ничего не изменит)
        if not _ensure_cutter_overlaps_wall(wall_obj, cutter):
            print("  ❌ Куб не пересекает стену даже после расширения — пропускаю вырез")
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
            print(f"  ❌ Ошибка применения модификатора: {e}")
            try:
                wall_obj.modifiers.remove(mod)
            except Exception:
                pass

        if delete_cutters:
            try:
                bpy.data.objects.remove(cutter, do_unlink=True)
            except Exception:
                pass

    # Финальные нормали
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        _ensure_object_mode()

    print(f"Применено вырезов: {success}/{len(cutters)}")


# ----------------------------
# Проверка
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
    """Проверяет, что куб реально пересекает стену. Если нет —
    последовательно увеличивает масштаб куба по XY (не по Z),
    применяет трансформацию и проверяет снова. Возвращает True/False."""
    for attempt in range(max_attempts + 1):
        # Проверка INTERSECT на копии стены
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
        # Растим по XY
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


def verify_openings_through(wall_obj_after, wall_obj_before, cutters):
    """Для каждого куба создаёт временную копию стены и делает INTERSECT.
    Если после INTERSECT остаются грани (>0), значит стена пересекается с объёмом куба —
    вырез не сквозной. Если граней 0 — сквозной.
    Возвращает (ok_ids, fail_ids).
    """
    ok, fail = [], []
    for i, cutter in enumerate(cutters):
        print(f"[Verify] {i+1}/{len(cutters)}: {cutter.name}")
        # 1) До выреза — должен быть конфликт (иначе куб не попадал в стену)
        probe_pre = _duplicate_object(wall_obj_before, name_suffix=f"_ProbePre_{i}")
        if not _apply_boolean_intersect(probe_pre, cutter):
            print("  ⚠️ Не удалось выполнить INTERSECT на исходной стене")
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
            print("  ❌ Куб НЕ пересекал стену до выреза (вырез не мог образоваться)")
            fail.append(cutter.name)
            # Переходим к следующему кубу — смысла проверять постфактум нет
            continue

        # 2) После выреза — не должно оставаться геометрии внутри куба
        probe_post = _duplicate_object(wall_obj_after, name_suffix=f"_ProbePost_{i}")
        if not _apply_boolean_intersect(probe_post, cutter):
            # Если INTERSECT не применился — считаем как сбой
            fail.append(cutter.name)
            try:
                bpy.data.objects.remove(probe_post, do_unlink=True)
            except Exception:
                pass
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
            print(f"  ❌ Найдена геометрия стены внутри объёма куба после выреза (полигонов: {post_poly})")
            fail.append(cutter.name)
    return ok, fail


def main():
    json_path, heights_obj, delete_cutters, clear_scene = _parse_args(sys.argv)
    if not heights_obj:
        heights_obj = _derive_defaults_from_json(json_path)['heights_obj']

    print("=" * 70)
    print("ПОСТРОЕНИЕ СТЕН, КУБОВ И ВЫРЕЗОВ + ПРОВЕРКА")
    print("=" * 70)
    print(f"JSON: {json_path}")
    print(f"OBJ высот: {heights_obj}")
    print(f"Очистка сцены: {clear_scene}")
    print(f"Удалять кубы после выреза: {delete_cutters}")

    if clear_scene:
        _clear_scene()

    wall_obj, cutters = build_walls_and_cubes(json_path, heights_obj)
    print(f"Создан объект стен: {wall_obj.name}; кубов: {len(cutters)}")

    # Сохраним копию стены ДО вырезов для верификации
    wall_before = _duplicate_object(wall_obj, name_suffix="_BeforeCuts")

    apply_boolean_cuts(wall_obj, cutters, delete_cutters=delete_cutters)

    # Если кубы удалили — соберём список объектов-кубов повторно (их уже нет, проверка невозможна)
    if delete_cutters:
        print("Кубы удалены — пропускаю проверку сквозности.")
        return

    ok, fail = verify_openings_through(wall_obj, wall_before, cutters)

    # Удалим копию стены ДО вырезов
    try:
        bpy.data.objects.remove(wall_before, do_unlink=True)
    except Exception:
        pass
    print()
    print("РЕЗЮМЕ ПРОВЕРКИ")
    print(f"  Успешно: {len(ok)}")
    print(f"  Проблемы: {len(fail)}")
    if fail:
        print("  Непрошедшие (оставили геометрию внутри объёма куба):")
        for name in fail:
            print(f"    - {name}")
    print("Готово.")


if __name__ == "__main__":
    main()
