#!/usr/bin/env python3
"""
Создаёт 3D фундамент и потолок на основе данных из JSON файла.
Фундамент основан на функции create_foundation из create_outline_with_openings.py.
Потолок строится по периметру стен из building_outline (угловые вершины corner=1) и экструзится вверх.
"""

import bpy
import bmesh
import os
import json
import sys
from mathutils import Vector

# ---------------------------------
# Константы
# ---------------------------------
SCALE_FACTOR = 0.01        # 1 px = 1 см
FOUNDATION_Z_OFFSET = 0.0   # верх фундамента на уровне низа здания
FOUNDATION_THICKNESS = 0.75 # толщина фундамента в метрах

# Потолок (перекрытие)
CEILING_Z_OFFSET = 3.0      # уровень нижней плоскости потолка
CEILING_THICKNESS = 0.20    # толщина потолка (20 см)

# Пути по умолчанию
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Вшитые входы/выходы (правьте под свои файлы)
JSON_PATH = os.path.join(SCRIPT_DIR, "2_wall_coordinates_inverted.json")
OUTLINE_OBJ_IMPORT = os.path.join(SCRIPT_DIR, "2_outline_with_openings.obj")
OUTPUT_OBJ = os.path.join(SCRIPT_DIR, "2_foundation_and_ceiling.obj")

def _derive_defaults_from_json(json_path):
    """Определяет пути по умолчанию на основе имени JSON файла"""
    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    return {
        'out_foundation': os.path.join(base_dir, f"{base_name}_foundation_3d.obj"),
        'out_ceiling': os.path.join(base_dir, f"{base_name}_ceiling_3d.obj"),
    }

def _apply_cli_overrides(argv):
    """Необязательные CLI-переопределения: --json, --out.
    Возвращает (json_path, output_obj). По умолчанию берёт значения из констант JSON_PATH/OUTPUT_OBJ.
    """
    json_path = JSON_PATH
    output_obj = OUTPUT_OBJ

    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = []

    for i, a in enumerate(script_args):
        if a == '--json' and i + 1 < len(script_args):
            json_path = script_args[i + 1]
        elif a.startswith('--json='):
            json_path = a.split('=', 1)[1]
        elif a == '--out' and i + 1 < len(script_args):
            output_obj = script_args[i + 1]
        elif a.startswith('--out='):
            output_obj = a.split('=', 1)[1]

    if not os.path.isabs(json_path):
        candidate = os.path.join(SCRIPT_DIR, json_path)
        if os.path.exists(candidate):
            json_path = candidate
    if not os.path.exists(json_path):
        print(f"Ошибка: JSON файл не найден: {json_path}")
        sys.exit(1)
    if not os.path.isabs(output_obj):
        output_obj = os.path.join(SCRIPT_DIR, output_obj)

    return json_path, output_obj

def clear_scene():
    """Очищает сцену от всех объектов"""
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    for coll in list(bpy.context.scene.collection.children):
        bpy.context.scene.collection.children.unlink(coll)
    
    # Убедимся, что View Layer настроен правильно
    if hasattr(bpy.context, 'view_layer') and bpy.context.view_layer:
        bpy.context.view_layer.update()

def load_json_data(path):
    """Загружает данные из JSON файла"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _outline_corners_from_json(data, scale_factor=SCALE_FACTOR):
    """Возвращает вершины периметра стен по building_outline.corner=1, в метрах."""
    out = []
    bo = data.get('building_outline') or {}
    verts = bo.get('vertices') or []
    for v in verts:
        try:
            if int(v.get('corner', 0)) != 1:
                continue
        except Exception:
            continue
        out.append({'x': float(v['x']) * scale_factor, 'y': float(v['y']) * scale_factor})
    if len(out) >= 2:
        x0, y0 = out[0]['x'], out[0]['y']
        x1, y1 = out[-1]['x'], out[-1]['y']
        if abs(x0 - x1) < 1e-6 and abs(y0 - y1) < 1e-6:
            out.pop()
    return out

def create_foundation(foundation_data, z_offset=FOUNDATION_Z_OFFSET, thickness=FOUNDATION_THICKNESS, scale_factor=SCALE_FACTOR):
    """Создает 3D меш фундамента из данных JSON"""
    if not foundation_data or 'vertices' not in foundation_data:
        print("    ⚠️  Данные фундамента не найдены в JSON")
        return None

    vertices_2d = foundation_data['vertices']

    mesh = bpy.data.meshes.new(name="Foundation_Mesh")
    foundation_obj = bpy.data.objects.new("Foundation", mesh)
    bpy.context.collection.objects.link(foundation_obj)
    
    # Убедимся, что объект добавлен в View Layer
    if hasattr(bpy.context, 'view_layer') and bpy.context.view_layer:
        bpy.context.view_layer.update()
        # Принудительно добавляем объект в View Layer
        if foundation_obj.name not in bpy.context.view_layer.objects:
            bpy.context.view_layer.objects.link(foundation_obj)

    vertices = []
    # Верхние вершины фундамента
    for v in vertices_2d:
        vertices.append((v['x'] * scale_factor, v['y'] * scale_factor, z_offset))
    # Нижние вершины фундамента
    for v in vertices_2d:
        vertices.append((v['x'] * scale_factor, v['y'] * scale_factor, z_offset - thickness))

    num_verts = len(vertices_2d)
    faces = []
    # Верхняя грань
    top_face = list(range(num_verts))
    faces.append(top_face)
    # Нижняя грань (обратный порядок для корректных нормалей)
    bottom_face = list(range(num_verts, 2 * num_verts))
    bottom_face.reverse()
    faces.append(bottom_face)
    # Боковые грани
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        face = [i, next_i, next_i + num_verts, i + num_verts]
        faces.append(face)

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Корректируем нормали через bmesh (работает в фоновом режиме)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    # Создаем материал тёмно-серого цвета для фундамента
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

def create_ceiling(outline_vertices, z_offset=CEILING_Z_OFFSET, thickness=CEILING_THICKNESS):
    """Создаёт 3D-меш потолка по списку 2D-вершин outline_vertices (метры)."""
    if not outline_vertices:
        print("    ⚠️  Нет вершин периметра для потолка")
        return None

    mesh = bpy.data.meshes.new(name="Ceiling_Mesh")
    ceiling_obj = bpy.data.objects.new("Ceiling", mesh)
    bpy.context.collection.objects.link(ceiling_obj)
    if hasattr(bpy.context, 'view_layer') and bpy.context.view_layer:
        bpy.context.view_layer.update()

    verts = []
    for v in outline_vertices:
        verts.append((v['x'], v['y'], z_offset))
    for v in outline_vertices:
        verts.append((v['x'], v['y'], z_offset + thickness))

    n = len(outline_vertices)
    faces = []
    bottom = list(range(0, n))
    bottom.reverse()
    faces.append(bottom)
    top = list(range(n, 2 * n))
    faces.append(top)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    mat = bpy.data.materials.new(name="Ceiling_Material_Light_Gray")
    mat.use_nodes = True
    try:
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.85, 0.85, 0.85, 1.0)
    except Exception:
        pass
    ceiling_obj.data.materials.append(mat)

    print(f"    Создан потолок: {len(verts)} вершин, {len(faces)} граней")
    print(f"    Z: {z_offset}м до {z_offset + thickness}м, толщина: {thickness}м")
    return ceiling_obj

def _frame_object_in_viewport(obj):
    """В не-фоновом режиме центрирует и кадрирует объект в 3D-вьюпорте."""
    try:
        if not bpy.context.window_manager.windows:
            return
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    if not region:
                        continue
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        bpy.ops.object.select_all(action='DESELECT')
                        obj.select_set(True)
                        bpy.context.view_layer.objects.active = obj
                        try:
                            bpy.ops.view3d.view_selected()
                        except Exception:
                            bpy.ops.view3d.view_all(center=True)
                    return
    except Exception:
        # Тихо игнорируем проблемы с UI в фоновом режиме
        pass

def export_objs(objs, output_path):
    """Экспортирует список объектов в один OBJ файл (только выбранные объекты)."""
    # Обновляем слой и убеждаемся, что объекты видимы
    if hasattr(bpy.context, 'view_layer') and bpy.context.view_layer:
        bpy.context.view_layer.update()
        for o in objs:
            if o and (o.name not in bpy.context.view_layer.objects):
                try:
                    bpy.context.view_layer.objects.link(o)
                except Exception:
                    pass

    try:
        bpy.ops.object.select_all(action='DESELECT')
        to_export = [o for o in objs if o is not None]
        for o in to_export:
            o.select_set(True)
        if to_export:
            bpy.context.view_layer.objects.active = to_export[0]

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
    except Exception as e:
        print(f"    ⚠️  Ошибка при экспорте: {e}")
        return

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"    Экспортирован OBJ: {output_path} ({size / 1024:.1f} KB)")
    else:
        print("    ⚠️  Экспорт OBJ не создал файл.")

def _frame_objects_in_viewport(objs):
    """Кадрирует список объектов во вьюпорте (не в фоновом режиме)."""
    try:
        if not bpy.context.window_manager.windows:
            return
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    if not region:
                        continue
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        bpy.ops.object.select_all(action='DESELECT')
                        for o in objs:
                            if o is not None:
                                o.select_set(True)
                        if objs and objs[0] is not None:
                            bpy.context.view_layer.objects.active = objs[0]
                        try:
                            bpy.ops.view3d.view_selected()
                        except Exception:
                            bpy.ops.view3d.view_all(center=True)
                    return
    except Exception:
        pass

def import_outline_obj(path):
    """Импортирует OBJ с контуром в сцену (только для визуального контекста)."""
    if not path:
        return []
    obj_path = path
    if not os.path.isabs(obj_path):
        obj_path = os.path.join(SCRIPT_DIR, obj_path)
    if not os.path.exists(obj_path):
        print(f"    ⚠️ OBJ для импорта не найден: {obj_path}")
        return []
    imported = []
    try:
        try:
            bpy.ops.wm.obj_import(filepath=obj_path)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=obj_path)
        imported = list(bpy.context.selected_objects)
        print(f"    Импортировано объектов из OBJ: {len(imported)}")
    except Exception as e:
        print(f"    ⚠️ Ошибка импорта OBJ: {e}")
    return imported

def main():
    """Основная функция скрипта"""
    print("=" * 70)
    print("СОЗДАНИЕ 3D ФУНДАМЕНТА И ПОТОЛКА")
    print("=" * 70)
    
    # Пути берутся из констант; опционально их можно переопределить через CLI (--json, --out)
    json_path, output_obj = _apply_cli_overrides(sys.argv)
    
    print(f"JSON: {json_path}")
    print(f"Импортируемый контур: {OUTLINE_OBJ_IMPORT}")
    print(f"Выходной OBJ (фундамент+потолок): {output_obj}")
    print()

    # Очищаем сцену
    clear_scene()
    
    # Загружаем данные
    data = load_json_data(json_path)
    
    # Проверяем наличие данных фундамента
    if 'foundation' not in data:
        print("❌ Ошибка: в JSON отсутствуют данные фундамента")
        sys.exit(1)
    
    # Создаем фундамент
    print("\n[FOUNDATION] Создание 3D фундамента")
    foundation_obj = create_foundation(
        data['foundation'],
        z_offset=FOUNDATION_Z_OFFSET,
        thickness=FOUNDATION_THICKNESS,
        scale_factor=SCALE_FACTOR,
    )
    
    if foundation_obj is None:
        print("❌ Ошибка: не удалось создать фундамент")
        sys.exit(1)
    
    # Создаем потолок
    print("\n[CEILING] Создание 3D потолка")
    outline_vertices = _outline_corners_from_json(data)
    if not outline_vertices:
        print("    ⚠️  В JSON нет корректного building_outline.corner=1 — потолок пропущен")
        ceiling_obj = None
    else:
        ceiling_obj = create_ceiling(outline_vertices, z_offset=CEILING_Z_OFFSET, thickness=CEILING_THICKNESS)

    # Импортируем контур в сцену (не включаем в экспорт)
    print("\n[IMPORT] Импорт контура в сцену")
    import_outline_obj(OUTLINE_OBJ_IMPORT)

    # В обычном (не фоновом) режиме — кадрируем оба объекта в окне, чтобы их было видно
    is_background = "--background" in sys.argv
    if not is_background:
        objs_to_frame = [o for o in [foundation_obj, ceiling_obj] if o is not None]
        if objs_to_frame:
            _frame_objects_in_viewport(objs_to_frame)

    # Единый экспорт (фундамент + потолок в один файл)
    print("\n[EXPORT] Экспорт единого OBJ (фундамент+потолок)")
    objs_to_export = [o for o in [foundation_obj, ceiling_obj] if o is not None]
    export_objs(objs_to_export, output_obj)
    
    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print(f"Фундамент: {foundation_obj.name}")
    print(f"Вершин фундамента: {len(foundation_obj.data.vertices)}")
    print(f"Граней фундамента: {len(foundation_obj.data.polygons)}")
    if 'ceiling_obj' in locals() and ceiling_obj is not None:
        print(f"Потолок: {ceiling_obj.name}")
        print(f"Вершин потолка: {len(ceiling_obj.data.vertices)}")
        print(f"Граней потолка: {len(ceiling_obj.data.polygons)}")
    print(f"Экспортировано в: {output_obj}")

if __name__ == "__main__":
    # Проверяем, передан ли параметр --no-auto-run
    auto_run = "--no-auto-run" not in sys.argv

    if auto_run:
        try:
            main()
            print("=" * 70)
            print("АВТОМАТИЧЕСКИЙ ЗАПУСК УСПЕШНО ЗАВЕРШЕН")
            print("=" * 70)
            
            # Проверяем, запущен ли Blender в фоновом режиме
            is_background = "--background" in sys.argv

            if not is_background:
                # В обычном режиме оставляем Blender открытым для просмотра результата
                print("Blender остается открытым для просмотра 3D моделей (фундамент+потолок)")
                # Не выходим, позволяем Blender остаться открытым
            else:
                # В фоновом режиме выходим корректно
                import sys
                sys.exit(0)
                
        except Exception as e:
            print(f"Ошибка при выполнении: {e}")
            print("=" * 70)
            print("АВТОМАТИЧЕСКИЙ ЗАПУСК ЗАВЕРШИЛСЯ С ОШИБКОЙ")
            print("=" * 70)
            sys.exit(1)
    else:
        print("Скрипт загружен без автоматического запуска (--no-auto-run)")
