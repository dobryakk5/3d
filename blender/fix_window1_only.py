#!/usr/bin/env python3
"""
Специальный скрипт ТОЛЬКО для window_1.
Удаляет существующий куб window_1, создает новый ОГРОМНЫЙ куб,
и применяет Boolean DIFFERENCE заново.
"""

import bpy
import sys

def fix_window1():
    print("=" * 70)
    print("ИСПРАВЛЕНИЕ ТОЛЬКО WINDOW_1")
    print("=" * 70)

    # Найдем стену
    wall = bpy.data.objects.get("Outline_Walls")
    if not wall:
        print("❌ Стена не найдена!")
        return

    print(f"✓ Стена найдена: {len(wall.data.vertices)} вершин, {len(wall.data.polygons)} граней")

    # Удалим старый куб window_1 если есть
    old_cube = bpy.data.objects.get("Opening_Cube_window_1")
    if old_cube:
        print(f"  Удаляю старый куб window_1...")
        bpy.data.objects.remove(old_cube, do_unlink=True)

    # Создадим новый ОГРОМНЫЙ куб для window_1
    # Позиция из JSON: x=44.5, y=1000.5, width=227.0, height=21.0
    # После масштабирования: x=0.445-2.715, y=10.005-10.215

    x_center = 1.58  # Центр окна по X
    y_center = 10.11  # Центр окна по Y
    z_center = 1.55  # Центр по высоте (от 0.65 до 2.45)

    # Размеры куба - ОГРОМНЫЕ для гарантии
    width = 2.5  # Ширина окна (2.27 м) + запас
    depth = 3.0  # ОГРОМНАЯ толщина - гарантированно прорежет стену
    height = 2.0  # Высота окна (1.8 м)

    print(f"\n  Создаю новый куб:")
    print(f"    Location: ({x_center:.3f}, {y_center:.3f}, {z_center:.3f})")
    print(f"    Size: {width:.3f} x {depth:.3f} x {height:.3f}")

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, y_center, z_center))
    cube = bpy.context.active_object
    cube.name = "Opening_Cube_window_1_FIX"
    cube.scale = (width / 2.0, depth / 2.0, height / 2.0)

    # Применим трансформации
    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    print(f"  ✓ Куб создан и трансформирован")

    # Создадим красный материал для видимости
    mat = bpy.data.materials.new(name="Window1_Fix_Red")
    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    cube.data.materials.clear()
    cube.data.materials.append(mat)

    # Применим Boolean DIFFERENCE
    print(f"\n  Применяю Boolean DIFFERENCE...")

    bpy.ops.object.select_all(action='DESELECT')
    wall.select_set(True)
    bpy.context.view_layer.objects.active = wall

    mod = wall.modifiers.new(name="Window1_FIX", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cube
    mod.show_viewport = True
    mod.show_render = True

    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        print(f"  ✅ Boolean DIFFERENCE применен успешно!")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        try:
            wall.modifiers.remove(mod)
        except:
            pass
        return

    # Проверим результат
    print(f"\n  После Boolean:")
    print(f"    Стена: {len(wall.data.vertices)} вершин, {len(wall.data.polygons)} граней")

    # Проверим, есть ли грани внутри куба
    cube_bounds_x = (cube.location.x - 1.3, cube.location.x + 1.3)
    cube_bounds_y = (cube.location.y - 1.6, cube.location.y + 1.6)
    cube_bounds_z = (cube.location.z - 1.1, cube.location.z + 1.1)

    faces_in_cube = 0
    for poly in wall.data.polygons:
        center = poly.center
        if (cube_bounds_x[0] <= center.x <= cube_bounds_x[1] and
            cube_bounds_y[0] <= center.y <= cube_bounds_y[1] and
            cube_bounds_z[0] <= center.z <= cube_bounds_z[1]):
            faces_in_cube += 1

    print(f"\n    Граней стены внутри куба: {faces_in_cube}")

    if faces_in_cube > 0:
        print(f"  ❌ ПРОБЛЕМА: Вырез не сквозной! Остались грани внутри.")
        print(f"  Возможные причины:")
        print(f"    1. Overlapping геометрия")
        print(f"    2. Несколько слоев стены в этом месте")
        print(f"    3. Проблема с Boolean solver")
    else:
        print(f"  ✅ Вырез сквозной! Нет граней внутри куба.")

    # Установим камеру на window_1
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.camera_add(location=(x_center, y_center - 8, z_center + 2))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.2, 0, 0)
    bpy.context.scene.camera = camera

    print(f"\n✅ Готово! Нажмите Numpad 0 для вида с камеры.")
    print(f"Проверьте визуально, виден ли вырез window_1.")

if __name__ == "__main__":
    fix_window1()
