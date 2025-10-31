#!/usr/bin/env python3
"""
Скрипт для запуска ВНУТРИ открытого Blender (через Scripting workspace).
Проверяет и визуализирует window_1.
"""

import bpy

def check_window1():
    # Найдем куб и стену
    cube = bpy.data.objects.get("Opening_Cube_window_1")
    wall = bpy.data.objects.get("Outline_Walls")

    if not cube:
        print("❌ Куб Opening_Cube_window_1 не найден!")
        return

    if not wall:
        print("❌ Стена Outline_Walls не найдена!")
        return

    print(f"✓ Найдены: куб и стена")
    print(f"  Куб location: {cube.location}")
    print(f"  Куб scale: {cube.scale}")
    print(f"  Стена: {len(wall.data.vertices)} вершин, {len(wall.data.polygons)} граней")

    # Выделим куб и стену для визуального просмотра
    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(True)
    wall.select_set(True)
    bpy.context.view_layer.objects.active = wall

    # Переключимся на Edit Mode стены чтобы увидеть вершины
    bpy.ops.object.mode_set(mode='OBJECT')

    # Проверим, есть ли вершины стены в области window_1
    target_y = 10.1
    target_x_min = 0.4
    target_x_max = 2.7
    tolerance = 0.5

    matching_verts = []
    for v in wall.data.vertices:
        if (abs(v.co.y - target_y) < tolerance and
            target_x_min <= v.co.x <= target_x_max):
            matching_verts.append(v)

    print(f"\n  Вершин стены в области window_1: {len(matching_verts)}")

    if matching_verts:
        x_coords = [v.co.x for v in matching_verts]
        y_coords = [v.co.y for v in matching_verts]
        z_coords = [v.co.z for v in matching_verts]

        print(f"    Диапазон X: {min(x_coords):.3f} - {max(x_coords):.3f}")
        print(f"    Диапазон Y: {min(y_coords):.3f} - {max(y_coords):.3f}")
        print(f"    Диапазон Z: {min(z_coords):.3f} - {max(z_coords):.3f}")

    # Проверим грани стены в области куба
    cube_bounds_x = (cube.location.x - 1.2, cube.location.x + 1.2)
    cube_bounds_y = (cube.location.y - 0.3, cube.location.y + 0.3)
    cube_bounds_z = (cube.location.z - 1.0, cube.location.z + 1.0)

    faces_in_cube = 0
    for poly in wall.data.polygons:
        center = poly.center
        if (cube_bounds_x[0] <= center.x <= cube_bounds_x[1] and
            cube_bounds_y[0] <= center.y <= cube_bounds_y[1] and
            cube_bounds_z[0] <= center.z <= cube_bounds_z[1]):
            faces_in_cube += 1

    print(f"\n  Граней стены ВНУТРИ куба: {faces_in_cube}")

    if faces_in_cube > 0:
        print(f"  ❌ ПРОБЛЕМА: Есть грани стены внутри куба - вырез не сквозной!")
    else:
        print(f"  ✅ Нет граней внутри куба - вырез успешен!")

    # Установим камеру так, чтобы смотреть на window_1
    print(f"\n📷 Установка камеры на window_1...")
    bpy.ops.object.camera_add(location=(cube.location.x, cube.location.y - 5, cube.location.z))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.5708, 0, 0)  # 90 градусов вокруг X
    bpy.context.scene.camera = camera

    print(f"\n✓ Готово! Теперь:")
    print(f"  1. Нажмите Numpad 0 для вида с камеры")
    print(f"  2. Или вручную приблизьтесь к координатам {cube.location}")
    print(f"  3. Проверьте, виден ли вырез в стене")

if __name__ == "__main__":
    check_window1()
