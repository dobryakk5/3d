#!/usr/bin/env python3
"""
Диагностический скрипт для проверки пересечения window_1 со стеной.
Выводит координаты куба window_1 и соответствующей стены.
"""

import bpy
import sys

def check_intersection():
    print("=" * 70)
    print("ДИАГНОСТИКА WINDOW_1")
    print("=" * 70)

    # Найдем куб window_1
    cube_name = "Opening_Cube_window_1"
    cube = bpy.data.objects.get(cube_name)

    if not cube:
        print(f"❌ Куб {cube_name} не найден!")
        return

    print(f"\n✓ Куб найден: {cube.name}")
    print(f"  Location: ({cube.location.x:.3f}, {cube.location.y:.3f}, {cube.location.z:.3f})")
    print(f"  Scale: ({cube.scale.x:.3f}, {cube.scale.y:.3f}, {cube.scale.z:.3f})")

    # Вычислим фактические границы куба (с учетом того, что cube size=1.0 означает от -0.5 до 0.5)
    # После scale границы становятся: location ± scale
    print(f"\n  Границы куба (после scale):")
    print(f"    X: {cube.location.x - cube.scale.x:.3f} до {cube.location.x + cube.scale.x:.3f}")
    print(f"    Y: {cube.location.y - cube.scale.y:.3f} до {cube.location.y + cube.scale.y:.3f}")
    print(f"    Z: {cube.location.z - cube.scale.z:.3f} до {cube.location.z + cube.scale.z:.3f}")

    # Найдем стену
    wall_name = "Outline_Walls"
    wall = bpy.data.objects.get(wall_name)

    if not wall:
        print(f"\n❌ Стена {wall_name} не найдена!")
        return

    print(f"\n✓ Стена найдена: {wall.name}")
    print(f"  Вершин: {len(wall.data.vertices)}")
    print(f"  Граней: {len(wall.data.polygons)}")

    # Найдем вершины стены в области window_1 (около y=10.1)
    target_y = 10.1
    tolerance = 0.5
    matching_verts = []

    for v in wall.data.vertices:
        if abs(v.co.y - target_y) < tolerance:
            matching_verts.append(v)

    print(f"\n  Вершины стены около y={target_y} (±{tolerance}):")
    print(f"    Найдено вершин: {len(matching_verts)}")

    if matching_verts:
        # Найдем границы стены в этой области
        x_coords = [v.co.x for v in matching_verts]
        y_coords = [v.co.y for v in matching_verts]
        z_coords = [v.co.z for v in matching_verts]

        print(f"    X: {min(x_coords):.3f} до {max(x_coords):.3f}")
        print(f"    Y: {min(y_coords):.3f} до {max(y_coords):.3f}")
        print(f"    Z: {min(z_coords):.3f} до {max(z_coords):.3f}")

        # Проверим пересечение
        cube_x_min = cube.location.x - cube.scale.x
        cube_x_max = cube.location.x + cube.scale.x
        cube_y_min = cube.location.y - cube.scale.y
        cube_y_max = cube.location.y + cube.scale.y
        cube_z_min = cube.location.z - cube.scale.z
        cube_z_max = cube.location.z + cube.scale.z

        wall_x_min = min(x_coords)
        wall_x_max = max(x_coords)
        wall_y_min = min(y_coords)
        wall_y_max = max(y_coords)
        wall_z_min = min(z_coords)
        wall_z_max = max(z_coords)

        print(f"\n  Проверка пересечения:")

        x_overlap = (cube_x_min <= wall_x_max) and (cube_x_max >= wall_x_min)
        y_overlap = (cube_y_min <= wall_y_max) and (cube_y_max >= wall_y_min)
        z_overlap = (cube_z_min <= wall_z_max) and (cube_z_max >= wall_z_min)

        print(f"    X пересечение: {'✓' if x_overlap else '✗'}")
        print(f"    Y пересечение: {'✓' if y_overlap else '✗'}")
        print(f"    Z пересечение: {'✓' if z_overlap else '✗'}")

        if x_overlap and y_overlap and z_overlap:
            print(f"\n  ✅ Куб ДОЛЖЕН пересекать стену!")
        else:
            print(f"\n  ❌ Куб НЕ пересекает стену!")

    else:
        print("  ❌ Не найдено вершин стены в этой области!")

    # Проверим модификаторы
    print(f"\n  Модификаторы на стене:")
    if len(wall.modifiers) == 0:
        print("    Нет модификаторов")
    else:
        for mod in wall.modifiers:
            print(f"    - {mod.name}: {mod.type}")
            if mod.type == 'BOOLEAN':
                print(f"      Operation: {mod.operation}")
                print(f"      Object: {mod.object.name if mod.object else 'None'}")

if __name__ == "__main__":
    check_intersection()
