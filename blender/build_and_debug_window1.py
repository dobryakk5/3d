#!/usr/bin/env python3
"""
Объединенный скрипт: строит стены и кубы, делает вырезы, затем диагностирует window_1
"""

import bpy
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Импортируем основной скрипт
from build_openings_all_in_one import main as build_main

def diagnose_window1():
    """Диагностика куба и стены для window_1"""
    print("\n" + "=" * 70)
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

    # Вычислим фактические границы куба из его вершин (точный метод)
    if len(cube.data.vertices) > 0:
        # Получим мировые координаты вершин
        verts_world = [cube.matrix_world @ v.co for v in cube.data.vertices]
        cube_x_coords = [v.x for v in verts_world]
        cube_y_coords = [v.y for v in verts_world]
        cube_z_coords = [v.z for v in verts_world]

        cube_x_min = min(cube_x_coords)
        cube_x_max = max(cube_x_coords)
        cube_y_min = min(cube_y_coords)
        cube_y_max = max(cube_y_coords)
        cube_z_min = min(cube_z_coords)
        cube_z_max = max(cube_z_coords)

        print(f"\n  Реальные границы куба (из вершин в мировых координатах):")
        print(f"    X: {cube_x_min:.3f} до {cube_x_max:.3f}  (размер: {cube_x_max - cube_x_min:.3f})")
        print(f"    Y: {cube_y_min:.3f} до {cube_y_max:.3f}  (размер: {cube_y_max - cube_y_min:.3f})")
        print(f"    Z: {cube_z_min:.3f} до {cube_z_max:.3f}  (размер: {cube_z_max - cube_z_min:.3f})")
    else:
        print(f"\n  ⚠️  Куб не имеет вершин!")
        cube_x_min = cube.location.x - 0.5
        cube_x_max = cube.location.x + 0.5
        cube_y_min = cube.location.y - 0.5
        cube_y_max = cube.location.y + 0.5
        cube_z_min = cube.location.z - 0.5
        cube_z_max = cube.location.z + 0.5

    # Найдем стену
    wall_name = "Outline_Walls"
    wall = bpy.data.objects.get(wall_name)

    if not wall:
        print(f"\n❌ Стена {wall_name} не найдена!")
        return

    print(f"\n✓ Стена найдена: {wall.name}")
    print(f"  Вершин: {len(wall.data.vertices)}")
    print(f"  Граней: {len(wall.data.polygons)}")

    # Найдем вершины стены в области window_1
    # window_1 должен быть около y=10.1, x от 0.1 до 3.0
    target_y = 10.1
    target_x_min = 0.0
    target_x_max = 3.5
    tolerance_y = 0.3
    matching_verts = []

    for v in wall.data.vertices:
        if (abs(v.co.y - target_y) < tolerance_y and
            target_x_min <= v.co.x <= target_x_max):
            matching_verts.append(v)

    print(f"\n  Вершины стены около y={target_y} (±{tolerance_y}), x=[{target_x_min},{target_x_max}]:")
    print(f"    Найдено вершин: {len(matching_verts)}")

    if matching_verts:
        # Границы стены в этой области
        x_coords = [v.co.x for v in matching_verts]
        y_coords = [v.co.y for v in matching_verts]
        z_coords = [v.co.z for v in matching_verts]

        wall_x_min = min(x_coords)
        wall_x_max = max(x_coords)
        wall_y_min = min(y_coords)
        wall_y_max = max(y_coords)
        wall_z_min = min(z_coords)
        wall_z_max = max(z_coords)

        print(f"    X: {wall_x_min:.3f} до {wall_x_max:.3f}  (размер: {wall_x_max - wall_x_min:.3f})")
        print(f"    Y: {wall_y_min:.3f} до {wall_y_max:.3f}  (размер: {wall_y_max - wall_y_min:.3f})")
        print(f"    Z: {wall_z_min:.3f} до {wall_z_max:.3f}  (размер: {wall_z_max - wall_z_min:.3f})")

        # Проверка пересечения (cube_x_min и др. уже определены выше из вершин)
        print(f"\n  Проверка пересечения:")

        x_overlap = (cube_x_min <= wall_x_max) and (cube_x_max >= wall_x_min)
        y_overlap = (cube_y_min <= wall_y_max) and (cube_y_max >= wall_y_min)
        z_overlap = (cube_z_min <= wall_z_max) and (cube_z_max >= wall_z_min)

        print(f"    X пересечение: {'✓' if x_overlap else '✗'}")
        if x_overlap:
            overlap_x_min = max(cube_x_min, wall_x_min)
            overlap_x_max = min(cube_x_max, wall_x_max)
            print(f"      Перекрытие X: {overlap_x_min:.3f} до {overlap_x_max:.3f}")

        print(f"    Y пересечение: {'✓' if y_overlap else '✗'}")
        if y_overlap:
            overlap_y_min = max(cube_y_min, wall_y_min)
            overlap_y_max = min(cube_y_max, wall_y_max)
            print(f"      Перекрытие Y: {overlap_y_min:.3f} до {overlap_y_max:.3f}")

        print(f"    Z пересечение: {'✓' if z_overlap else '✗'}")
        if z_overlap:
            overlap_z_min = max(cube_z_min, wall_z_min)
            overlap_z_max = min(cube_z_max, wall_z_max)
            print(f"      Перекрытие Z: {overlap_z_min:.3f} до {overlap_z_max:.3f}")

        if x_overlap and y_overlap and z_overlap:
            print(f"\n  ✅ Куб ПЕРЕСЕКАЕТ стену!")

            # Проверим размеры перекрытия
            if cube_x_min < wall_x_min:
                print(f"\n  ⚠️  ПРОБЛЕМА: Левый край куба ({cube_x_min:.3f}) выходит ЗА пределы стены ({wall_x_min:.3f})")
                print(f"      Выступ слева: {wall_x_min - cube_x_min:.3f} м")

            if cube_x_max > wall_x_max:
                print(f"  ⚠️  ПРОБЛЕМА: Правый край куба ({cube_x_max:.3f}) выходит ЗА пределы стены ({wall_x_max:.3f})")
                print(f"      Выступ справа: {cube_x_max - wall_x_max:.3f} м")

            # Проверим, не слишком ли большой куб
            cube_width = cube_x_max - cube_x_min
            wall_width = wall_x_max - wall_x_min
            if cube_width > wall_width:
                print(f"\n  ❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Куб ({cube_width:.3f} м) БОЛЬШЕ стены ({wall_width:.3f} м)!")
                print(f"      Это может привести к сбою Boolean операции!")
                print(f"      Рекомендация: уменьшить OPENING_WIDTH_MULTIPLIER или размер куба")
        else:
            print(f"\n  ❌ Куб НЕ пересекает стену!")

    else:
        print("  ❌ Не найдено вершин стены в этой области!")
        print("  Возможно, стена не построена для window_1")


if __name__ == "__main__":
    # Сначала запускаем основной процесс
    print("Запуск основного процесса...")
    build_main()

    # Затем диагностируем window_1
    diagnose_window1()
