#!/usr/bin/env python3
"""
Диагностика: какие прямоугольники стен создаются в области window_1
"""

import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from create_outline_with_opening_cubes import create_rectangles, load_json_data

# Область window_1 (после масштабирования 0.01)
# x: 0.445 - 2.715 (window_1 bbox x=44.5, width=227)
# y: 10.005 - 10.215 (window_1 bbox y=1000.5, height=21)

WINDOW1_X_MIN = 0.4
WINDOW1_X_MAX = 2.8
WINDOW1_Y_MIN = 9.9
WINDOW1_Y_MAX = 10.3

def analyze_rectangles(json_path):
    print("=" * 70)
    print("АНАЛИЗ ПРЯМОУГОЛЬНИКОВ СТЕН В ОБЛАСТИ WINDOW_1")
    print("=" * 70)

    data = load_json_data(json_path)
    rectangles = create_rectangles(data)

    print(f"\nВсего прямоугольников: {len(rectangles)}")
    print(f"\nОбласть window_1:")
    print(f"  X: {WINDOW1_X_MIN:.3f} - {WINDOW1_X_MAX:.3f}")
    print(f"  Y: {WINDOW1_Y_MIN:.3f} - {WINDOW1_Y_MAX:.3f}")

    overlapping = []

    for i, rect in enumerate(rectangles):
        corners = rect['corners']

        # Найдем bbox прямоугольника
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]

        rect_x_min = min(x_coords)
        rect_x_max = max(x_coords)
        rect_y_min = min(y_coords)
        rect_y_max = max(y_coords)

        # Проверим пересечение с областью window_1
        x_overlap = (rect_x_min <= WINDOW1_X_MAX) and (rect_x_max >= WINDOW1_X_MIN)
        y_overlap = (rect_y_min <= WINDOW1_Y_MAX) and (rect_y_max >= WINDOW1_Y_MIN)

        if x_overlap and y_overlap:
            overlapping.append({
                'index': i,
                'id': rect.get('id', 'unknown'),
                'x_min': rect_x_min,
                'x_max': rect_x_max,
                'y_min': rect_y_min,
                'y_max': rect_y_max,
                'orientation': rect.get('orientation', 'unknown'),
                'corners': corners
            })

    print(f"\n🔍 Прямоугольников, пересекающих область window_1: {len(overlapping)}")

    if len(overlapping) == 0:
        print("\n❌ НЕТ стен в области window_1! Проблема в построении контура.")
        return

    for i, r in enumerate(overlapping):
        print(f"\n  [{i+1}] {r['id']}")
        print(f"      X: {r['x_min']:.3f} - {r['x_max']:.3f}  (ширина: {r['x_max'] - r['x_min']:.3f})")
        print(f"      Y: {r['y_min']:.3f} - {r['y_max']:.3f}  (толщина: {r['y_max'] - r['y_min']:.3f})")
        print(f"      Ориентация: {r['orientation']}")
        print(f"      Углы: {r['corners']}")

    # Проверим перекрытия между прямоугольниками
    if len(overlapping) > 1:
        print(f"\n⚠️  ВНИМАНИЕ: {len(overlapping)} прямоугольников в одной области!")
        print(f"Проверяю перекрытия...")

        for i in range(len(overlapping)):
            for j in range(i + 1, len(overlapping)):
                r1 = overlapping[i]
                r2 = overlapping[j]

                # Проверим, полностью ли перекрываются прямоугольники
                x_overlap_min = max(r1['x_min'], r2['x_min'])
                x_overlap_max = min(r1['x_max'], r2['x_max'])
                y_overlap_min = max(r1['y_min'], r2['y_min'])
                y_overlap_max = min(r1['y_max'], r2['y_max'])

                if x_overlap_min < x_overlap_max and y_overlap_min < y_overlap_max:
                    overlap_area = (x_overlap_max - x_overlap_min) * (y_overlap_max - y_overlap_min)
                    r1_area = (r1['x_max'] - r1['x_min']) * (r1['y_max'] - r1['y_min'])
                    r2_area = (r2['x_max'] - r2['x_min']) * (r2['y_max'] - r2['y_min'])

                    print(f"\n  ❌ ПЕРЕКРЫТИЕ между [{i+1}] и [{j+1}]:")
                    print(f"      Область: X={x_overlap_min:.3f}-{x_overlap_max:.3f}, Y={y_overlap_min:.3f}-{y_overlap_max:.3f}")
                    print(f"      Площадь перекрытия: {overlap_area:.3f}")
                    print(f"      % от [{i+1}]: {100*overlap_area/r1_area:.1f}%")
                    print(f"      % от [{j+1}]: {100*overlap_area/r2_area:.1f}%")

                    if overlap_area / min(r1_area, r2_area) > 0.8:
                        print(f"      🔴 ДУБЛИКАТ! Прямоугольники почти полностью перекрываются!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python debug_wall_rectangles.py <path_to_json>")
        sys.exit(1)

    analyze_rectangles(sys.argv[1])
