#!/usr/bin/env python3
"""
Анализ прямоугольников стен БЕЗ Blender - читает JSON напрямую
"""

import json
import sys

SCALE_FACTOR = 0.01

# Область window_1 (после масштабирования 0.01)
WINDOW1_X_MIN = 0.0
WINDOW1_X_MAX = 3.2
WINDOW1_Y_MIN = 9.8
WINDOW1_Y_MAX = 10.4

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_json_structure(json_path):
    print("=" * 70)
    print("АНАЛИЗ СТРУКТУРЫ JSON ДЛЯ WINDOW_1")
    print("=" * 70)

    data = load_json(json_path)

    # Найдем window_1
    window_1 = None
    for opening in data.get('openings', []):
        if opening.get('id') == 'window_1':
            window_1 = opening
            break

    if not window_1:
        print("❌ window_1 не найден в openings!")
        return

    print(f"\n✓ window_1 найден:")
    bbox = window_1.get('bbox', {})
    print(f"  Тип: {window_1.get('type')}")
    print(f"  Bbox (пиксели): x={bbox.get('x')}, y={bbox.get('y')}, width={bbox.get('width')}, height={bbox.get('height')}")
    print(f"  Bbox (метры): x={bbox.get('x')*SCALE_FACTOR:.3f}, y={bbox.get('y')*SCALE_FACTOR:.3f}, width={bbox.get('width')*SCALE_FACTOR:.3f}, height={bbox.get('height')*SCALE_FACTOR:.3f}")

    # Найдем стеновые сегменты для window_1
    print(f"\n🔍 Сегменты стен для window_1:")

    window_1_segments = []
    for seg in data.get('wall_segments_from_openings', []):
        if seg.get('opening_id') == 'window_1':
            window_1_segments.append(seg)

    print(f"  Найдено сегментов: {len(window_1_segments)}")

    for seg in window_1_segments:
        seg_bbox = seg.get('bbox', {})
        print(f"\n    {seg.get('segment_id')}")
        print(f"      Edge: {seg.get('edge_side')}")
        print(f"      Junctions: {seg.get('start_junction_id')} → {seg.get('end_junction_id')}")
        print(f"      Bbox (px): x={seg_bbox.get('x')}, y={seg_bbox.get('y')}, w={seg_bbox.get('width')}, h={seg_bbox.get('height')}")
        print(f"      Bbox (m): x={seg_bbox.get('x')*SCALE_FACTOR:.3f}, y={seg_bbox.get('y')*SCALE_FACTOR:.3f}, w={seg_bbox.get('width')*SCALE_FACTOR:.3f}, h={seg_bbox.get('height')*SCALE_FACTOR:.3f}")

    # Проверим, какие угловые вершины (corners) есть в области window_1
    print(f"\n🔍 Угловые вершины (corners) в области window_1:")

    outline = data.get('building_outline', {})
    if not outline:
        print("  ❌ building_outline не найден!")
        return

    vertices = outline.get('vertices', [])
    corner_vertices = [v for v in vertices if v.get('corner', 0) == 1]

    print(f"  Всего угловых вершин в контуре: {len(corner_vertices)}")

    corners_in_area = []
    for v in corner_vertices:
        x = v.get('x', 0) * SCALE_FACTOR
        y = v.get('y', 0) * SCALE_FACTOR

        if WINDOW1_X_MIN <= x <= WINDOW1_X_MAX and WINDOW1_Y_MIN <= y <= WINDOW1_Y_MAX:
            corners_in_area.append(v)
            print(f"\n    ✓ J{v.get('junction_id')} (corner={v.get('corner')})")
            print(f"      Позиция: ({x:.3f}, {y:.3f})")
            print(f"      Тип: {v.get('junction_type')}")

    if len(corners_in_area) == 0:
        print(f"\n  ✅ НЕТ угловых вершин в области window_1 - это правильно!")
        print(f"     Стена должна быть простым горизонтальным прямоугольником.")
    else:
        print(f"\n  ⚠️  {len(corners_in_area)} угловых вершин в области window_1!")
        print(f"     Это может привести к созданию ЛИШНИХ прямоугольников стен!")

    # Проверим junction-based сегменты в этой области
    print(f"\n🔍 Junction-based сегменты в области window_1:")

    junction_segs_in_area = []
    for seg in data.get('wall_segments_from_junctions', []):
        seg_bbox = seg.get('bbox', {})
        if not seg_bbox:
            continue

        x = seg_bbox.get('x', 0) * SCALE_FACTOR
        y = seg_bbox.get('y', 0) * SCALE_FACTOR
        w = seg_bbox.get('width', 0) * SCALE_FACTOR
        h = seg_bbox.get('height', 0) * SCALE_FACTOR

        # Проверим пересечение
        seg_x_min = x
        seg_x_max = x + w
        seg_y_min = y
        seg_y_max = y + h

        x_overlap = (seg_x_min <= WINDOW1_X_MAX) and (seg_x_max >= WINDOW1_X_MIN)
        y_overlap = (seg_y_min <= WINDOW1_Y_MAX) and (seg_y_max >= WINDOW1_Y_MIN)

        if x_overlap and y_overlap:
            junction_segs_in_area.append(seg)
            print(f"\n    {seg.get('segment_id')}")
            print(f"      Junctions: {seg.get('start_junction_id')} → {seg.get('end_junction_id')}")
            print(f"      Bbox: x={x:.3f}-{x+w:.3f}, y={y:.3f}-{y+h:.3f}")
            print(f"      Размер: {w:.3f} x {h:.3f}")

    print(f"\n  Найдено junction-based сегментов: {len(junction_segs_in_area)}")

    # ИТОГОВЫЙ АНАЛИЗ
    print(f"\n" + "=" * 70)
    print("ИТОГ:")
    print("=" * 70)

    total_wall_elements = len(window_1_segments) + len(corners_in_area) + len(junction_segs_in_area)
    print(f"  Сегментов от window_1: {len(window_1_segments)} (левый + правый)")
    print(f"  Угловых вершин в области: {len(corners_in_area)}")
    print(f"  Junction сегментов в области: {len(junction_segs_in_area)}")
    print(f"  ИТОГО элементов стен: {total_wall_elements}")

    if len(corners_in_area) > 0:
        print(f"\n🔴 ПРОБЛЕМА: Угловые вершины в области window_1!")
        print(f"   Алгоритм create_rectangles строит прямоугольники между")
        print(f"   соседними углами, что создаст ПЕРЕКРЫВАЮЩИЕСЯ стены!")
        print(f"\n   РЕШЕНИЕ: Нужно модифицировать логику построения, чтобы")
        print(f"   пропускать прямоугольники, которые полностью перекрываются")
        print(f"   сегментами от openings.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        json_path = "blender/2_wall_coordinates_inverted.json"
    else:
        json_path = sys.argv[1]

    analyze_json_structure(json_path)
