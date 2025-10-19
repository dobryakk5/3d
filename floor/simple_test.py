#!/usr/bin/env python3
"""
Простой тест для проверки импорта функций
"""
import sys
import os

# Добавляем путь к текущей директории для импорта модуля
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

result = []

try:
    from improved_junction_type_analyzer import analyze_polygon_extensions_with_thickness, is_point_in_polygon
    result.append("✓ Improved junction type analyzer imported successfully")
except ImportError as e:
    result.append(f"✗ Error importing improved_junction_type_analyzer: {e}")

try:
    from visualize_polygons_align import align_walls_by_openings
    result.append("✓ Visualize polygons align imported successfully")
except ImportError as e:
    result.append(f"✗ Error importing visualize_polygons_align: {e}")

try:
    from visualize_polygons import (
        process_l_junction_extensions,
        extend_segment_to_polygon_edge,
        extend_segment_to_perpendicular_x,
        find_l_junctions
    )
    result.append("✓ L-junction functions imported successfully")
    result.append("✓ Available functions:")
    result.append("  - process_l_junction_extensions")
    result.append("  - extend_segment_to_polygon_edge")
    result.append("  - extend_segment_to_perpendicular_x")
    result.append("  - find_l_junctions")
    
except ImportError as e:
    result.append(f"✗ Error importing L-junction functions: {e}")

# Записываем результат в файл
with open('test_result.txt', 'w') as f:
    for line in result:
        f.write(line + '\n')

print("Результаты записаны в test_result.txt")