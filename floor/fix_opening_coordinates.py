#!/usr/bin/env python3
"""
Скрипт для исправления проблемы с зазором между стеной и дверью D1
"""

import re

def fix_visualize_polygons():
    """Исправляет файл visualize_polygons_with_json.py"""
    
    # Читаем файл
    with open('visualize_polygons_with_json.py', 'r') as f:
        content = f.read()
    
    # Находим и заменяем вызов add_opening_to_json
    pattern = r'(\s+# Добавляем проем в JSON\s+if json_data:\s+)(add_opening_to_json\(json_data, opening, edge_junctions\))'
    
    def replacement(match):
        indent = match.group(1)
        original_call = match.group(2)
        return f'{indent}# Переносим вызов add_opening_to_json после construct_wall_segment_from_opening\n{indent}# чтобы сохранять измененные координаты проема\n{indent}# {original_call}\n'
    
    content = re.sub(pattern, replacement, content)
    
    # Находим строку с all_wall_segments.extend(wall_segments)
    pattern = r'(\s+)(all_wall_segments\.extend\(wall_segments\))'
    
    def replacement2(match):
        indent = match.group(1)
        extend_call = match.group(2)
        return f'{indent}{extend_call}\n\n{indent}# Добавляем проем в JSON с измененными координатами\n{indent}if json_data:\n{indent}    add_opening_to_json(json_data, {{\"id\": opening_id, \"type\": opening_type, \"bbox\": opening_with_junction.bbox}}, edge_junctions)\n'
    
    content = re.sub(pattern, replacement2, content)
    
    # Записываем измененный файл
    with open('visualize_polygons_with_json.py', 'w') as f:
        f.write(content)
    
    print("Файл visualize_polygons_with_json.py исправлен")

if __name__ == "__main__":
    fix_visualize_polygons()
