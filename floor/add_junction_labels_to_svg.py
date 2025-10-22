#!/usr/bin/env python3
"""
Скрипт для добавления меток внешних junctions на SVG файл плана здания.
Использует данные из JSON файла для извлечения junction IDs из контура здания.
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

def add_junction_labels_to_svg(json_path, svg_path, output_path=None):
    """
    Добавляет метки junction IDs на SVG файл
    
    Args:
        json_path: Путь к JSON файлу с данными о building_outline
        svg_path: Путь к исходному SVG файлу
        output_path: Путь для сохранения обновленного SVG (если None, перезаписывает исходный)
    """
    if output_path is None:
        output_path = svg_path
    
    # 1. Читаем JSON файл и получаем данные о building_outline
    print(f"Чтение JSON файла: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'building_outline' not in data:
        print("Ошибка: в JSON файле отсутствует 'building_outline'")
        return False
    
    building_outline = data['building_outline']
    if 'vertices' not in building_outline:
        print("Ошибка: в building_outline отсутствуют 'vertices'")
        return False
    
    # 2. Извлекаем координаты и ID junctions из вершин
    vertices = building_outline['vertices']
    junction_points = []
    
    for vertex in vertices:
        if 'junction_id' in vertex and 'junction_name' in vertex:
            junction_points.append({
                'x': vertex['x'],
                'y': vertex['y'],
                'id': vertex['junction_id'],
                'name': vertex['junction_name'],
                'type': vertex.get('junction_type', 'unknown')
            })
    
    print(f"Найдено {len(junction_points)} junctions для отображения")
    
    # Определяем стартовую точку (первая вершина в списке)
    if vertices:
        start_junction = vertices[0]
        if 'junction_id' in start_junction:
            print(f"Стартовая точка: J{start_junction['junction_id']} в координатах ({start_junction['x']}, {start_junction['y']})")
    
    # 3. Читаем SVG файл
    print(f"Чтение SVG файла: {svg_path}")
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Определяем пространство имен SVG
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Регистрируем пространство имен для создания элементов
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        
    except Exception as e:
        print(f"Ошибка при чтении SVG файла: {e}")
        return False
    
    # 4. Добавляем группу для junction labels
    # Проверяем, есть ли уже группа для junction labels
    junction_group = root.find(".//g[@id='junction_labels']")
    if junction_group is None:
        # Создаем новую группу с правильным пространством имен
        junction_group = ET.SubElement(root, '{http://www.w3.org/2000/svg}g', id='junction_labels')
    
    # 5. Добавляем текстовые элементы для каждого junction
    scale_factor = data.get('metadata', {}).get('scale_factor', 1.0)
    
    # Сначала добавляем стрелки направления обхода
    for i in range(len(vertices)):
        current = vertices[i]
        next_vertex = vertices[(i + 1) % len(vertices)]  # Замыкаем контур
        
        if 'junction_id' in current and 'junction_id' in next_vertex:
            # Масштабируем координаты
            x1 = float(current['x']) * scale_factor
            y1 = float(current['y']) * scale_factor
            x2 = float(next_vertex['x']) * scale_factor
            y2 = float(next_vertex['y']) * scale_factor
            
            # Вычисляем середину отрезка
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Вычисляем направление вектора
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            
            if length > 0:
                # Нормализуем вектор
                dx /= length
                dy /= length
                
                # Создаем стрелку направления
                arrow_len = 15
                arrow_width = 8
                
                # Кончик стрелки
                tip_x = mid_x + dx * arrow_len/2
                tip_y = mid_y + dy * arrow_len/2
                
                # Основание стрелки
                base_x = mid_x - dx * arrow_len/2
                base_y = mid_y - dy * arrow_len/2
                
                # Боковые точки стрелки
                perp_dx = -dy * arrow_width/2
                perp_dy = dx * arrow_width/2
                
                left_x = base_x + perp_dx
                left_y = base_y + perp_dy
                
                right_x = base_x - perp_dx
                right_y = base_y - perp_dy
                
                # Создаем полигон для стрелки с правильным пространством имен
                arrow_points = f"{tip_x},{tip_y} {left_x},{left_y} {right_x},{right_y}"
                arrow_elem = ET.SubElement(junction_group, '{http://www.w3.org/2000/svg}polygon', {
                    'points': arrow_points,
                    'fill': '#00AA00',  # Зеленый цвет для стрелок
                    'stroke': '#005500',
                    'stroke-width': '0.5',
                    'opacity': '0.7'
                })
    
    # Теперь добавляем метки junction
    for junction in junction_points:
        # Масштабируем координаты
        x = float(junction['x']) * scale_factor
        y = float(junction['y']) * scale_factor
        
        # Определяем, является ли это стартовой точкой
        is_start = (junction['id'] == start_junction['junction_id'])
        
        # Выбираем цвет в зависимости от типа junction
        if is_start:
            circle_color = '#8B00FF'  # Фиолетовый для стартовой точки
            circle_radius = '6'
        else:
            circle_color = '#FF0000'  # Красный для остальных
            circle_radius = '4'
        
        # Создаем текстовый элемент с правильным пространством имен
        text_elem = ET.SubElement(junction_group, '{http://www.w3.org/2000/svg}text', {
            'x': str(x),
            'y': str(y - 10),  # Смещаем текст немного выше точки
            'text-anchor': 'middle',
            'fill': '#FF0000',  # Красный цвет для текста
            'font-size': '8',
            'font-weight': 'bold',
            'font-family': 'Arial',
            'stroke': '#FFFFFF',  # Белая обводка для лучшей видимости
            'stroke-width': '0.5'
        })
        
        # Устанавливаем текстовое содержимое (ID junction)
        text_elem.text = f"J{junction['id']}"
        
        # Добавляем круг в месте расположения junction с правильным пространством имен
        circle_elem = ET.SubElement(junction_group, '{http://www.w3.org/2000/svg}circle', {
            'cx': str(x),
            'cy': str(y),
            'r': circle_radius,
            'fill': circle_color,
            'stroke': '#FFFFFF',
            'stroke-width': '1'
        })
        
        # Для стартовой точки добавляем дополнительную метку
        if is_start:
            start_text_elem = ET.SubElement(junction_group, '{http://www.w3.org/2000/svg}text', {
                'x': str(x),
                'y': str(y - 20),  # Еще выше основной метки
                'text-anchor': 'middle',
                'fill': '#8B00FF',  # Фиолетовый цвет
                'font-size': '10',
                'font-weight': 'bold',
                'font-family': 'Arial'
            })
            start_text_elem.text = "START"
        
        print(f"Добавлен junction J{junction['id']} в координатах ({x:.1f}, {y:.1f})" +
              (" (СТАРТ)" if is_start else ""))
    
    # 6. Сохраняем обновленный SVG файл
    print(f"Сохранение обновленного SVG файла: {output_path}")
    
    # Используем minidom для красивого форматирования
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(reparsed.toprettyxml(indent="  "))
    
    print("Готово! Junction IDs успешно добавлены на SVG")
    return True

def main():
    parser = argparse.ArgumentParser(description='Добавление junction IDs на SVG файл')
    parser.add_argument('--json', default='plan_floor1_objects.json', 
                       help='Путь к JSON файлу с данными (по умолчанию: plan_floor1_objects.json)')
    parser.add_argument('--svg', default='plan_floor1_objects_colored.svg',
                       help='Путь к SVG файлу (по умолчанию: plan_floor1_objects_colored.svg)')
    parser.add_argument('--output', 
                       help='Путь для сохранения результата (по умолчанию перезаписывает исходный SVG)')
    
    args = parser.parse_args()
    
    success = add_junction_labels_to_svg(args.json, args.svg, args.output)
    if not success:
        exit(1)

if __name__ == '__main__':
    main()