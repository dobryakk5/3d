#!/usr/bin/env python3
"""
Скрипт для визуализации типов junctions с использованием improved_junction_type_analyzer.py
"""

import json
import sys
import os
import xml.etree.ElementTree as ET

# Добавляем путь к текущей директории для импорта модуля
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_junction_type_analyzer import analyze_polygon_extensions_with_thickness, is_point_in_polygon

def analyze_all_junctions_improved(data, wall_thickness=20.0):
    """
    Анализирует все junctions с использованием улучшенной логики
    
    Args:
        data: Данные плана с junctions и wall_polygons
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Список junctions с определенными типами
    """
    junctions = data.get('junctions', [])
    wall_polygons = data.get('wall_polygons', [])
    
    analyzed_junctions = []
    
    print(f"Анализ {len(junctions)} junctions...")
    
    for idx, junction in enumerate(junctions):
        # Находим полигон, содержащий junction
        containing_polygon = None
        for wall in wall_polygons:
            vertices = wall.get('vertices', [])
            if vertices and is_point_in_polygon(junction['x'], junction['y'], vertices):
                containing_polygon = wall
                break
        
        if containing_polygon:
            # Определяем тип с учетом толщины стены
            analysis = analyze_polygon_extensions_with_thickness(
                junction, containing_polygon['vertices'], wall_thickness
            )
            extensions = analysis['significant_extensions']
            distances = analysis['distances']
            
            # Подсчитываем количество значимых расширений
            significant_directions = [direction for direction, is_significant in extensions.items() if is_significant]
            count = len(significant_directions)
            
            # Определяем тип junction на основе количества значимых расширений
            if count == 0:
                junction_type = 'unknown'
            elif count == 1:
                junction_type = 'unknown'
            elif count == 2:
                if ('left' in significant_directions and 'right' in significant_directions):
                    junction_type = 'straight'
                elif ('up' in significant_directions and 'down' in significant_directions):
                    junction_type = 'straight'
                else:
                    junction_type = 'L'
            elif count == 3:
                junction_type = 'T'
            elif count == 4:
                junction_type = 'X'
            else:
                junction_type = 'unknown'
            
            analyzed_junctions.append({
                'junction': junction,
                'type': junction_type,
                'polygon': containing_polygon,
                'directions': significant_directions,
                'distances': distances
            })
            
            print(f"  Junction {idx+1}: {junction_type} (направления: {', '.join(significant_directions)})")
        else:
            # Junction не внутри полигона
            analyzed_junctions.append({
                'junction': junction,
                'type': 'unknown',
                'polygon': None,
                'directions': [],
                'distances': {}
            })
            
            print(f"  Junction {idx+1}: unknown (не внутри полигона)")
    
    return analyzed_junctions

def create_junctions_svg(analyzed_junctions, input_svg, output_svg):
    """
    Создает SVG с визуализацией junctions
    
    Args:
        analyzed_junctions: Список проанализированных junctions
        input_svg: Путь к исходному SVG файлу
        output_svg: Путь для сохранения результата
    """
    # Цвета для разных типов junctions
    type_colors = {
        'L': '#ff0000',      # Красный
        'T': '#00ff00',      # Зеленый
        'X': '#0000ff',      # Синий
        'straight': '#ffff00',  # Желтый
        'unknown': '#888888'   # Серый
    }
    
    # Загружаем SVG
    tree = ET.parse(input_svg)
    root = tree.getroot()
    
    # Добавляем стили
    style = ET.SubElement(root, 'style')
    style.text = """
    .junction-label {
        font-family: Arial, sans-serif;
        font-size: 12px;
        font-weight: bold;
        text-anchor: middle;
        dominant-baseline: middle;
    }
    .junction-point {
        stroke-width: 2;
        fill-opacity: 0.7;
    }
    """
    
    # Создаем группу для junctions
    junctions_group = ET.SubElement(root, 'g')
    junctions_group.set('id', 'junctions')
    
    # Добавляем каждый junction
    for idx, analyzed in enumerate(analyzed_junctions):
        junction = analyzed['junction']
        jx, jy = junction['x'], junction['y']
        jtype = analyzed['type']
        
        # Определяем цвет
        color = type_colors.get(jtype, '#888888')
        
        # Добавляем точку
        circle = ET.SubElement(junctions_group, 'circle')
        circle.set('cx', str(jx))
        circle.set('cy', str(jy))
        circle.set('r', '5')
        circle.set('fill', color)
        circle.set('class', 'junction-point')
        circle.set('stroke', '#000000')
        
        # Добавляем метку с типом
        text = ET.SubElement(junctions_group, 'text')
        text.set('x', str(jx))
        text.set('y', str(jy - 10))
        text.set('class', 'junction-label')
        text.set('fill', color)
        text.text = jtype
        
        # Добавляем метку с ID
        text_id = ET.SubElement(junctions_group, 'text')
        text_id.set('x', str(jx))
        text_id.set('y', str(jy + 15))
        text_id.set('class', 'junction-label')
        text_id.set('fill', '#000000')
        text_id.set('font-size', '8px')
        text_id.text = f"J{idx+1}"
    
    # Добавляем легенду
    legend_group = ET.SubElement(root, 'g')
    legend_group.set('id', 'legend')
    
    legend_x = 50
    legend_y = 50
    
    legend_title = ET.SubElement(legend_group, 'text')
    legend_title.set('x', str(legend_x))
    legend_title.set('y', str(legend_y))
    legend_title.set('font-family', 'Arial, sans-serif')
    legend_title.set('font-size', '14px')
    legend_title.set('font-weight', 'bold')
    legend_title.text = 'Junction Types:'
    
    for i, (jtype, color) in enumerate(type_colors.items()):
        y_pos = legend_y + 20 + i * 20
        
        # Кружок с цветом
        legend_circle = ET.SubElement(legend_group, 'circle')
        legend_circle.set('cx', str(legend_x + 10))
        legend_circle.set('cy', str(y_pos))
        legend_circle.set('r', '5')
        legend_circle.set('fill', color)
        legend_circle.set('stroke', '#000000')
        
        # Текст с типом
        legend_text = ET.SubElement(legend_group, 'text')
        legend_text.set('x', str(legend_x + 20))
        legend_text.set('y', str(y_pos + 5))
        legend_text.set('font-family', 'Arial, sans-serif')
        legend_text.set('font-size', '12px')
        legend_text.text = f"{jtype} - {jtype} junction"
    
    # Сохраняем результат
    tree.write(output_svg, encoding='utf-8', xml_declaration=True)
    print(f"Результат сохранен в: {output_svg}")

def main():
    print("="*60)
    print("ВИЗУАЛИЗАЦИЯ JUNCTIONS С УЛУЧШЕННОЙ ЛОГИКОЙ")
    print("="*60)
    
    # Загружаем данные
    with open('plan_floor1_objects.json', 'r') as f:
        data = json.load(f)
    
    # Анализируем junctions
    analyzed_junctions = analyze_all_junctions_improved(data, wall_thickness=20.0)
    
    # Создаем визуализацию
    create_junctions_svg(
        analyzed_junctions,
        'wall_polygons_opening_based.svg',
        'wall_polygons_with_improved_junction_types.svg'
    )
    
    # Выводим статистику
    type_counts = {}
    for analyzed in analyzed_junctions:
        jtype = analyzed['type']
        type_counts[jtype] = type_counts.get(jtype, 0) + 1
    
    print("\nСтатистика по типам junctions:")
    for jtype, count in type_counts.items():
        print(f"  {jtype}: {count}")
    
    print(f"\nВсего junctions: {len(analyzed_junctions)}")
    
    # Проверяем J15
    j15 = analyzed_junctions[14]
    print(f"\nJ15 (индекс 15):")
    print(f"  Тип: {j15['type']}")
    print(f"  Направления: {j15['directions']}")
    print(f"  Расстояния: {j15['distances']}")
    
    if j15['type'] == 'T':
        print("  ✅ УСПЕХ: J15 корректно определен как T-junction")
    else:
        print(f"  ❌ ПРОБЛЕМА: J15 определяется как {j15['type']} вместо T")

if __name__ == '__main__':
    main()