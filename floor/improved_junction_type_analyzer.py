#!/usr/bin/env python3
"""
Улучшенный анализатор типов junctions с учетом расстояний до краев полигона
"""

import json
import math
import sys
import os

# Добавляем путь к текущей директории для импорта модуля
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from junction_type_analyzer import is_point_in_polygon, point_to_segment_distance

def analyze_polygon_extensions_with_thickness(junction, polygon_vertices, wall_thickness=20.0):
    """
    Анализирует расширения полигона с учетом толщины стены
    
    Args:
        junction: Точка junction с координатами x, y
        polygon_vertices: Вершины полигона
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Словарь с информацией о значимых расширениях
    """
    jx, jy = junction['x'], junction['y']
    
    # Находим точки пересечения с ребрами полигона
    intersections = {
        'left': None,
        'right': None,
        'up': None,
        'down': None
    }
    
    n = len(polygon_vertices)
    
    for i in range(n):
        v1 = polygon_vertices[i]
        v2 = polygon_vertices[(i + 1) % n]
        
        x1, y1 = v1['x'], v1['y']
        x2, y2 = v2['x'], v2['y']
        
        # Используем более точную проверку с учетом погрешности
        epsilon = 0.001
        
        # Проверяем пересечение с горизонтальной линией на уровне jy
        y_in_range = (min(y1, y2) - epsilon <= jy <= max(y1, y2) + epsilon)
        
        if y_in_range and abs(y1 - y2) > epsilon:
            # Находим x координату пересечения
            t = (jy - y1) / (y2 - y1)
            x_intersect = x1 + t * (x2 - x1)
            
            # Проверяем, находится ли точка пересечения на отрезке
            if 0 <= t <= 1:
                if x_intersect < jx - epsilon and (intersections['left'] is None or x_intersect > intersections['left']):
                    intersections['left'] = x_intersect
                elif x_intersect > jx + epsilon and (intersections['right'] is None or x_intersect < intersections['right']):
                    intersections['right'] = x_intersect
        
        # Проверяем пересечение с вертикальной линией на уровне jx
        x_in_range = (min(x1, x2) - epsilon <= jx <= max(x1, x2) + epsilon)
        
        if x_in_range and abs(x1 - x2) > epsilon:
            # Находим y координату пересечения
            t = (jx - x1) / (x2 - x1)
            y_intersect = y1 + t * (y2 - y1)
            
            # Проверяем, находится ли точка пересечения на отрезке
            if 0 <= t <= 1:
                if y_intersect > jy + epsilon and (intersections['down'] is None or y_intersect < intersections['down']):
                    intersections['down'] = y_intersect
                elif y_intersect < jy - epsilon and (intersections['up'] is None or y_intersect > intersections['up']):
                    intersections['up'] = y_intersect
    
    # Определяем значимые расширения (учитывая толщину стены)
    significant_extensions = {}
    
    if intersections['left'] is not None:
        distance_left = jx - intersections['left']
        significant_extensions['left'] = distance_left > wall_thickness
    
    if intersections['right'] is not None:
        distance_right = intersections['right'] - jx
        significant_extensions['right'] = distance_right > wall_thickness
    
    if intersections['up'] is not None:
        distance_up = jy - intersections['up']  # up = меньшее значение Y
        significant_extensions['up'] = distance_up > wall_thickness
    
    if intersections['down'] is not None:
        distance_down = intersections['down'] - jy  # down = большее значение Y
        significant_extensions['down'] = distance_down > wall_thickness
    
    return {
        'intersections': intersections,
        'significant_extensions': significant_extensions,
        'distances': {
            'left': jx - intersections['left'] if intersections['left'] is not None else 0,
            'right': intersections['right'] - jx if intersections['right'] is not None else 0,
            'up': jy - intersections['up'] if intersections['up'] is not None else 0,  # up = меньшее значение Y
            'down': intersections['down'] - jy if intersections['down'] is not None else 0  # down = большее значение Y
        }
    }

def determine_junction_type_with_thickness(junction, polygon_vertices, wall_thickness=20.0):
    """
    Определяет тип junction с учетом толщины стены
    
    Args:
        junction: Точка junction с координатами x, y
        polygon_vertices: Вершины полигона
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Тип junction ('L', 'T', 'X', 'straight', 'unknown')
    """
    # Проверяем, находится ли junction внутри полигона
    if not is_point_in_polygon(junction['x'], junction['y'], polygon_vertices):
        return 'unknown'
    
    # Анализируем расширения с учетом толщины
    analysis = analyze_polygon_extensions_with_thickness(junction, polygon_vertices, wall_thickness)
    extensions = analysis['significant_extensions']
    distances = analysis['distances']
    
    # Подсчитываем количество значимых расширений
    significant_directions = [direction for direction, is_significant in extensions.items() if is_significant]
    count = len(significant_directions)
    
    print(f"  Анализ расширений (толщина стены: {wall_thickness}):")
    for direction in ['left', 'right', 'up', 'down']:
        if direction in extensions:
            status = "✓" if extensions[direction] else "✗"
            print(f"    {direction}: {distances[direction]:.1f} пикселей {status}")
    
    print(f"  Значимые направления: {', '.join(significant_directions)} (всего: {count})")
    
    # Определяем тип junction на основе количества значимых расширений
    if count == 0:
        return 'unknown'
    elif count == 1:
        return 'unknown'  # Точка внутри полигона, но только одно направление
    elif count == 2:
        # Проверяем, противоположные ли направления
        if ('left' in significant_directions and 'right' in significant_directions):
            return 'straight'  # Горизонтальная стена
        elif ('up' in significant_directions and 'down' in significant_directions):
            return 'straight'  # Вертикальная стена
        else:
            return 'L'  # L-образное соединение
    elif count == 3:
        return 'T'  # T-образное соединение
    elif count == 4:
        return 'X'  # X-образное соединение
    else:
        return 'unknown'

def test_j15_with_thickness():
    """Тестируем определение типа J15 с учетом толщины стены"""
    print("="*60)
    print("ТЕСТ ОПРЕДЕЛЕНИЯ ТИПА J15 С УЧЕТОМ ТОЛЩИНЫ СТЕНЫ")
    print("="*60)
    
    # Загружаем данные
    with open('plan_floor1_objects.json', 'r') as f:
        data = json.load(f)
    
    # Находим J15 (индекс 14)
    j15 = data['junctions'][14]
    print(f"J15 координаты: ({j15['x']}, {j15['y']})")
    
    # Находим wall_polygon_7
    wall_polygon_7 = None
    for wall in data['wall_polygons']:
        if wall.get('id') == 'wall_polygon_7':
            wall_polygon_7 = wall
            break
    
    print(f"\nАнализ wall_polygon_7")
    
    # Проверяем, находится ли J15 внутри полигона
    is_inside = is_point_in_polygon(j15['x'], j15['y'], wall_polygon_7['vertices'])
    print(f"J15 находится внутри wall_polygon_7: {is_inside}")
    
    if not is_inside:
        print("J15 не находится внутри полигона")
        return
    
    # Тестируем с разной толщиной стены
    for thickness in [10.0, 15.0, 20.0, 25.0]:
        print(f"\nТест с толщиной стены: {thickness} пикселей")
        junction_type = determine_junction_type_with_thickness(j15, wall_polygon_7['vertices'], thickness)
        print(f"  Определенный тип junction: {junction_type}")

if __name__ == '__main__':
    test_j15_with_thickness()