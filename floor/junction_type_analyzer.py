#!/usr/bin/env python3
"""
Анализатор типов junctions (L, T, X) на основе wall_polygons
Определяет тип каждого junction на основе количества и ориентации соединенных стен
"""

import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class JunctionWithConnections:
    junction: Dict
    junction_type: str  # 'L', 'T', 'X', or 'unknown'
    connected_walls: List[Dict]
    wall_orientations: List[str]  # 'horizontal' or 'vertical'
    connection_count: int

def load_data(json_path: str) -> Dict:
    """Загружает данные из JSON файла"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_wall_orientation(wall_segment: Dict) -> str:
    """
    Анализирует ориентацию сегмента стены
    
    Args:
        wall_segment: Сегмент стены с bbox, vertices, start/end
    
    Returns:
        'horizontal', 'vertical', или 'unknown'
    """
    # Проверяем простые стены (отрезки)
    if 'start' in wall_segment and 'end' in wall_segment:
        start = wall_segment['start']
        end = wall_segment['end']
        
        dx = abs(end['x'] - start['x'])
        dy = abs(end['y'] - start['y'])
        
        if dx > dy * 1.5:
            return 'horizontal'
        elif dy > dx * 1.5:
            return 'vertical'
        else:
            return 'unknown'
    
    # Проверяем стены с bbox
    if 'bbox' in wall_segment:
        bbox = wall_segment['bbox']
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        # Определяем ориентацию по соотношению сторон
        if width > height * 1.5:
            return 'horizontal'
        elif height > width * 1.5:
            return 'vertical'
        else:
            # Для квадратных сегментов смотрим на vertices
            if 'vertices' in wall_segment:
                return analyze_orientation_from_vertices(wall_segment['vertices'])
            return 'unknown'
    
    elif 'vertices' in wall_segment:
        return analyze_orientation_from_vertices(wall_segment['vertices'])
    
    return 'unknown'

def analyze_orientation_from_vertices(vertices: List[Dict]) -> str:
    """
    Анализирует ориентацию по вершинам полигона
    """
    if len(vertices) < 2:
        return 'unknown'
    
    # Вычисляем общее направление по всем ребрам
    total_dx = 0
    total_dy = 0
    
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]
        
        dx = v2['x'] - v1['x']
        dy = v2['y'] - v1['y']
        
        total_dx += abs(dx)
        total_dy += abs(dy)
    
    # Определяем преобладающую ориентацию
    if total_dx > total_dy * 1.5:
        return 'horizontal'
    elif total_dy > total_dx * 1.5:
        return 'vertical'
    else:
        return 'unknown'

def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Вычисляет расстояние от точки до отрезка"""
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def is_point_in_polygon(px: float, py: float, vertices: List[Dict]) -> bool:
    """
    Определяет, находится ли точка внутри полигона с помощью метода Ray Casting
    
    Args:
        px, py: Координаты точки
        vertices: Список вершин полигона
    
    Returns:
        True, если точка находится внутри полигона, иначе False
    """
    n = len(vertices)
    inside = False
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vertices[i]['x'], vertices[i]['y']
        xj, yj = vertices[j]['x'], vertices[j]['y']
        
        # Проверяем, пересекает ли луч от точки вправо ребро (i,j)
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
    
    return inside

def find_walls_connected_to_junction(junction: Dict, walls: List[Dict], wall_polygons: List[Dict], tolerance: float = 25.0) -> List[Dict]:
    """
    Находит все стены, подключенные к данному junction
    
    Args:
        junction: Точка junction
        walls: Список простых стен (отрезков)
        wall_polygons: Список стеновых полигонов
        tolerance: Максимальное расстояние для определения подключения
    
    Returns:
        Список подключенных стен
    """
    connected_walls = []
    jx, jy = junction['x'], junction['y']
    
    # Проверяем простые стены (отрезки)
    for wall in walls:
        start = wall.get('start', {})
        end = wall.get('end', {})
        
        if not start or not end:
            continue
        
        # Проверяем расстояние до отрезка стены
        distance = point_to_segment_distance(jx, jy, start['x'], start['y'], end['x'], end['y'])
        
        if distance <= tolerance:
            connected_walls.append(wall)
    
    # Проверяем стеновые полигоны
    for wall in wall_polygons:
        vertices = wall.get('vertices', [])
        if not vertices:
            continue
        
        # Сначала проверяем, находится ли junction внутри полигона
        if is_point_in_polygon(jx, jy, vertices):
            connected_walls.append(wall)
            continue
        
        # Если не внутри, проверяем, находится ли junction близко к какому-либо ребру полигона
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            distance = point_to_segment_distance(jx, jy, v1['x'], v1['y'], v2['x'], v2['y'])
            
            if distance <= tolerance:
                connected_walls.append(wall)
                break
    
    return connected_walls

def analyze_polygon_directions_with_thickness(junction: Dict, polygon_vertices: List[Dict], wall_thickness: float = 20.0) -> Tuple[List[str], Dict[str, float]]:
    """
    Анализирует направления расширения полигона с учетом толщины стены
    
    Args:
        junction: Точка junction с координатами x, y
        polygon_vertices: Вершины полигона
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Кортеж (список значимых направлений, словарь расстояний)
    """
    jx, jy = junction['x'], junction['y']
    
    # Находим точки пересечения с ребрами полигона
    intersections = {'left': None, 'right': None, 'up': None, 'down': None}
    
    n = len(polygon_vertices)
    for i in range(n):
        v1 = polygon_vertices[i]
        v2 = polygon_vertices[(i + 1) % n]
        
        x1, y1 = v1['x'], v1['y']
        x2, y2 = v2['x'], v2['y']
        
        # Пересечение с горизонтальной линией
        if (y1 <= jy <= y2) or (y2 <= jy <= y1):
            if y1 != y2:
                t = (jy - y1) / (y2 - y1)
                x_intersect = x1 + t * (x2 - x1)
                
                if x_intersect < jx and (intersections['left'] is None or x_intersect > intersections['left']):
                    intersections['left'] = x_intersect
                elif x_intersect > jx and (intersections['right'] is None or x_intersect < intersections['right']):
                    intersections['right'] = x_intersect
        
        # Пересечение с вертикальной линией
        if (x1 <= jx <= x2) or (x2 <= jx <= x1):
            if x1 != x2:
                t = (jx - x1) / (x2 - x1)
                y_intersect = y1 + t * (y2 - y1)
                
                if y_intersect > jy and (intersections['up'] is None or y_intersect < intersections['up']):
                    intersections['up'] = y_intersect
                elif y_intersect < jy and (intersections['down'] is None or y_intersect > intersections['down']):
                    intersections['down'] = y_intersect
    
    # Определяем значимые расширения
    significant_directions = []
    distances = {}
    
    for direction, intersection in intersections.items():
        if intersection is not None:
            if direction == 'left':
                distance = jx - intersection
            elif direction == 'right':
                distance = intersection - jx
            elif direction == 'up':
                distance = intersection - jy
            elif direction == 'down':
                distance = jy - intersection
            
            distances[direction] = distance
            if distance > wall_thickness:
                significant_directions.append(direction)
    
    return significant_directions, distances

def determine_junction_type_from_directions(directions: List[str]) -> str:
    """
    Определяет тип junction на основе значимых направлений расширения
    
    Args:
        directions: Список значимых направлений
    
    Returns:
        'L', 'T', 'X', 'straight', или 'unknown'
    """
    count = len(directions)
    
    if count == 0:
        return 'unknown'
    elif count == 1:
        return 'unknown'
    elif count == 2:
        if ('left' in directions and 'right' in directions) or ('up' in directions and 'down' in directions):
            return 'straight'
        else:
            return 'L'
    elif count == 3:
        return 'T'
    elif count == 4:
        return 'X'
    else:
        return 'unknown'

def determine_junction_type(connected_walls: List[Dict], wall_orientations: List[str]) -> str:
    """
    Определяет тип junction на основе подключенных стен
    
    Args:
        connected_walls: Список подключенных стен
        wall_orientations: Список ориентаций стен
    
    Returns:
        'L', 'T', 'X', или 'unknown'
    """
    connection_count = len(connected_walls)
    
    if connection_count < 2:
        return 'unknown'
    
    # Считаем количество горизонтальных и вертикальных стен
    horizontal_count = wall_orientations.count('horizontal')
    vertical_count = wall_orientations.count('vertical')
    
    # Определяем тип junction
    if connection_count == 2:
        # L-junction: 2 стены, обычно разной ориентации
        if horizontal_count == 1 and vertical_count == 1:
            return 'L'
        elif horizontal_count == 2 or vertical_count == 2:
            # Это может быть продолжение прямой стены
            return 'straight'
        else:
            return 'unknown'
    
    elif connection_count == 3:
        # T-junction: 3 стены, 2 одной ориентации, 1 другой
        if (horizontal_count == 2 and vertical_count == 1) or \
           (horizontal_count == 1 and vertical_count == 2):
            return 'T'
        else:
            return 'unknown'
    
    elif connection_count == 4:
        # X-junction: 4 стены, по 2 каждой ориентации
        if horizontal_count == 2 and vertical_count == 2:
            return 'X'
        else:
            return 'unknown'
    
    else:
        return 'unknown'

def determine_junction_type_with_thickness(junction: Dict, wall_polygons: List[Dict], wall_thickness: float = 20.0) -> Tuple[str, List[Dict]]:
    """
    Определяет тип junction с учетом толщины стены для полигонов
    
    Args:
        junction: Точка junction с координатами x, y
        wall_polygons: Список полигонов стен
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Кортеж (тип junction, список полигонов, содержащих junction)
    """
    containing_polygons = []
    
    # Находим все полигоны, содержащие junction
    for wall in wall_polygons:
        vertices = wall.get('vertices', [])
        if vertices and is_point_in_polygon(junction['x'], junction['y'], vertices):
            containing_polygons.append(wall)
    
    if not containing_polygons:
        return 'unknown', []
    
    # Если junction находится внутри нескольких полигонов, используем самый большой
    if len(containing_polygons) > 1:
        containing_polygons.sort(key=lambda w: w.get('area', 0), reverse=True)
    
    main_polygon = containing_polygons[0]
    vertices = main_polygon['vertices']
    
    # Анализируем направления с учетом толщины
    directions, distances = analyze_polygon_directions_with_thickness(junction, vertices, wall_thickness)
    
    # Определяем тип junction на основе направлений
    junction_type = determine_junction_type_from_directions(directions)
    
    return junction_type, containing_polygons

def analyze_all_junctions(data: Dict, tolerance: float = 15.0, use_thickness_logic: bool = True, wall_thickness: float = 20.0) -> List[JunctionWithConnections]:
    """
    Анализирует все junctions и определяет их типы
    
    Args:
        data: Данные плана с junctions, walls и wall_polygons
        tolerance: Допуск расстояния для определения подключения
        use_thickness_logic: Использовать ли логику с учетом толщины стены
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Список junctions с определенными типами
    """
    junctions = data.get('junctions', [])
    walls = data.get('walls', [])
    wall_polygons = data.get('wall_polygons', [])
    
    analyzed_junctions = []
    
    print(f"Анализ {len(junctions)} junctions...")
    print(f"  Доступно стен: {len(walls)} простых, {len(wall_polygons)} полигонов")
    if use_thickness_logic:
        print(f"  Используется логика с толщиной стены: {wall_thickness} пикселей")
    
    for idx, junction in enumerate(junctions):
        if use_thickness_logic:
            # Сначала пробуем определить тип с учетом толщины стены
            junction_type, containing_polygons = determine_junction_type_with_thickness(junction, wall_polygons, wall_thickness)
            
            # Если junction найден внутри полигона, используем этот результат
            if containing_polygons:
                connected_walls = containing_polygons
                wall_orientations = ['polygon'] * len(connected_walls)
            else:
                # Иначе используем старую логику
                connected_walls = find_walls_connected_to_junction(junction, walls, wall_polygons, tolerance)
                wall_orientations = []
                for wall in connected_walls:
                    orientation = analyze_wall_orientation(wall)
                    wall_orientations.append(orientation)
                junction_type = determine_junction_type(connected_walls, wall_orientations)
        else:
            # Используем старую логику
            connected_walls = find_walls_connected_to_junction(junction, walls, wall_polygons, tolerance)
            wall_orientations = []
            for wall in connected_walls:
                orientation = analyze_wall_orientation(wall)
                wall_orientations.append(orientation)
            junction_type = determine_junction_type(connected_walls, wall_orientations)
        
        # Создаем объект с результатами анализа
        analyzed_junction = JunctionWithConnections(
            junction=junction,
            junction_type=junction_type,
            connected_walls=connected_walls,
            wall_orientations=wall_orientations,
            connection_count=len(connected_walls)
        )
        
        analyzed_junctions.append(analyzed_junction)
        
        print(f"  Junction {idx+1}: тип={junction_type}, подключено стен={len(connected_walls)}")
    
    return analyzed_junctions

def print_junction_statistics(analyzed_junctions: List[JunctionWithConnections]) -> None:
    """Выводит статистику по типам junctions"""
    type_counts = {}
    
    for junction in analyzed_junctions:
        jtype = junction.junction_type
        type_counts[jtype] = type_counts.get(jtype, 0) + 1
    
    print(f"\nСтатистика junctions:")
    for jtype, count in type_counts.items():
        print(f"  {jtype}-junctions: {count}")
    
    print(f"\nВсего junctions: {len(analyzed_junctions)}")

def update_junctions_in_data(data: Dict, analyzed_junctions: List[JunctionWithConnections]) -> Dict:
    """
    Обновляет данные, добавляя информацию о типах junctions
    
    Args:
        data: Исходные данные
        analyzed_junctions: Результаты анализа
    
    Returns:
        Обновленные данные с типами junctions
    """
    updated_data = data.copy()
    
    # Создаем словарь для быстрого доступа к junction по координатам
    junction_map = {}
    for junction in updated_data.get('junctions', []):
        key = f"{junction['x']:.1f}_{junction['y']:.1f}"
        junction_map[key] = junction
    
    # Обновляем типы junctions
    for analyzed_junction in analyzed_junctions:
        junction = analyzed_junction.junction
        key = f"{junction['x']:.1f}_{junction['y']:.1f}"
        
        if key in junction_map:
            junction_map[key]['detected_type'] = analyzed_junction.junction_type
            junction_map[key]['connection_count'] = analyzed_junction.connection_count
            junction_map[key]['wall_orientations'] = analyzed_junction.wall_orientations
    
    return updated_data

def main():
    """Основная функция для анализа junctions"""
    print("="*60)
    print("АНАЛИЗ ТИПОВ JUNCTIONS (L, T, X)")
    print("="*60)
    
    # Загружаем данные
    data = load_data('plan_floor1_objects.json')
    
    # Анализируем junctions
    analyzed_junctions = analyze_all_junctions(data)
    
    # Выводим статистику
    print_junction_statistics(analyzed_junctions)
    
    # Обновляем данные
    updated_data = update_junctions_in_data(data, analyzed_junctions)
    
    # Сохраняем результаты
    output_path = 'plan_floor1_objects_with_junction_types.json'
    with open(output_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"\nРезультаты сохранены в: {output_path}")
    
    return analyzed_junctions

if __name__ == '__main__':
    main()