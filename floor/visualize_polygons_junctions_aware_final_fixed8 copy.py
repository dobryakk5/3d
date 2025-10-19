#!/usr/bin/env python3
"""
Улучшенная версия visualize_polygons_two_colors_bbox.py с учетом junctions
Создает SVG файл с ограничивающими прямоугольниками:
- Красные bbox для стен (с учетом junctions)
- Коричневые bbox для колонн
- Синие bbox для окон
- Зеленые bbox для дверей
- Серые полигоны для всех стеновых полигонов из JSON с нумерацией
- Синие кружки для junctions с нумерацией

КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
1. Полигоны стен идут от проема до junction
2. Изогнутые полигоны разбиваются на сегменты от junction к junction
3. Середина края проема совпадает с junction
4. Улучшенная обработка T-соединений
5. Исправлена проблема с определением ориентации сегментов (w60)
6. Ограничение толщины полигонов минимальной толщиной двери
7. Исправлена проблема с полигоном w51
8. Обеспечено, что от каждой стороны проема отходить только один полигон
9. Обеспечено, что от каждого junction отходить только один полигон
10. Правильное разбиение Г-образных поворотов с использованием junctions
11. Правильная обработка T-соединений с тремя направлениями
12. Добавлены метки ориентации (h/v) для стеновых сегментов и проемов
13. ИСПРАВЛЕНО: Учет ориентации проемов при определении соседства со стенами
14. ИСПРАВЛЕНО: Учет ориентации ближайшего проема при создании сегментов от junctions
15. ИСПРАВЛЕНО: Учет ориентации ближайшего проема при создании сегментов из полигонов
"""

import json
import svgwrite
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
import math
from dataclasses import dataclass
from collections import defaultdict

# =============================================================================
# СТРУКТУРЫ ДАННЫХ
# =============================================================================

@dataclass
class JunctionPoint:
    x: float
    y: float
    junction_type: str
    id: int
    confidence: float = 1.0

@dataclass
class OpeningEdge:
    x1: float
    y1: float
    x2: float
    y2: float
    opening_id: str
    opening_type: str
    side: str  # 'left', 'right', 'top', 'bottom'
    midpoint: Tuple[float, float] = None
    orientation: str = 'unknown'  # 'horizontal' или 'vertical'

@dataclass
class WallSegment:
    vertices: List[Dict[str, float]]
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    start_type: str  # 'junction', 'opening', 'vertex'
    end_type: str    # 'junction', 'opening', 'vertex'
    polygon_id: str
    segment_id: str
    opening_id: Optional[str] = None
    side: Optional[str] = None

# =============================================================================
# ФУНКЦИИ ЗАГРУЗКИ И ПОДГОТОВКИ ДАННЫХ
# =============================================================================

def load_objects_data(json_path: str) -> Dict:
    """Загружает данные объектов из JSON файла"""
    print(f"Загрузка данных из: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"  ✓ Данные успешно загружены")
        return data
    except Exception as e:
        print(f"  ✗ Ошибка загрузки данных: {e}")
        return {}

def parse_junctions(data: Dict) -> List[JunctionPoint]:
    """Парсит junctions из JSON в структурированный формат"""
    junctions_data = data.get('junctions', [])
    junctions = []
    
    for idx, junction_data in enumerate(junctions_data):
        junction = JunctionPoint(
            x=junction_data.get('x', 0),
            y=junction_data.get('y', 0),
            junction_type=junction_data.get('type', 'unknown'),
            id=idx + 1
        )
        junctions.append(junction)
    
    print(f"  ✓ Загружено {len(junctions)} junctions")
    return junctions

def parse_openings(data: Dict) -> Tuple[List[Dict], List[OpeningEdge]]:
    """
    Парсит проемы и их края из JSON
    
    Returns:
        Tuple: (список проемов, список краев проемов)
    """
    openings = data.get('openings', [])
    opening_edges = []
    
    for opening in openings:
        opening_id = opening.get('id', '')
        opening_type = opening.get('type', '')
        bbox = opening.get('bbox', {})
        
        if not bbox:
            continue
        
        # Определяем ориентацию проема
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        orientation = 'horizontal' if width > height else 'vertical'
        
        x, y = bbox['x'], bbox['y']
        
        if orientation == 'horizontal':
            # Левый край (вертикальная линия)
            left_edge = OpeningEdge(
                x1=x, y1=y,
                x2=x, y2=y + height,
                opening_id=opening_id,
                opening_type=opening_type,
                side='left',
                midpoint=(x, y + height / 2),
                orientation='vertical'  # Край вертикальный
            )
            
            # Правый край (вертикальная линия)
            right_edge = OpeningEdge(
                x1=x + width, y1=y,
                x2=x + width, y2=y + height,
                opening_id=opening_id,
                opening_type=opening_type,
                side='right',
                midpoint=(x + width, y + height / 2),
                orientation='vertical'  # Край вертикальный
            )
            
            opening_edges.extend([left_edge, right_edge])
        else:  # vertical
            # Верхний край (горизонтальная линия)
            top_edge = OpeningEdge(
                x1=x, y1=y,
                x2=x + width, y2=y,
                opening_id=opening_id,
                opening_type=opening_type,
                side='top',
                midpoint=(x + width / 2, y),
                orientation='horizontal'  # Край горизонтальный
            )
            
            # Нижний край (горизонтальная линия)
            bottom_edge = OpeningEdge(
                x1=x, y1=y + height,
                x2=x + width, y2=y + height,
                opening_id=opening_id,
                opening_type=opening_type,
                side='bottom',
                midpoint=(x + width / 2, y + height),
                orientation='horizontal'  # Край горизонтальный
            )
            
            opening_edges.extend([top_edge, bottom_edge])
    
    print(f"  ✓ Обработано {len(openings)} проемов, {len(opening_edges)} краев")
    return openings, opening_edges

def get_wall_thickness_from_doors(data: Dict) -> float:
    """Определяет толщину стен на основе минимальной толщины дверей"""
    openings = data.get('openings', [])
    door_thicknesses = []
    
    for opening in openings:
        if opening.get('type') == 'door':
            bbox = opening.get('bbox', {})
            # Для дверей берем меньшую сторону как толщину
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            thickness = min(width, height)
            if thickness > 0:
                door_thicknesses.append(thickness)
    
    if door_thicknesses:
        min_thickness = min(door_thicknesses)  # Используем минимальную толщину
        print(f"  ✓ Толщина стен определена по минимальной толщине дверей: {min_thickness:.1f} px")
        return min_thickness
    else:
        # Если дверей нет, используем стандартную толщину
        default_thickness = 20.0
        print(f"  ✓ Двери не найдены, используем стандартную толщину: {default_thickness} px")
        return default_thickness

# =============================================================================
# ФУНКЦИИ РАБОТЫ С JUNCTIONS
# =============================================================================

def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Вычисляет расстояние от точки до отрезка"""
    # Вектор отрезка
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Отрезок вырожден в точку
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    # Параметр проекции точки на прямую
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

    # Ближайшая точка на отрезке
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def find_closest_point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Находит ближайшую точку на отрезке к заданной точке"""
    # Вектор отрезка
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Отрезок вырожден в точку
        return (x1, y1)

    # Параметр проекции точки на прямую
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

    # Ближайшая точка на отрезке
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return (closest_x, closest_y)

def find_junctions_on_polygon(polygon_vertices: List[Dict], 
                             junctions: List[JunctionPoint], 
                             threshold: float = 15.0) -> List[Tuple[int, JunctionPoint]]:
    """
    Находит все junctions, которые лежат на границе полигона
    
    Args:
        polygon_vertices: Список вершин полигона
        junctions: Список всех junctions
        threshold: Максимальное расстояние для определения принадлежности
    
    Returns:
        Список tuple (vertex_index, junction) для каждого найденного junction
    """
    junctions_on_polygon = []
    
    for junction in junctions:
        jx, jy = junction.x, junction.y
        
        # Проверяем каждую вершину полигона
        for i, vertex in enumerate(polygon_vertices):
            vx, vy = vertex['x'], vertex['y']
            distance = math.sqrt((jx - vx)**2 + (jy - vy)**2)
            
            if distance <= threshold:
                junctions_on_polygon.append((i, junction))
                break
        
        # Если не найдено на вершинах, проверяем рёбра
        if not any(j == junction for _, j in junctions_on_polygon):
            for i in range(len(polygon_vertices)):
                v1 = polygon_vertices[i]
                v2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
                
                # Проверяем расстояние до ребра
                dist_to_edge = point_to_segment_distance(jx, jy, v1['x'], v1['y'], v2['x'], v2['y'])
                
                if dist_to_edge <= threshold:
                    # Находим точную позицию на ребре
                    closest_point = find_closest_point_on_segment(jx, jy, v1['x'], v1['y'], v2['x'], v2['y'])
                    
                    # Создаем временный junction с точной позицией
                    precise_junction = JunctionPoint(
                        x=closest_point[0],
                        y=closest_point[1],
                        junction_type=junction.junction_type,
                        id=junction.id,
                        confidence=junction.confidence
                    )
                    
                    junctions_on_polygon.append((i, precise_junction))
                    break
    
    return junctions_on_polygon

def insert_junctions_in_polygon(polygon_vertices: List[Dict], 
                               junctions_on_polygon: List[Tuple[int, JunctionPoint]]) -> List[Dict]:
    """
    Вставляет junctions в последовательность вершин полигона
    
    Args:
        polygon_vertices: Исходные вершины полигона
        junctions_on_polygon: Список tuple (vertex_index, junction)
    
    Returns:
        Обновленный список вершин с junctions
    """
    if not junctions_on_polygon:
        return polygon_vertices
    
    # Сортируем junctions по индексу вершины
    junctions_on_polygon.sort(key=lambda x: x[0])
    
    new_vertices = []
    current_junction_idx = 0
    
    for i, vertex in enumerate(polygon_vertices):
        # Добавляем исходную вершину
        new_vertices.append(vertex)
        
        # Добавляем все junctions, которые должны идти после этой вершины
        while (current_junction_idx < len(junctions_on_polygon) and 
               junctions_on_polygon[current_junction_idx][0] == i):
            
            junction = junctions_on_polygon[current_junction_idx][1]
            
            # Проверяем, не совпадает ли junction с текущей вершиной
            vx, vy = vertex['x'], vertex['y']
            jx, jy = junction.x, junction.y
            distance = math.sqrt((jx - vx)**2 + (jy - vy)**2)
            
            if distance > 1.0:  # Если не совпадает, добавляем как отдельную вершину
                junction_vertex = {
                    'x': junction.x,
                    'y': junction.y,
                    'type': 'junction',
                    'junction_id': junction.id,
                    'junction_type': junction.junction_type
                }
                new_vertices.append(junction_vertex)
            
            current_junction_idx += 1
    
    return new_vertices

# =============================================================================
# ФУНКЦИИ РАБОТЫ С ПРОЕМАМИ
# =============================================================================

def get_edge_orientation(x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Определяет ориентацию ребра: 'horizontal', 'vertical', 'diagonal'
    
    Args:
        x1, y1: Координаты начала ребра
        x2, y2: Координаты конца ребра
    
    Returns:
        Ориентация ребра
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Порог для определения горизонтальности/вертикальности
    tolerance = 5.0
    
    if dx > dy and dy <= tolerance:
        return 'horizontal'
    elif dy > dx and dx <= tolerance:
        return 'vertical'
    else:
        return 'diagonal'

def find_opening_edges_on_polygon(polygon_vertices: List[Dict], 
                                 opening_edges: List[OpeningEdge], 
                                 threshold: float = 15.0) -> List[Tuple[int, OpeningEdge, Tuple[float, float]]]:
    """
    Находит все края проемов, которые примыкают к полигону с учетом ориентации
    
    Returns:
        Список tuple (edge_index, opening_edge, midpoint_on_polygon)
    """
    edges_on_polygon = []
    
    for edge in opening_edges:
        # Проверяем середину края проема
        mx, my = edge.midpoint
        
        # Ищем ближайшее ребро полигона с совместимой ориентацией
        min_distance = float('inf')
        closest_edge_idx = -1
        closest_point = None
        
        for i in range(len(polygon_vertices)):
            v1 = polygon_vertices[i]
            v2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
            
            # Определяем ориентацию ребра полигона
            edge_orientation = get_edge_orientation(v1['x'], v1['y'], v2['x'], v2['y'])
            
            # Проверяем совместимость ориентаций
            # Вертикальный край проема должен примыкать к вертикальному ребру полигона
            # Горизонтальный край проема должен примыкать к горизонтальному ребру полигона
            if edge.orientation != edge_orientation:
                continue
            
            dist_to_edge = point_to_segment_distance(mx, my, v1['x'], v1['y'], v2['x'], v2['y'])
            
            if dist_to_edge < min_distance:
                min_distance = dist_to_edge
                closest_edge_idx = i
                closest_point = find_closest_point_on_segment(mx, my, v1['x'], v1['y'], v2['x'], v2['y'])
        
        if min_distance <= threshold:
            edges_on_polygon.append((closest_edge_idx, edge, closest_point))
    
    return edges_on_polygon

def insert_opening_midpoints_in_polygon(polygon_vertices: List[Dict], 
                                      opening_edges_on_polygon: List[Tuple[int, OpeningEdge, Tuple[float, float]]]) -> List[Dict]:
    """
    Вставляет середины краев проемов в последовательность вершин полигона
    """
    if not opening_edges_on_polygon:
        return polygon_vertices
    
    # Сортируем края проемов по индексу ребра
    opening_edges_on_polygon.sort(key=lambda x: x[0])
    
    new_vertices = []
    current_edge_idx = 0
    
    for i, vertex in enumerate(polygon_vertices):
        # Добавляем исходную вершину
        new_vertices.append(vertex)
        
        # Добавляем все середины краев проемов, которые должны идти после этой вершины
        while (current_edge_idx < len(opening_edges_on_polygon) and 
               opening_edges_on_polygon[current_edge_idx][0] == i):
            
            edge, midpoint = opening_edges_on_polygon[current_edge_idx][1], opening_edges_on_polygon[current_edge_idx][2]
            
            # Проверяем, не совпадает ли середина с текущей вершиной
            vx, vy = vertex['x'], vertex['y']
            mx, my = midpoint
            distance = math.sqrt((mx - vx)**2 + (my - vy)**2)
            
            if distance > 1.0:  # Если не совпадает, добавляем как отдельную вершину
                opening_vertex = {
                    'x': mx,
                    'y': my,
                    'type': 'opening',
                    'opening_id': edge.opening_id,
                    'opening_type': edge.opening_type,
                    'side': edge.side
                }
                new_vertices.append(opening_vertex)
            
            current_edge_idx += 1
    
    return new_vertices

# =============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ УЧЕТА ОРИЕНТАЦИИ JUNCTIONS И ПОЛИГОНОВ
# =============================================================================

def find_nearest_opening_to_junction(junction: JunctionPoint, 
                                   openings: List[Dict], 
                                   threshold: float = 30.0) -> Optional[Dict]:
    """
    Находит ближайший проем к junction с улучшенным алгоритмом
    
    Args:
        junction: Точка junction
        openings: Список всех проемов
        threshold: Максимальное расстояние для определения соседства
    
    Returns:
        Ближайший проем или None, если проем не найден
    """
    nearest_opening = None
    min_distance = float('inf')
    
    for opening in openings:
        bbox = opening.get('bbox', {})
        if not bbox:
            continue
        
        # ИСПРАВЛЕНО: Проверяем не только центр проема, но и края проема
        # Это более точно определяет соседство junction с проемом
        
        # 1. Проверяем расстояние до центра проема
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        center_distance = math.sqrt((junction.x - center_x)**2 + (junction.y - center_y)**2)
        
        # 2. Проверяем расстояние до всех четырех углов проема
        corners = [
            (bbox['x'], bbox['y']),  # левый верхний
            (bbox['x'] + bbox['width'], bbox['y']),  # правый верхний
            (bbox['x'], bbox['y'] + bbox['height']),  # левый нижний
            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])  # правый нижний
        ]
        
        min_corner_distance = min(
            math.sqrt((junction.x - corner[0])**2 + (junction.y - corner[1])**2)
            for corner in corners
        )
        
        # 3. Проверяем расстояние до середин краев проема
        edge_midpoints = [
            (bbox['x'] + bbox['width'] / 2, bbox['y']),  # середина верхнего края
            (bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height']),  # середина нижнего края
            (bbox['x'], bbox['y'] + bbox['height'] / 2),  # середина левого края
            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height'] / 2)  # середина правого края
        ]
        
        min_edge_distance = min(
            math.sqrt((junction.x - midpoint[0])**2 + (junction.y - midpoint[1])**2)
            for midpoint in edge_midpoints
        )
        
        # 4. Используем минимальное расстояние из всех проверок
        actual_distance = min(center_distance, min_corner_distance, min_edge_distance)
        
        if actual_distance < min_distance and actual_distance <= threshold:
            min_distance = actual_distance
            nearest_opening = opening
    
    return nearest_opening

def find_nearest_opening_to_vertex(vertex: Dict, 
                                 openings: List[Dict], 
                                 threshold: float = 30.0) -> Optional[Dict]:
    """
    Находит ближайший проем к вершине
    
    Args:
        vertex: Вершина полигона
        openings: Список всех проемов
        threshold: Максимальное расстояние для определения соседства
    
    Returns:
        Ближайший проем или None, если проем не найден
    """
    nearest_opening = None
    min_distance = float('inf')
    
    for opening in openings:
        bbox = opening.get('bbox', {})
        if not bbox:
            continue
        
        # Вычисляем центр проема
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        
        # Вычисляем расстояние от вершины до центра проема
        distance = math.sqrt((vertex['x'] - center_x)**2 + (vertex['y'] - center_y)**2)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            nearest_opening = opening
    
    return nearest_opening

def get_opening_orientation(opening: Dict) -> str:
    """
    Определяет ориентацию проема
    
    Args:
        opening: Проем с bbox
    
    Returns:
        'horizontal' или 'vertical'
    """
    bbox = opening.get('bbox', {})
    width = bbox.get('width', 0)
    height = bbox.get('height', 0)
    
    return 'horizontal' if width > height else 'vertical'

# =============================================================================
# ФУНКЦИИ РАЗБИЕНИЯ ПОЛИГОНОВ
# =============================================================================

def identify_vertex_type(vertex: Dict) -> str:
    """Определяет тип вершины: 'junction', 'opening', 'vertex'"""
    if 'type' in vertex:
        if vertex['type'] == 'junction':
            return 'junction'
        elif vertex['type'] == 'opening':
            return 'opening'
    
    return 'vertex'

def get_used_opening_sides(segments: List[WallSegment]) -> Dict[str, set]:
    """Создает словарь отслеживания использованных сторон проемов"""
    used_sides = {}
    for segment in segments:
        opening_id = segment.opening_id
        side = segment.side
        if opening_id and side:
            if opening_id not in used_sides:
                used_sides[opening_id] = set()
            used_sides[opening_id].add(side)
    return used_sides

def get_used_junctions(segments: List[WallSegment]) -> set:
    """Создает множество уже использованных junctions"""
    used_junctions = set()
    for segment in segments:
        if segment.start_type == 'junction':
            used_junctions.add(segment.start_point)
        if segment.end_type == 'junction':
            used_junctions.add(segment.end_point)
    return used_junctions

def detect_corner_junctions(polygon_vertices: List[Dict], 
                           junctions: List[JunctionPoint], 
                           wall_thickness: float) -> List[Tuple[int, JunctionPoint]]:
    """
    Обнаруживает junctions в углах полигона (Г-образные повороты)
    
    Args:
        polygon_vertices: Вершины полигона
        junctions: Список всех junctions
        wall_thickness: Толщина стены
    
    Returns:
        Список tuple (vertex_index, junction) для junctions в углах
    """
    corner_junctions = []
    
    for junction in junctions:
        jx, jy = junction.x, junction.y
        
        # Проверяем, находится ли junction внутри полигона
        # Для этого проверяем пересечение с ребрами полигона в радиусе wall_thickness
        for i in range(len(polygon_vertices)):
            v1 = polygon_vertices[i]
            v2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
            
            # Проверяем расстояние до ребра
            dist_to_edge = point_to_segment_distance(jx, jy, v1['x'], v1['y'], v2['x'], v2['y'])
            
            if dist_to_edge <= wall_thickness:
                # Junction находится близко к ребру, это может быть угол
                # Проверяем, находится ли junction между двумя ребрами с разными направлениями
                v_prev = polygon_vertices[(i - 1) % len(polygon_vertices)]
                v_next = polygon_vertices[(i + 2) % len(polygon_vertices)]
                
                # Вычисляем направления рёбер
                dir1_x, dir1_y = v2['x'] - v1['x'], v2['y'] - v1['y']
                dir2_x, dir2_y = v_next['x'] - v2['x'], v_next['y'] - v2['y']
                
                # Нормализуем направления
                len1 = math.sqrt(dir1_x**2 + dir1_y**2)
                len2 = math.sqrt(dir2_x**2 + dir2_y**2)
                
                if len1 > 0 and len2 > 0:
                    dir1_x, dir1_y = dir1_x/len1, dir1_y/len1
                    dir2_x, dir2_y = dir2_x/len2, dir2_y/len2
                    
                    # Вычисляем скалярное произведение
                    dot_product = dir1_x * dir2_x + dir1_y * dir2_y
                    
                    # Если скалярное произведение близко к 0, то угол близок к 90°
                    if abs(dot_product) < 0.3:  # Порог для определения прямого угла
                        corner_junctions.append((i, junction))
                        break
    
    return corner_junctions

def split_polygon_at_corner_junctions(polygon_vertices: List[Dict], 
                                    corner_junctions: List[Tuple[int, JunctionPoint]]) -> List[Dict]:
    """
    Разбивает полигон в точках угловых junctions
    
    Args:
        polygon_vertices: Вершины полигона
        corner_junctions: Список tuple (vertex_index, junction) для угловых junctions
    
    Returns:
        Обновленный список вершин с разбитием в углах
    """
    if not corner_junctions:
        return polygon_vertices
    
    # Сортируем junctions по индексу вершины
    corner_junctions.sort(key=lambda x: x[0])
    
    new_vertices = []
    
    for i, vertex in enumerate(polygon_vertices):
        # Добавляем исходную вершину
        new_vertices.append(vertex)
        
        # Проверяем, есть ли угловой junction в этой позиции
        for vertex_idx, junction in corner_junctions:
            if vertex_idx == i:
                # Добавляем junction как отдельную вершину
                junction_vertex = {
                    'x': junction.x,
                    'y': junction.y,
                    'type': 'junction',
                    'junction_id': junction.id,
                    'junction_type': junction.junction_type
                }
                new_vertices.append(junction_vertex)
                break
    
    return new_vertices

def split_polygon_into_segments(polygon_vertices: List[Dict], 
                              used_opening_sides: Dict[str, set] = None,
                              used_junctions: set = None,
                              junctions: List[JunctionPoint] = None,
                              openings: List[Dict] = None) -> Tuple[List[WallSegment], Dict[str, set], set]:
    """
    Разбивает полигон на сегменты между junctions и краями проемов
    
    Args:
        polygon_vertices: Вершины полигона с junctions и серединами проемов
        used_opening_sides: Словарь уже использованных сторон проемов
        used_junctions: Множество уже использованных junctions
        junctions: Список всех junctions
        openings: Список всех проемов
    
    Returns:
        Кортеж: (список сегментов стен, обновленный словарь использованных сторон, обновленное множество использованных junctions)
    """
    if len(polygon_vertices) < 2:
        return [], {}, set()
    
    if used_opening_sides is None:
        used_opening_sides = {}
    
    if used_junctions is None:
        used_junctions = set()
    
    segments = []
    segment_vertices = []
    
    # Идем по всем вершинам полигона
    for i, vertex in enumerate(polygon_vertices):
        vertex_type = identify_vertex_type(vertex)
        vertex_point = (vertex['x'], vertex['y'])
        
        # Добавляем вершину к текущему сегменту
        segment_vertices.append(vertex)
        
        # Проверяем, нужно ли завершить текущий сегмент
        if vertex_type in ['junction', 'opening', 'vertex'] and i > 0:
            # НОВОЕ: Находим ближайший проем к текущей вершине
            nearest_opening = None
            if openings:
                nearest_opening = find_nearest_opening_to_vertex(vertex, openings)
            
            # Для проемов проверяем, не использована ли уже эта сторона
            if vertex_type == 'opening':
                opening_id = vertex.get('opening_id')
                side = vertex.get('side')
                
                if opening_id and side:
                    if opening_id in used_opening_sides and side in used_opening_sides[opening_id]:
                        # Эта сторона проема уже использована, пропускаем
                        segment_vertices = [vertex]
                        continue
                    
                    # Помечаем сторону как использованную
                    if opening_id not in used_opening_sides:
                        used_opening_sides[opening_id] = set()
                    used_opening_sides[opening_id].add(side)
            
            # Для junctions проверяем, не использован ли уже этот junction
            if vertex_type == 'junction':
                # ИСПРАВЛЕНО: Проверяем использованность junction с учетом проема
                if junctions and openings:
                    # Находим соответствующий junction
                    junction_obj = None
                    for j in junctions:
                        if abs(j.x - vertex['x']) < 1.0 and abs(j.y - vertex['y']) < 1.0:
                            junction_obj = j
                            break
                    
                    if junction_obj:
                        # Находим ближайший проем к junction
                        nearest_opening = find_nearest_opening_to_junction(junction_obj, openings)
                        
                        # Создаем ключ для проверки использованности: (junction_point, opening_id)
                        opening_id = nearest_opening.get('id') if nearest_opening else None
                        junction_opening_key = (vertex_point, opening_id)
                        
                        # Проверяем, использована ли уже эта пара (junction, проем)
                        if junction_opening_key in used_junctions:
                            # Эта пара уже использована, пропускаем
                            segment_vertices = [vertex]
                            continue
                        
                        # Помечаем пару (junction, проем) как использованную
                        used_junctions.add(junction_opening_key)
                    else:
                        # Если не найден соответствующий junction, используем старую логику
                        if vertex_point in used_junctions:
                            segment_vertices = [vertex]
                            continue
                        used_junctions.add(vertex_point)
                else:
                    # Если нет junctions или openings, используем старую логику
                    if vertex_point in used_junctions:
                        segment_vertices = [vertex]
                        continue
                    used_junctions.add(vertex_point)
                
                # НОВОЕ: Находим ближайший проем к junction (уже сделано выше)
            
            # Завершаем текущий сегмент
            if len(segment_vertices) >= 2:
                start_vertex = segment_vertices[0]
                end_vertex = vertex
                
                start_type = identify_vertex_type(start_vertex)
                end_type = vertex_type
                
                segment = WallSegment(
                    vertices=segment_vertices.copy(),
                    start_point=(start_vertex['x'], start_vertex['y']),
                    end_point=(end_vertex['x'], end_vertex['y']),
                    start_type=start_type,
                    end_type=end_type,
                    polygon_id="unknown",  # Будет установлено извне
                    segment_id=f"segment_{len(segments) + 1}",
                    opening_id=vertex.get('opening_id') if vertex_type == 'opening' else None,
                    side=vertex.get('side') if vertex_type == 'opening' else None
                )
                
                # НОВОЕ: Если найден ближайший проем, проверяем совместимость ориентации
                if nearest_opening:
                    opening_orientation = get_opening_orientation(nearest_opening)
                    segment_orientation = analyze_segment_orientation(segment.vertices)
                    
                    # Если ориентации не совпадают, пропускаем создание сегмента
                    if opening_orientation != segment_orientation:
                        segment_vertices = [vertex]
                        continue
                
                segments.append(segment)
            
            # Начинаем новый сегмент с текущей вершины
            segment_vertices = [vertex]
    
    # Обрабатываем последний сегмент (замыкаем полигон)
    if len(segment_vertices) >= 2:
        start_vertex = segment_vertices[0]
        end_vertex = segment_vertices[-1]
        
        start_type = identify_vertex_type(start_vertex)
        end_type = identify_vertex_type(end_vertex)
        
        segment = WallSegment(
            vertices=segment_vertices,
            start_point=(start_vertex['x'], start_vertex['y']),
            end_point=(end_vertex['x'], end_vertex['y']),
            start_type=start_type,
            end_type=end_type,
            polygon_id="unknown",  # Будет установлено извне
            segment_id=f"segment_{len(segments) + 1}",
            opening_id=end_vertex.get('opening_id') if end_type == 'opening' else None,
            side=end_vertex.get('side') if end_type == 'opening' else None
        )
        
        # НОВОЕ: Если найден ближайший проем, проверяем совместимость ориентации
        nearest_opening = None
        if openings and end_vertex:
            nearest_opening = find_nearest_opening_to_vertex(end_vertex, openings)
        
        if nearest_opening:
            opening_orientation = get_opening_orientation(nearest_opening)
            segment_orientation = analyze_segment_orientation(segment.vertices)
            
            # Если ориентации не совпадают, пропускаем создание сегмента
            if opening_orientation != segment_orientation:
                return segments, used_opening_sides, used_junctions
        
        segments.append(segment)
    
    return segments, used_opening_sides, used_junctions

# =============================================================================
# ФУНКЦИИ ПРЕОБРАЗОВАНИЯ В BBOX
# =============================================================================

def analyze_segment_orientation(vertices: List[Dict]) -> str:
    """Анализирует ориентацию сегмента: 'horizontal', 'vertical', 'curved'"""
    if len(vertices) < 2:
        return 'unknown'
    
    # Проверяем все рёбра сегмента
    horizontal_count = 0
    vertical_count = 0
    
    for i in range(len(vertices) - 1):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        
        dx = abs(v2['x'] - v1['x'])
        dy = abs(v2['y'] - v1['y'])
        
        # Считаем ребро горизонтальным или вертикальным, если оно близко к этим направлениям
        tolerance = 10.0  # Увеличено с 5.0 до 10.0 пикселей
        
        if dx > dy and dy <= tolerance:
            horizontal_count += 1
        elif dy > dx and dx <= tolerance:
            vertical_count += 1
    
    total_edges = len(vertices) - 1
    
    if total_edges == 0:
        return 'unknown'
    
    if horizontal_count / total_edges >= 0.8:
        return 'horizontal'
    elif vertical_count / total_edges >= 0.8:
        return 'vertical'
    else:
        return 'curved'

def create_bbox_from_vertices(vertices: List[Dict], wall_thickness: float) -> Dict:
    """Создает bbox по списку вершин для изогнутого сегмента"""
    min_x = min(v['x'] for v in vertices)
    max_x = max(v['x'] for v in vertices)
    min_y = min(v['y'] for v in vertices)
    max_y = max(v['y'] for v in vertices)
    
    return {
        'x': min_x,
        'y': min_y,
        'width': max_x - min_x,
        'height': max_y - min_y,
        'orientation': 'curved',
        'segment_id': 'unknown',  # Будет установлено извне
        'polygon_id': 'unknown',  # Будет установлено извне
        'start_type': 'unknown',
        'end_type': 'unknown'
    }

def segment_to_bbox(segment: WallSegment, wall_thickness: float) -> Dict:
    """
    Преобразует сегмент в bbox
    
    Args:
        segment: Сегмент стены
        wall_thickness: Толщина стены (минимальная толщина двери)
    
    Returns:
        Словарь с параметрами bbox
    """
    vertices = segment.vertices
    if len(vertices) < 2:
        return None
    
    # Анализируем ориентацию сегмента
    orientation = analyze_segment_orientation(vertices)
    
    # Дополнительная проверка на основе соотношения сторон для изогнутых сегментов
    if orientation == 'curved':
        width = max(v['x'] for v in vertices) - min(v['x'] for v in vertices)
        height = max(v['y'] for v in vertices) - min(v['y'] for v in vertices)
        
        if width > height * 2:  # Ширина значительно больше высоты
            orientation = 'horizontal'
        elif height > width * 2:  # Высота значительно больше ширины
            orientation = 'vertical'
    
    if orientation == 'horizontal':
        # Горизонтальный сегмент
        min_x = min(v['x'] for v in vertices)
        max_x = max(v['x'] for v in vertices)
        y = vertices[0]['y']  # Все Y должны быть примерно одинаковыми
        
        return {
            'x': min_x,
            'y': y - wall_thickness / 2,
            'width': max_x - min_x,
            'height': wall_thickness,  # Ограничиваем толщиной стены
            'orientation': 'horizontal',
            'segment_id': segment.segment_id,
            'polygon_id': segment.polygon_id,
            'start_type': segment.start_type,
            'end_type': segment.end_type,
            'opening_id': segment.opening_id,
            'side': segment.side
        }
    
    elif orientation == 'vertical':
        # Вертикальный сегмент
        min_y = min(v['y'] for v in vertices)
        max_y = max(v['y'] for v in vertices)
        x = vertices[0]['x']  # Все X должны быть примерно одинаковыми
        
        return {
            'x': x - wall_thickness / 2,
            'y': min_y,
            'width': wall_thickness,  # Ограничиваем толщиной стены
            'height': max_y - min_y,
            'orientation': 'vertical',
            'segment_id': segment.segment_id,
            'polygon_id': segment.polygon_id,
            'start_type': segment.start_type,
            'end_type': segment.end_type,
            'opening_id': segment.opening_id,
            'side': segment.side
        }
    
    else:
        # Изогнутый сегмент - создаем bbox по экстремальным точкам
        bbox = create_bbox_from_vertices(vertices, wall_thickness)
        
        # Для изогнутых сегментов также ограничиваем толщину
        width = bbox['width']
        height = bbox['height']
        
        # Если ширина больше высоты, ограничиваем высоту
        if width > height:
            bbox['height'] = min(height, wall_thickness)
            bbox['y'] = bbox['y'] + (height - bbox['height']) / 2  # Центрируем по высоте
        # Если высота больше ширины, ограничиваем ширину
        else:
            bbox['width'] = min(width, wall_thickness)
            bbox['x'] = bbox['x'] + (width - bbox['width']) / 2  # Центрируем по ширине
        
        bbox.update({
            'segment_id': segment.segment_id,
            'polygon_id': segment.polygon_id,
            'start_type': segment.start_type,
            'end_type': segment.end_type,
            'opening_id': segment.opening_id,
            'side': segment.side
        })
        return bbox

# =============================================================================
# ФУНКЦИИ ОБРАБОТКИ T-СОЕДИНЕНИЙ
# =============================================================================

def check_segment_intersection(seg1: Dict, seg2: Dict) -> Tuple[bool, Dict]:
    """
    Проверяет пересечение двух сегментов и возвращает информацию о пересечении
    
    Returns:
        Tuple[bool, Dict]: (флаг пересечения, информация о пересечении)
    """
    x1_min, y1_min = seg1['x'], seg1['y']
    x1_max = x1_min + seg1['width']
    y1_max = y1_min + seg1['height']
    
    x2_min, y2_min = seg2['x'], seg2['y']
    x2_max = x2_min + seg2['width']
    y2_max = y2_min + seg2['height']
    
    # Проверяем пересечение
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    if x_overlap > 0 and y_overlap > 0:
        # Вычисляем область пересечения
        intersection = {
            'x': max(x1_min, x2_min),
            'y': max(y1_min, y2_min),
            'width': x_overlap,
            'height': y_overlap,
            'area': x_overlap * y_overlap
        }
        return True, intersection
    
    return False, {}

def find_intersection_with_junction(segment1: Dict, segment2: Dict,
                                  junctions: List[JunctionPoint], tolerance: float = 10.0) -> Optional[JunctionPoint]:
    """Находит junction, который совпадает с пересечением двух сегментов"""
    has_intersection, intersection = check_segment_intersection(segment1, segment2)
    
    if not has_intersection:
        return None
    
    # Вычисляем центр пересечения
    intersection_center_x = intersection['x'] + intersection['width'] / 2
    intersection_center_y = intersection['y'] + intersection['height'] / 2
    
    # Ищем junction близкий к центру пересечения
    for junction in junctions:
        distance = math.sqrt((junction.x - intersection_center_x)**2 +
                           (junction.y - intersection_center_y)**2)
        if distance <= tolerance:
            return junction
    
    return None

def trim_segment_to_junction(segment: Dict, junction: JunctionPoint) -> Dict:
    """
    Обрезает сегмент до junction
    Исправленная версия с правильной логикой определения положения junction
    """
    trimmed_segment = segment.copy()
    
    # Определяем ориентацию сегмента
    if segment['orientation'] == 'horizontal':
        # Горизонтальный сегмент
        seg_center_x = segment['x'] + segment['width'] / 2
        seg_center_y = segment['y'] + segment['height'] / 2
        
        # Обрезаем слева или справа в зависимости от положения junction
        if junction.x < seg_center_x:
            # Junction слева, обрезаем левую часть
            new_width = segment['x'] + segment['width'] - junction.x
            if new_width > 1:  # Проверяем, что ширина положительная
                trimmed_segment['x'] = junction.x
                trimmed_segment['width'] = new_width
            else:
                return None  # Сегмент слишком короткий после обрезки
        else:
            # Junction справа, обрезаем правую часть
            new_width = junction.x - segment['x']
            if new_width > 1:  # Проверяем, что ширина положительная
                trimmed_segment['width'] = new_width
            else:
                return None  # Сегмент слишком короткий после обрезки
    
    elif segment['orientation'] == 'vertical':
        # Вертикальный сегмент
        seg_center_x = segment['x'] + segment['width'] / 2
        seg_center_y = segment['y'] + segment['height'] / 2
        
        # Обрезаем сверху или снизу в зависимости от положения junction
        if junction.y < seg_center_y:
            # Junction сверху, обрезаем верхнюю часть
            new_height = segment['y'] + segment['height'] - junction.y
            if new_height > 1:  # Проверяем, что высота положительная
                trimmed_segment['y'] = junction.y
                trimmed_segment['height'] = new_height
            else:
                return None  # Сегмент слишком короткий после обрезки
        else:
            # Junction снизу, обрезаем нижнюю часть
            new_height = junction.y - segment['y']
            if new_height > 1:  # Проверяем, что высота положительная
                trimmed_segment['height'] = new_height
            else:
                return None  # Сегмент слишком короткий после обрезки
    
    return trimmed_segment

def determine_junction_type(junction: JunctionPoint, 
                          segments: List[Dict], 
                          wall_thickness: float,
                          tolerance: float = 15.0) -> str:
    """
    Определяет тип junction: 'T_junction', 'L_junction', 'cross_junction', 'unknown'
    
    Args:
        junction: Точка junction
        segments: Список всех сегментов
        wall_thickness: Толщина стены
        tolerance: Допуск для определения принадлежности
    
    Returns:
        Тип junction
    """
    # Находим все сегменты, которые проходят через junction
    connected_segments = []
    
    for segment in segments:
        # Проверяем, проходит ли сегмент через junction
        if segment['orientation'] == 'horizontal':
            # Горизонтальный сегмент
            seg_y = segment['y'] + segment['height'] / 2
            if abs(junction.y - seg_y) <= wall_thickness:
                # Проверяем, находится ли junction в пределах X сегмента
                seg_x_min = segment['x']
                seg_x_max = segment['x'] + segment['width']
                if seg_x_min - tolerance <= junction.x <= seg_x_max + tolerance:
                    connected_segments.append(segment)
        elif segment['orientation'] == 'vertical':
            # Вертикальный сегмент
            seg_x = segment['x'] + segment['width'] / 2
            if abs(junction.x - seg_x) <= wall_thickness:
                # Проверяем, находится ли junction в пределах Y сегмента
                seg_y_min = segment['y']
                seg_y_max = segment['y'] + segment['height']
                if seg_y_min - tolerance <= junction.y <= seg_y_max + tolerance:
                    connected_segments.append(segment)
    
    # Определяем тип junction на основе количества и ориентации подключенных сегментов
    if len(connected_segments) == 2:
        # Проверяем, перпендикулярны ли сегменты
        if (connected_segments[0]['orientation'] == 'horizontal' and 
            connected_segments[1]['orientation'] == 'vertical') or \
           (connected_segments[0]['orientation'] == 'vertical' and 
            connected_segments[1]['orientation'] == 'horizontal'):
            return 'L_junction'
        else:
            return 'unknown'
    elif len(connected_segments) == 3:
        # Проверяем, есть ли два сегмента одной ориентации и один другой
        horizontal_count = sum(1 for s in connected_segments if s['orientation'] == 'horizontal')
        vertical_count = sum(1 for s in connected_segments if s['orientation'] == 'vertical')
        
        if (horizontal_count == 2 and vertical_count == 1) or \
           (horizontal_count == 1 and vertical_count == 2):
            return 'T_junction'
        else:
            return 'unknown'
    elif len(connected_segments) == 4:
        # Проверяем, есть ли по два сегмента каждой ориентации
        horizontal_count = sum(1 for s in connected_segments if s['orientation'] == 'horizontal')
        vertical_count = sum(1 for s in connected_segments if s['orientation'] == 'vertical')
        
        if horizontal_count == 2 and vertical_count == 2:
            return 'cross_junction'
        else:
            return 'unknown'
    else:
        return 'unknown'

def create_segments_from_t_junction(junction: JunctionPoint, 
                                   segments: List[Dict], 
                                   wall_thickness: float) -> List[Dict]:
    """
    Создает три сегмента для T-соединения
    
    Args:
        junction: Точка T-соединения
        segments: Список всех сегментов
        wall_thickness: Толщина стены
    
    Returns:
        Список из трех сегментов
    """
    # Находим все сегменты, которые проходят через junction
    connected_segments = []
    
    for segment in segments:
        # Проверяем, проходит ли сегмент через junction
        if segment['orientation'] == 'horizontal':
            # Горизонтальный сегмент
            seg_y = segment['y'] + segment['height'] / 2
            if abs(junction.y - seg_y) <= wall_thickness:
                # Проверяем, находится ли junction в пределах X сегмента
                seg_x_min = segment['x']
                seg_x_max = segment['x'] + segment['width']
                if seg_x_min <= junction.x <= seg_x_max:
                    connected_segments.append(segment)
        elif segment['orientation'] == 'vertical':
            # Вертикальный сегмент
            seg_x = segment['x'] + segment['width'] / 2
            if abs(junction.x - seg_x) <= wall_thickness:
                # Проверяем, находится ли junction в пределах Y сегмента
                seg_y_min = segment['y']
                seg_y_max = segment['y'] + segment['height']
                if seg_y_min <= junction.y <= seg_y_max:
                    connected_segments.append(segment)
    
    # Разделяем сегменты по ориентации
    horizontal_segments = [s for s in connected_segments if s['orientation'] == 'horizontal']
    vertical_segments = [s for s in connected_segments if s['orientation'] == 'vertical']
    
    # Создаем новые сегменты
    new_segments = []
    
    # Создаем горизонтальный сегмент
    if horizontal_segments:
        # Находим самый левый и самый правый концы
        left_x = min(s['x'] for s in horizontal_segments)
        right_x = max(s['x'] + s['width'] for s in horizontal_segments)
        y = junction.y
        
        # Создаем сегмент от junction до левого конца
        if left_x < junction.x:
            left_segment = {
                'x': left_x,
                'y': y - wall_thickness / 2,
                'width': junction.x - left_x,
                'height': wall_thickness,
                'orientation': 'horizontal',
                'segment_id': f"t_junction_h_left_{junction.id}",
                'polygon_id': 't_junction',
                'start_type': 'junction',
                'end_type': 'vertex'
            }
            new_segments.append(left_segment)
        
        # Создаем сегмент от junction до правого конца
        if right_x > junction.x:
            right_segment = {
                'x': junction.x,
                'y': y - wall_thickness / 2,
                'width': right_x - junction.x,
                'height': wall_thickness,
                'orientation': 'horizontal',
                'segment_id': f"t_junction_h_right_{junction.id}",
                'polygon_id': 't_junction',
                'start_type': 'junction',
                'end_type': 'vertex'
            }
            new_segments.append(right_segment)
    
    # Создаем вертикальный сегмент
    if vertical_segments:
        # Находим самый верхний и самый нижний концы
        top_y = min(s['y'] for s in vertical_segments)
        bottom_y = max(s['y'] + s['height'] for s in vertical_segments)
        x = junction.x
        
        # Создаем сегмент от junction до верхнего конца
        if top_y < junction.y:
            top_segment = {
                'x': x - wall_thickness / 2,
                'y': top_y,
                'width': wall_thickness,
                'height': junction.y - top_y,
                'orientation': 'vertical',
                'segment_id': f"t_junction_v_top_{junction.id}",
                'polygon_id': 't_junction',
                'start_type': 'junction',
                'end_type': 'vertex'
            }
            new_segments.append(top_segment)
        
        # Создаем сегмент от junction до нижнего конца
        if bottom_y > junction.y:
            bottom_segment = {
                'x': x - wall_thickness / 2,
                'y': junction.y,
                'width': wall_thickness,
                'height': bottom_y - junction.y,
                'orientation': 'vertical',
                'segment_id': f"t_junction_v_bottom_{junction.id}",
                'polygon_id': 't_junction',
                'start_type': 'junction',
                'end_type': 'vertex'
            }
            new_segments.append(bottom_segment)
    
    return new_segments

def create_junction_aware_t_junctions(segments: List[Dict],
                                     junctions: List[JunctionPoint],
                                     wall_thickness: float,
                                     openings: List[Dict] = None) -> List[Dict]:
    """
    Улучшенная функция создания T-соединений между сегментами с учетом junctions
    
    Args:
        segments: Список сегментов
        junctions: Список junctions
        wall_thickness: Толщина стены
        openings: Список проемов для проверки ориентации
    
    Returns:
        Обновленный список сегментов с правильными T-соединениями
    """
    if not segments:
        return []
    
    processed_segments = segments.copy()
    junctions_to_process = junctions.copy()
    
    # ИСПРАВЛЕНО: Создаем словарь для отслеживания связей проем-junction
    # Это поможет гарантировать выполнение правила "от проема до junction - только один bbox стены"
    opening_junction_to_segments = {}
    
    # Сначала собираем все существующие связи проем-junction
    if openings:
        for segment in processed_segments:
            opening_id = segment.get('opening_id')
            start_type = segment.get('start_type', 'vertex')
            end_type = segment.get('end_type', 'vertex')
            
            # Ищем сегменты, которые начинаются или заканчиваются в проеме
            if opening_id and (start_type == 'opening' or end_type == 'opening'):
                # Проверяем, соединяется ли этот сегмент с junction
                segment_start = (segment['x'], segment['y'])
                segment_end = (segment['x'] + segment['width'], segment['y'] + segment['height'])
                
                # Определяем центр сегмента
                if segment['orientation'] == 'horizontal':
                    segment_center = (segment['x'] + segment['width'] / 2, segment['y'] + segment['height'] / 2)
                else:
                    segment_center = (segment['x'] + segment['width'] / 2, segment['y'] + segment['height'] / 2)
                
                # Ищем ближайший junction к этому сегменту
                for junction in junctions_to_process:
                    junction_point = (junction.x, junction.y)
                    
                    # Проверяем, находится ли junction на одном из концов сегмента
                    dist_to_start = math.sqrt((junction_point[0] - segment_start[0])**2 +
                                             (junction_point[1] - segment_start[1])**2)
                    dist_to_end = math.sqrt((junction_point[0] - segment_end[0])**2 +
                                           (junction_point[1] - segment_end[1])**2)
                    dist_to_center = math.sqrt((junction_point[0] - segment_center[0])**2 +
                                              (junction_point[1] - segment_center[1])**2)
                    
                    # Если junction находится близко к сегменту, считаем их соединенными
                    tolerance = 10.0
                    if (dist_to_start <= tolerance or dist_to_end <= tolerance or
                        (dist_to_center <= tolerance and
                         ((segment['orientation'] == 'horizontal' and abs(junction.y - segment_center[1]) <= tolerance) or
                          (segment['orientation'] == 'vertical' and abs(junction.x - segment_center[0]) <= tolerance)))):
                        
                        # Создаем ключ для пары (opening_id, junction_id)
                        key = (opening_id, junction.id)
                        if key not in opening_junction_to_segments:
                            opening_junction_to_segments[key] = []
                        
                        opening_junction_to_segments[key].append(segment)
                        break
    
    # Обрабатываем каждый junction
    for junction in junctions_to_process:
        # НОВОЕ: Находим ближайший проем к junction
        nearest_opening = None
        if openings:
            nearest_opening = find_nearest_opening_to_junction(junction, openings)
        
        # НОВОЕ: Если найден ближайший проем, определяем его ориентацию
        required_orientation = None
        if nearest_opening:
            required_orientation = get_opening_orientation(nearest_opening)
        
        # Определяем тип junction
        junction_type = determine_junction_type(junction, processed_segments, wall_thickness)
        
        if junction_type == 'T_junction':
            # Находим все сегменты, которые проходят через junction
            connected_segments = []
            
            for segment in processed_segments:
                # Проверяем, проходит ли сегмент через junction
                if segment['orientation'] == 'horizontal':
                    # Горизонтальный сегмент
                    seg_y = segment['y'] + segment['height'] / 2
                    if abs(junction.y - seg_y) <= wall_thickness:
                        # Проверяем, находится ли junction в пределах X сегмента
                        seg_x_min = segment['x']
                        seg_x_max = segment['x'] + segment['width']
                        if seg_x_min <= junction.x <= seg_x_max:
                            # НОВОЕ: Проверяем совместимость ориентации
                            if required_orientation is None or segment['orientation'] == required_orientation:
                                connected_segments.append(segment)
                elif segment['orientation'] == 'vertical':
                    # Вертикальный сегмент
                    seg_x = segment['x'] + segment['width'] / 2
                    if abs(junction.x - seg_x) <= wall_thickness:
                        # Проверяем, находится ли junction в пределах Y сегмента
                        seg_y_min = segment['y']
                        seg_y_max = segment['y'] + segment['height']
                        if seg_y_min <= junction.y <= seg_y_max:
                            # НОВОЕ: Проверяем совместимость ориентации
                            if required_orientation is None or segment['orientation'] == required_orientation:
                                connected_segments.append(segment)
            
            # ИСПРАВЛЕНО: Проверяем, не нарушает ли создание T-соединения правило
            # "от проема до junction - только один bbox стены"
            if nearest_opening:
                opening_id = nearest_opening.get('id')
                if opening_id:
                    # Проверяем, есть ли уже сегменты для этой пары (opening_id, junction.id)
                    key = (opening_id, junction.id)
                    if key in opening_junction_to_segments:
                        # Если есть уже сегменты для этой пары, не создаем T-соединение
                        # чтобы не нарушить правило
                        print(f"    Пропуск T-соединения для junction {junction.id} из-за существующей связи с проемом {opening_id}")
                        continue
            
            # Создаем три сегмента для T-соединения
            new_segments = create_segments_from_t_junction(junction, connected_segments, wall_thickness)
            
            # Удаляем старые сегменты, которые проходят через junction
            segments_to_remove = []
            for segment in processed_segments:
                if segment['orientation'] == 'horizontal':
                    # Горизонтальный сегмент
                    seg_y = segment['y'] + segment['height'] / 2
                    if abs(junction.y - seg_y) <= wall_thickness:
                        # Проверяем, находится ли junction в пределах X сегмента
                        seg_x_min = segment['x']
                        seg_x_max = segment['x'] + segment['width']
                        if seg_x_min <= junction.x <= seg_x_max:
                            # НОВОЕ: Проверяем совместимость ориентации
                            if required_orientation is None or segment['orientation'] == required_orientation:
                                segments_to_remove.append(segment)
                elif segment['orientation'] == 'vertical':
                    # Вертикальный сегмент
                    seg_x = segment['x'] + segment['width'] / 2
                    if abs(junction.x - seg_x) <= wall_thickness:
                        # Проверяем, находится ли junction в пределах Y сегмента
                        seg_y_min = segment['y']
                        seg_y_max = segment['y'] + segment['height']
                        if seg_y_min <= junction.y <= seg_y_max:
                            # НОВОЕ: Проверяем совместимость ориентации
                            if required_orientation is None or segment['orientation'] == required_orientation:
                                segments_to_remove.append(segment)
            
            # Удаляем старые сегменты
            for segment in segments_to_remove:
                if segment in processed_segments:
                    processed_segments.remove(segment)
            
            # Добавляем новые сегменты
            processed_segments.extend(new_segments)
    
    # Фильтруем сегменты с нулевыми размерами
    result_segments = []
    for seg in processed_segments:
        if seg['width'] > 1 and seg['height'] > 1:
            result_segments.append(seg)
    
    return result_segments

# =============================================================================
# НОВАЯ ФУНКЦИЯ ДЛЯ ПРИМЕНЕНИЯ ПРАВИЛА "ОТ ПРОЕМА ДО JUNCTION - ТОЛЬКО ОДИН BBOX"
# =============================================================================

def validate_opening_junction_rule(segments: List[Dict],
                                  openings: List[Dict],
                                  junctions: List[JunctionPoint],
                                  tolerance: float = 10.0) -> Dict[str, Any]:
    """
    Валидирует выполнение правила "от проема до junction - только один bbox стены"
    
    Args:
        segments: Список всех сегментов стен
        openings: Список всех проемов
        junctions: Список всех junctions
        tolerance: Допуск для определения принадлежности
    
    Returns:
        Словарь с результатами валидации
    """
    validation_result = {
        'is_valid': True,
        'violations': [],
        'total_pairs': 0,
        'valid_pairs': 0,
        'invalid_pairs': 0
    }
    
    if not segments or not openings or not junctions:
        return validation_result
    
    # Создаем словарь для отслеживания связей проем-junction
    opening_junction_to_segments = {}
    
    # Находим все сегменты, соединяющие проемы с junctions
    for segment in segments:
        opening_id = segment.get('opening_id')
        start_type = segment.get('start_type', 'vertex')
        end_type = segment.get('end_type', 'vertex')
        
        # Ищем сегменты, которые начинаются или заканчиваются в проеме
        if opening_id and (start_type == 'opening' or end_type == 'opening'):
            # Проверяем, соединяется ли этот сегмент с junction
            segment_start = (segment['x'], segment['y'])
            segment_end = (segment['x'] + segment['width'], segment['y'] + segment['height'])
            
            # Определяем центр сегмента
            if segment['orientation'] == 'horizontal':
                segment_center = (segment['x'] + segment['width'] / 2, segment['y'] + segment['height'] / 2)
            else:
                segment_center = (segment['x'] + segment['width'] / 2, segment['y'] + segment['height'] / 2)
            
            # Ищем ближайший junction к этому сегменту
            for junction in junctions:
                junction_point = (junction.x, junction.y)
                
                # Проверяем, находится ли junction на одном из концов сегмента
                dist_to_start = math.sqrt((junction_point[0] - segment_start[0])**2 +
                                         (junction_point[1] - segment_start[1])**2)
                dist_to_end = math.sqrt((junction_point[0] - segment_end[0])**2 +
                                       (junction_point[1] - segment_end[1])**2)
                dist_to_center = math.sqrt((junction_point[0] - segment_center[0])**2 +
                                          (junction_point[1] - segment_center[1])**2)
                
                # Если junction находится близко к сегменту, считаем их соединенными
                if (dist_to_start <= tolerance or dist_to_end <= tolerance or
                    (dist_to_center <= tolerance and
                     ((segment['orientation'] == 'horizontal' and abs(junction.y - segment_center[1]) <= tolerance) or
                      (segment['orientation'] == 'vertical' and abs(junction.x - segment_center[0]) <= tolerance)))):
                    
                    # Создаем ключ для пары (opening_id, junction_id)
                    key = (opening_id, junction.id)
                    if key not in opening_junction_to_segments:
                        opening_junction_to_segments[key] = []
                    
                    opening_junction_to_segments[key].append(segment)
                    break
    
    # Проверяем каждую пару (проем, junction)
    for (opening_id, junction_id), segment_list in opening_junction_to_segments.items():
        validation_result['total_pairs'] += 1
        
        if len(segment_list) > 1:
            # Нарушение правила: несколько сегментов для одной пары
            validation_result['is_valid'] = False
            validation_result['invalid_pairs'] += 1
            validation_result['violations'].append({
                'opening_id': opening_id,
                'junction_id': junction_id,
                'segment_count': len(segment_list),
                'segment_ids': [s.get('segment_id', 'unknown') for s in segment_list]
            })
        else:
            validation_result['valid_pairs'] += 1
    
    return validation_result

def filter_segments_opening_to_junction(segments: List[Dict],
                                     openings: List[Dict],
                                     junctions: List[JunctionPoint],
                                     tolerance: float = 10.0) -> List[Dict]:
    """
    Применяет правило "от проема до junction - только один bbox стены"
    
    Args:
        segments: Список всех сегментов стен
        openings: Список всех проемов
        junctions: Список всех junctions
        tolerance: Допуск для определения принадлежности
    
    Returns:
        Отфильтрованный список сегментов
    """
    if not segments or not openings or not junctions:
        return segments
    
    # Создаем словарь для отслеживания связей проем-junction
    # Ключ: (opening_id, junction_id), Значение: список сегментов
    opening_junction_to_segments = {}
    
    # Находим все сегменты, соединяющие проемы с junctions
    for segment in segments:
        opening_id = segment.get('opening_id')
        start_type = segment.get('start_type', 'vertex')
        end_type = segment.get('end_type', 'vertex')
        
        # Ищем сегменты, которые начинаются или заканчиваются в проеме
        if opening_id and (start_type == 'opening' or end_type == 'opening'):
            # Проверяем, соединяется ли этот сегмент с junction
            segment_start = (segment['x'], segment['y'])
            segment_end = (segment['x'] + segment['width'], segment['y'] + segment['height'])
            
            # Определяем центр сегмента
            if segment['orientation'] == 'horizontal':
                segment_center = (segment['x'] + segment['width'] / 2, segment['y'] + segment['height'] / 2)
            else:
                segment_center = (segment['x'] + segment['width'] / 2, segment['y'] + segment['height'] / 2)
            
            # Ищем ближайший junction к этому сегменту
            for junction in junctions:
                junction_point = (junction.x, junction.y)
                
                # Проверяем, находится ли junction на одном из концов сегмента
                dist_to_start = math.sqrt((junction_point[0] - segment_start[0])**2 +
                                         (junction_point[1] - segment_start[1])**2)
                dist_to_end = math.sqrt((junction_point[0] - segment_end[0])**2 +
                                       (junction_point[1] - segment_end[1])**2)
                dist_to_center = math.sqrt((junction_point[0] - segment_center[0])**2 +
                                          (junction_point[1] - segment_center[1])**2)
                
                # Если junction находится близко к сегменту, считаем их соединенными
                if (dist_to_start <= tolerance or dist_to_end <= tolerance or
                    (dist_to_center <= tolerance and
                     ((segment['orientation'] == 'horizontal' and abs(junction.y - segment_center[1]) <= tolerance) or
                      (segment['orientation'] == 'vertical' and abs(junction.x - segment_center[0]) <= tolerance)))):
                    
                    # Создаем ключ для пары (opening_id, junction_id)
                    key = (opening_id, junction.id)
                    if key not in opening_junction_to_segments:
                        opening_junction_to_segments[key] = []
                    
                    opening_junction_to_segments[key].append(segment)
                    break
    
    # Фильтруем сегменты, оставляя только один сегмент для каждой пары проем-junction
    filtered_segments = []
    used_segments = set()
    
    # Сначала добавляем все сегменты, не соединяющие проемы с junctions
    for segment in segments:
        opening_id = segment.get('opening_id')
        start_type = segment.get('start_type', 'vertex')
        end_type = segment.get('end_type', 'vertex')
        
        # Если сегмент не соединяет проем с junction, добавляем его
        if not opening_id or (start_type != 'opening' and end_type != 'opening'):
            filtered_segments.append(segment)
            used_segments.add(id(segment))
    
    # Затем для каждой пары (проем, junction) оставляем только один сегмент
    filtered_pairs = 0
    for (opening_id, junction_id), segment_list in opening_junction_to_segments.items():
        if not segment_list:
            continue
        
        # ИСПРАВЛЕНО: Для каждой пары (проем, junction) оставляем только один сегмент
        if len(segment_list) > 1:
            # Выбираем самый длинный сегмент
            longest_segment = max(segment_list, key=lambda s: s['width'] + s['height'])
            filtered_segments.append(longest_segment)
            used_segments.add(id(longest_segment))
            filtered_pairs += 1
        else:
            # Если сегмент всего один, просто добавляем его
            filtered_segments.append(segment_list[0])
            used_segments.add(id(segment_list[0]))
            filtered_pairs += 1
    
    print(f"  ✓ Отфильтровано пар (проем-junction): {filtered_pairs}")
    
    return filtered_segments

# =============================================================================
# ОСНОВНОЙ АЛГОРИТМ ОБРАБОТКИ
# =============================================================================

def process_wall_polygons_with_junctions(data: Dict, wall_thickness: float) -> Tuple[List[Dict], Dict[str, set], set]:
    """
    Основная функция обработки стеновых полигонов с учетом junctions
    
    Алгоритм:
    1. Загружаем и парсим junctions и проемы
    2. Для каждого полигона:
       a. Находим junctions на границе полигона
       b. Находим угловые junctions (Г-образные повороты)
       c. Вставляем junctions в полигон, включая угловые
       d. Находим края проемов на границе полигона с учетом ориентации
       e. Вставляем середины краев проемов в полигон
       f. Разбиваем полигон на сегменты
       g. Преобразуем сегменты в bbox
    3. Создаем T-соединения с учетом junctions
    4. Возвращаем список bbox стен, использованные стороны проемов и использованные junctions
    """
    
    # Шаг 1: Подготовка данных
    junctions = parse_junctions(data)
    openings, opening_edges = parse_openings(data)
    wall_polygons = data.get('wall_polygons', [])
    
    all_segments = []
    all_used_sides = {}
    all_used_junctions = set()
    
    # Шаг 2: Обработка каждого полигона
    for polygon in wall_polygons:
        polygon_id = polygon.get('id', 'unknown')
        vertices = polygon.get('vertices', [])
        
        if len(vertices) < 2:
            continue
        
        print(f"  Обработка полигона {polygon_id} ({len(vertices)} вершин)")
        
        # Шаг 2a: Поиск junctions на полигоне
        junctions_on_polygon = find_junctions_on_polygon(vertices, junctions)
        print(f"    Найдено junctions: {len(junctions_on_polygon)}")
        
        # Шаг 2b: Поиск угловых junctions (Г-образные повороты)
        corner_junctions = detect_corner_junctions(vertices, junctions, wall_thickness)
        print(f"    Найдено угловых junctions: {len(corner_junctions)}")
        
        # Шаг 2c: Вставка junctions в полигон
        vertices_with_junctions = insert_junctions_in_polygon(vertices, junctions_on_polygon)
        vertices_with_corner_junctions = split_polygon_at_corner_junctions(vertices_with_junctions, corner_junctions)
        print(f"    Всего вершин после вставки junctions: {len(vertices_with_corner_junctions)}")
        
        # Шаг 2d: Поиск краев проемов на полигоне с учетом ориентации
        opening_edges_on_polygon = find_opening_edges_on_polygon(vertices_with_corner_junctions, opening_edges)
        print(f"    Найдено краев проемов: {len(opening_edges_on_polygon)}")
        
        # Шаг 2e: Вставка середин краев проемов
        vertices_with_openings = insert_opening_midpoints_in_polygon(vertices_with_corner_junctions, opening_edges_on_polygon)
        print(f"    Всего вершин после вставки: {len(vertices_with_openings)}")
        
        # Шаг 2f: Разбиение на сегменты
        segments, used_sides, used_junctions = split_polygon_into_segments(
            vertices_with_openings, all_used_sides, all_used_junctions, junctions, openings
        )
        print(f"    Создано сегментов: {len(segments)}")
        
        # Обновляем глобальные словари/множества использованных сторон/junctions
        for opening_id, sides in used_sides.items():
            if opening_id not in all_used_sides:
                all_used_sides[opening_id] = set()
            all_used_sides[opening_id].update(sides)
        
        all_used_junctions.update(used_junctions)
        
        # Устанавливаем polygon_id для каждого сегмента
        for segment in segments:
            segment.polygon_id = polygon_id
        
        # Шаг 2g: Преобразование в bbox
        for segment in segments:
            bbox = segment_to_bbox(segment, wall_thickness)
            if bbox:
                all_segments.append(bbox)
    
    print(f"\nВсего создано {len(all_segments)} bbox стен")
    return all_segments, all_used_sides, all_used_junctions

# =============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# =============================================================================

def calculate_svg_dimensions(data: Dict, segments: List[Dict]) -> Tuple[float, float, float]:
    """
    Вычисляет размеры SVG и масштабный коэффициент
    Возвращает: (max_x, max_y, inverse_scale)
    """
    max_x = 0
    max_y = 0
    
    # Проверяем сегменты стен
    for segment in segments:
        max_x = max(max_x, segment['x'] + segment['width'])
        max_y = max(max_y, segment['y'] + segment['height'])
    
    # Проверяем окна и двери
    for opening in data.get('openings', []):
        bbox = opening.get('bbox', {})
        max_x = max(max_x, bbox.get('x', 0) + bbox.get('width', 0))
        max_y = max(max_y, bbox.get('y', 0) + bbox.get('height', 0))
    
    # Проверяем полигоны колонн
    for polygon in data.get('pillar_polygons', []):
        for vertex in polygon.get('vertices', []):
            max_x = max(max_x, vertex.get('x', 0))
            max_y = max(max_y, vertex.get('y', 0))
    
    # Проверяем junctions
    for junction in data.get('junctions', []):
        max_x = max(max_x, junction.get('x', 0))
        max_y = max(max_y, junction.get('y', 0))
    
    # Получаем масштабный коэффициент из метаданных
    scale_factor = data.get('metadata', {}).get('scale_factor', 1.0)
    inverse_scale = 1.0 / scale_factor
    
    print(f"  ✓ Максимальные координаты: {max_x}x{max_y}")
    print(f"  ✓ Масштабный коэффициент: {scale_factor}")
    print(f"  ✓ Обратный масштаб: {inverse_scale}")
    
    return max_x, max_y, inverse_scale

def transform_coordinates(x: float, y: float, inverse_scale: float, padding: float = 50) -> Tuple[float, float]:
    """Преобразует координаты из JSON в SVG с учетом масштаба и отступов"""
    svg_x = x * inverse_scale + padding
    svg_y = y * inverse_scale + padding
    return svg_x, svg_y

def create_svg_document(output_path: str, width: int, height: int, data: Dict) -> svgwrite.Drawing:
    """Создает SVG документ с базовыми настройками"""
    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='full')
    
    # Добавляем белый фон
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    
    print(f"  ✓ SVG документ создан: {width}x{height}")
    return dwg

def define_styles() -> Dict[str, Dict[str, Any]]:
    """Определяет стили для разных типов объектов"""
    return {
        'wall': {
            'stroke': '#FF0000',  # Красный
            'stroke_width': 2,
            'fill': 'none',
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        },
        'pillar': {
            'stroke': '#8B4513',  # Коричневый
            'stroke_width': 2,
            'fill': '#D2691E',   # Светло-коричневый
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        },
        'window': {
            'stroke': '#0000FF',  # Синий
            'stroke_width': 3,
            'fill': 'none',
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        },
        'door': {
            'stroke': '#008000',  # Зеленый
            'stroke_width': 3,
            'fill': 'none',
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        }
    }

def draw_all_wall_polygons(dwg: svgwrite.Drawing, data: Dict, inverse_scale: float, padding: float) -> None:
    """
    Отображает все стеновые полигоны из JSON серым цветом с нумерацией
    """
    wall_polygons_group = dwg.add(dwg.g(id='wall_polygons'))
    
    wall_polygons = data.get('wall_polygons', [])
    print(f"  ✓ Отрисовка {len(wall_polygons)} стеновых полигонов")
    
    # Стиль для полигонов стен
    wall_polygon_style = {
        'stroke': '#808080',  # Серый
        'stroke_width': 1,
        'fill': 'none',
        'stroke_dasharray': '5,5',  # Пунктирная линия
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round'
    }
    
    for idx, polygon in enumerate(wall_polygons):
        polygon_id = polygon.get('id', f'wall_polygon_{idx}')
        vertices = polygon.get('vertices', [])
        
        if vertices:
            # Преобразуем координаты
            scaled_vertices = [
                transform_coordinates(v['x'], v['y'], inverse_scale, padding)
                for v in vertices
            ]
            
            # Создаем полигон
            polygon_element = dwg.polygon(scaled_vertices, **wall_polygon_style)
            wall_polygons_group.add(polygon_element)
            
            # Добавляем номер полигона
            center_x = sum(v[0] for v in scaled_vertices) / len(scaled_vertices)
            center_y = sum(v[1] for v in scaled_vertices) / len(scaled_vertices)
            
            text = dwg.text(
                f"{polygon_id}",
                insert=(center_x, center_y),
                text_anchor='middle',
                fill='#606060',  # Темно-серый
                font_size='10px',
                font_weight='bold'
            )
            wall_polygons_group.add(text)

def draw_junctions(dwg: svgwrite.Drawing, junctions: List[JunctionPoint],
                   inverse_scale: float, padding: float) -> None:
    """Отображает junctions с улучшенной визуализацией"""
    junctions_group = dwg.add(dwg.g(id='junctions'))
    
    print(f"  ✓ Отрисовка {len(junctions)} junctions")
    
    # Стиль для junctions
    junction_style = {
        'stroke': '#0000FF',  # Синий
        'stroke_width': 2,
        'fill': '#0066FF',    # Ярко-синий
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round'
    }
    
    for idx, junction in enumerate(junctions):
        # Преобразуем координаты
        svg_x, svg_y = transform_coordinates(junction.x, junction.y, inverse_scale, padding)
        
        # Создаем кружок для junction
        circle = dwg.circle(center=(svg_x, svg_y), r=5, **junction_style)
        junctions_group.add(circle)
        
        # Добавляем номер и тип
        text = dwg.text(
            f"J{idx+1}",
            insert=(svg_x + 10, svg_y - 5),
            text_anchor='start',
            fill='#0000FF',
            font_size='8px',
            font_weight='bold'
        )
        junctions_group.add(text)
        
        # Добавляем тип (меньшим шрифтом)
        type_text = dwg.text(
            f"{junction.junction_type}",
            insert=(svg_x + 10, svg_y + 8),
            text_anchor='start',
            fill='#0000FF',
            font_size='6px'
        )
        junctions_group.add(type_text)

def draw_connections_to_junctions(dwg: svgwrite.Drawing, segments: List[Dict],
                                 junctions: List[JunctionPoint],
                                 inverse_scale: float, padding: float) -> None:
    """Отображает связи между сегментами и junctions"""
    connections_group = dwg.add(dwg.g(id='junction_connections'))
    
    # Стиль для линий связей
    connection_style = {
        'stroke': '#FF00FF',  # Фиолетовый
        'stroke_width': 1,
        'stroke_dasharray': '2,2',  # Пунктирная линия
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round',
        'opacity': 0.5
    }
    
    for segment in segments:
        start_type = segment.get('start_type', 'vertex')
        end_type = segment.get('end_type', 'vertex')
        
        # Если сегмент начинается или заканчивается в junction, рисуем связь
        if start_type == 'junction':
            start_x = segment['x']
            start_y = segment['y']
            if segment['orientation'] == 'horizontal':
                start_x += segment['width'] / 2
                start_y += segment['height'] / 2
            else:
                start_x += segment['width'] / 2
                start_y += segment['height'] / 2
            
            # Находим соответствующий junction
            for junction in junctions:
                dist = math.sqrt((junction.x - start_x)**2 + (junction.y - start_y)**2)
                if dist < 10:  # Порог для определения соответствия
                    svg_x, svg_y = transform_coordinates(junction.x, junction.y, inverse_scale, padding)
                    seg_x, seg_y = transform_coordinates(start_x, start_y, inverse_scale, padding)
                    
                    line = dwg.line(start=(seg_x, seg_y), end=(svg_x, svg_y), **connection_style)
                    connections_group.add(line)
                    break
        
        if end_type == 'junction':
            end_x = segment['x'] + segment['width']
            end_y = segment['y'] + segment['height']
            if segment['orientation'] == 'horizontal':
                end_x = segment['x'] + segment['width'] / 2
                end_y = segment['y'] + segment['height'] / 2
            else:
                end_x = segment['x'] + segment['width'] / 2
                end_y = segment['y'] + segment['height'] / 2
            
            # Находим соответствующий junction
            for junction in junctions:
                dist = math.sqrt((junction.x - end_x)**2 + (junction.y - end_y)**2)
                if dist < 10:  # Порог для определения соответствия
                    svg_x, svg_y = transform_coordinates(junction.x, junction.y, inverse_scale, padding)
                    seg_x, seg_y = transform_coordinates(end_x, end_y, inverse_scale, padding)
                    
                    line = dwg.line(start=(seg_x, seg_y), end=(svg_x, svg_y), **connection_style)
                    connections_group.add(line)
                    break

def draw_wall_bboxes(dwg: svgwrite.Drawing, segments: List[Dict], inverse_scale: float, padding: float, styles: Dict) -> None:
    """Отрисовывает bbox стен с учетом junctions"""
    walls_group = dwg.add(dwg.g(id='walls'))
    
    print(f"  ✓ Отрисовка {len(segments)} сегментов стен")
    
    for idx, segment in enumerate(segments):
        # Преобразуем координаты
        x, y = transform_coordinates(segment['x'], segment['y'], inverse_scale, padding)
        width = segment['width'] * inverse_scale
        height = segment['height'] * inverse_scale
        
        # Создаем прямоугольник
        rect = dwg.rect(insert=(x, y), size=(width, height), **styles['wall'])
        walls_group.add(rect)
        
        # Добавляем номер сегмента с меткой ориентации
        orientation_label = 'h' if segment['orientation'] == 'horizontal' else 'v'
        text = dwg.text(
            f"W{idx+1}{orientation_label}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='red',
            font_size='12px',
            font_weight='bold'
        )
        walls_group.add(text)

def draw_pillars(dwg: svgwrite.Drawing, data: Dict, inverse_scale: float, padding: float, styles: Dict) -> None:
    """Отрисовывает колонны (полигоны с заливкой)"""
    pillars_group = dwg.add(dwg.g(id='pillars'))
    
    pillar_polygons = data.get('pillar_polygons', [])
    print(f"  ✓ Отрисовка {len(pillar_polygons)} колонн")
    
    for idx, polygon in enumerate(pillar_polygons):
        vertices = polygon.get('vertices', [])
        if vertices:
            # Преобразуем координаты
            scaled_vertices = [
                transform_coordinates(v['x'], v['y'], inverse_scale, padding)
                for v in vertices
            ]
            
            # Создаем полигон с заливкой
            polygon_element = dwg.polygon(scaled_vertices, **styles['pillar'])
            pillars_group.add(polygon_element)
            
            # Добавляем номер колонны
            center_x = sum(v[0] for v in scaled_vertices) / len(scaled_vertices)
            center_y = sum(v[1] for v in scaled_vertices) / len(scaled_vertices)
            
            text = dwg.text(
                f"P{idx+1}",
                insert=(center_x, center_y),
                text_anchor='middle',
                fill='white',
                font_size='12px',
                font_weight='bold'
            )
            pillars_group.add(text)

def draw_openings_bboxes(dwg: svgwrite.Drawing, data: Dict, inverse_scale: float, padding: float, styles: Dict) -> None:
    """Отрисовывает окна и двери (прямоугольники)"""
    openings_group = dwg.add(dwg.g(id='openings'))
    
    windows_group = openings_group.add(dwg.g(id='windows'))
    doors_group = openings_group.add(dwg.g(id='doors'))
    
    openings = data.get('openings', [])
    windows_count = 0
    doors_count = 0
    
    for opening in openings:
        opening_type = opening.get('type', '')
        bbox = opening.get('bbox', {})
        
        if bbox:
            # Преобразуем координаты
            x, y = transform_coordinates(bbox['x'], bbox['y'], inverse_scale, padding)
            width = bbox['width'] * inverse_scale
            height = bbox['height'] * inverse_scale
            
            # Определяем ориентацию проема
            orientation_label = 'h' if width > height else 'v'
            
            if opening_type == 'window':
                # Создаем окно
                rect = dwg.rect(insert=(x, y), size=(width, height), **styles['window'])
                windows_group.add(rect)
                windows_count += 1
                
                # Добавляем метку
                opening_id = opening.get('id', '').split('_')[-1] if opening.get('id') else str(windows_count)
                text = dwg.text(
                    f"W{opening_id}{orientation_label}",
                    insert=(x + width/2, y + height/2),
                    text_anchor='middle',
                    fill='black',
                    font_size='10px',
                    font_weight='bold'
                )
                windows_group.add(text)
                
            elif opening_type == 'door':
                # Создаем дверь
                rect = dwg.rect(insert=(x, y), size=(width, height), **styles['door'])
                doors_group.add(rect)
                doors_count += 1
                
                # Добавляем метку
                opening_id = opening.get('id', '').split('_')[-1] if opening.get('id') else str(doors_count)
                text = dwg.text(
                    f"D{opening_id}{orientation_label}",
                    insert=(x + width/2, y + height/2),
                    text_anchor='middle',
                    fill='black',
                    font_size='10px',
                    font_weight='bold'
                )
                doors_group.add(text)
    
    print(f"  ✓ Отрисовка {windows_count} окон и {doors_count} дверей")

def add_legend(dwg: svgwrite.Drawing, width: int, height: int, styles: Dict) -> None:
    """Добавляет легенду с описанием цветов"""
    legend_group = dwg.add(dwg.g(id='legend'))
    
    # Позиция легенды
    legend_x = 20
    legend_y = height - 240  # Увеличил для junctions
    item_height = 25
    
    # Заголовок легенды
    title = dwg.text(
        "Легенда:",
        insert=(legend_x, legend_y),
        fill='black',
        font_size='16px',
        font_weight='bold'
    )
    legend_group.add(title)
    
    # Элементы легенды
    legend_items = [
        ("Стены (bbox)", styles['wall']),
        ("Стеновые полигоны (JSON)", {'stroke': '#808080', 'fill': 'none', 'stroke_dasharray': '5,5'}),
        ("Junctions (JSON)", {'stroke': '#0000FF', 'fill': '#0066FF'}),
        ("Связи с junctions", {'stroke': '#FF00FF', 'fill': 'none', 'stroke_dasharray': '2,2'}),
        ("Колонны", styles['pillar']),
        ("Окна", styles['window']),
        ("Двери", styles['door'])
    ]
    
    for i, (label, style) in enumerate(legend_items):
        y_pos = legend_y + (i + 1) * item_height
        
        # Прямоугольник с цветом
        rect = dwg.rect(
            insert=(legend_x, y_pos - 10),
            size=(20, 15),
            **{k: v for k, v in style.items() if k in ['fill', 'stroke', 'stroke_dasharray']}
        )
        legend_group.add(rect)
        
        # Текст
        text = dwg.text(
            label,
            insert=(legend_x + 30, y_pos + 3),
            fill='black',
            font_size='14px'
        )
        legend_group.add(text)
    
    print("  ✓ Легенда добавлена")

def add_title(dwg: svgwrite.Drawing, width: int, data: Dict) -> None:
    """Добавляет заголовок SVG"""
    title_group = dwg.add(dwg.g(id='title'))
    
    # Заголовок
    title = dwg.text(
        "План этажа - Векторное представление с учетом Junctions (исправленная версия v8)",
        insert=(width/2, 30),
        text_anchor='middle',
        fill='black',
        font_size='20px',
        font_weight='bold'
    )
    title_group.add(title)
    
    # Подзаголовок с информацией о масштабе
    scale_factor = data.get('metadata', {}).get('scale_factor', 1.0)
    subtitle = dwg.text(
        f"Масштабный коэффициент: {scale_factor}",
        insert=(width/2, 50),
        text_anchor='middle',
        fill='gray',
        font_size='14px'
    )
    title_group.add(subtitle)
    
    print("  ✓ Заголовок добавлен")

# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

def visualize_polygons_junctions_aware():
    """Основная функция создания векторной визуализации с учетом junctions"""
    print("="*60)
    print("СОЗДАНИЕ ВЕКТОРНОЙ ВИЗУАЛИЗАЦИИ С УЧЕТОМ JUNCTIONS (ИСПРАВЛЕННАЯ ВЕРСИЯ V8)")
    print("="*60)
    
    # Параметры
    input_path = 'plan_floor1_objects.json'
    output_path = 'wall_polygons_junctions_aware_final_fixed8.svg'
    padding = 50  # Отступы от краев SVG
    
    # Проверяем существование входного файла
    if not os.path.exists(input_path):
        print(f"✗ Ошибка: файл не найден - {input_path}")
        return
    
    # Загружаем данные
    data = load_objects_data(input_path)
    if not data:
        print("✗ Ошибка: не удалось загрузить данные")
        return
    
    # Определяем толщину стен (минимальная толщина двери)
    wall_thickness = get_wall_thickness_from_doors(data)
    
    # Обрабатываем полигоны с учетом junctions
    print(f"\n{'='*60}")
    print("ОБРАБОТКА ПОЛИГОНОВ С УЧЕТОМ JUNCTIONS")
    print(f"{'='*60}")
    
    wall_segments, used_opening_sides, used_junctions = process_wall_polygons_with_junctions(data, wall_thickness)
    
    # Создаем T-соединения с учетом junctions
    print(f"\n{'='*60}")
    print("СОЗДАНИЕ T-СОЕДИНЕНИЙ С УЧЕТОМ JUNCTIONS")
    print(f"{'='*60}")
    
    processed_segments = create_junction_aware_t_junctions(wall_segments, parse_junctions(data), wall_thickness, data.get('openings', []))
    
    # Применяем правило "от проема до junction - только один bbox стены"
    print(f"\n{'='*60}")
    print("ПРИМЕНЕНИЕ ПРАВИЛА: ОТ ПРОЕМА ДО JUNCTION - ТОЛЬКО ОДИН BBOX СТЕНЫ")
    print(f"{'='*60}")
    
    filtered_segments = filter_segments_opening_to_junction(
        processed_segments,
        data.get('openings', []),
        parse_junctions(data)
    )
    
    print(f"  ✓ Сегментов до фильтрации: {len(processed_segments)}")
    print(f"  ✓ Сегментов после фильтрации: {len(filtered_segments)}")
    
    # ИСПРАВЛЕНО: Валидация выполнения правила после всех преобразований
    print(f"\n{'='*60}")
    print("ВАЛИДАЦИЯ ПРАВИЛА: ОТ ПРОЕМА ДО JUNCTION - ТОЛЬКО ОДИН BBOX СТЕНЫ")
    print(f"{'='*60}")
    
    validation_result = validate_opening_junction_rule(
        filtered_segments,
        data.get('openings', []),
        parse_junctions(data)
    )
    
    if validation_result['is_valid']:
        print(f"  ✓ ✓ ПРАВИЛО ВЫПОЛНЯЕТСЯ: Все {validation_result['total_pairs']} пар (проем-junction) имеют только один сегмент")
    else:
        print(f"  ✗ ✗ НАРУШЕНИЕ ПРАВИЛА: Найдено {validation_result['invalid_pairs']} нарушений из {validation_result['total_pairs']} пар")
        print(f"    Детали нарушений:")
        for i, violation in enumerate(validation_result['violations'][:5]):  # Показываем первые 5 нарушений
            print(f"      {i+1}. Проем {violation['opening_id']} + Junction {violation['junction_id']}: {violation['segment_count']} сегментов")
        if len(validation_result['violations']) > 5:
            print(f"      ... и еще {len(validation_result['violations']) - 5} нарушений")
    
    print(f"    Статистика валидации:")
    print(f"      Всего пар (проем-junction): {validation_result['total_pairs']}")
    print(f"      Корректных пар: {validation_result['valid_pairs']}")
    print(f"      Некорректных пар: {validation_result['invalid_pairs']}")
    
    print(f"\n{'='*60}")
    print(f"ИТОГО: {len(filtered_segments)} сегментов стен создано")
    print(f"{'='*60}")
    
    # Вычисляем размеры и масштаб
    max_x, max_y, inverse_scale = calculate_svg_dimensions(data, filtered_segments)
    
    # Вычисляем размеры SVG с учетом масштаба и отступов
    svg_width = int(max_x * inverse_scale + padding * 2)
    svg_height = int(max_y * inverse_scale + padding * 2)
    
    print(f"  ✓ Размеры SVG: {svg_width}x{svg_height}")
    
    # Создаем SVG документ
    dwg = create_svg_document(output_path, svg_width, svg_height, data)
    
    # Определяем стили
    styles = define_styles()
    
    # Парсим junctions для визуализации
    junctions = parse_junctions(data)
    
    # Отрисовываем объекты в правильном порядке (слои)
    # 1. Сначала отрисовываем все стеновые полигоны (самый нижний слой)
    draw_all_wall_polygons(dwg, data, inverse_scale, padding)
    
    # 2. Затем отрисовываем junctions
    draw_junctions(dwg, junctions, inverse_scale, padding)
    
    # 3. Затем отрисовываем связи к junctions
    draw_connections_to_junctions(dwg, filtered_segments, junctions, inverse_scale, padding)
    
    # 4. Затем отрисовываем колонны
    draw_pillars(dwg, data, inverse_scale, padding, styles)
    
    # 5. Затем отрисовываем окна и двери
    draw_openings_bboxes(dwg, data, inverse_scale, padding, styles)
    
    # 6. Наконец отрисовываем сегменты стен (верхний слой)
    draw_wall_bboxes(dwg, filtered_segments, inverse_scale, padding, styles)
    
    # Добавляем легенду и заголовок
    add_legend(dwg, svg_width, svg_height, styles)
    add_title(dwg, svg_width, data)
    
    # Сохраняем SVG
    dwg.save()
    print(f"\n✓ SVG файл сохранен: {output_path}")
    
    # Выводим статистику
    wall_polygons_count = len(data.get('wall_polygons', []))
    pillar_polygons_count = len(data.get('pillar_polygons', []))
    windows_count = sum(1 for o in data.get('openings', []) if o.get('type') == 'window')
    doors_count = sum(1 for o in data.get('openings', []) if o.get('type') == 'door')
    junctions_count = len(junctions)
    
    print(f"\nСтатистика объектов:")
    print(f"  Полигоны стен: {wall_polygons_count}")
    print(f"  Сегменты стен: {len(filtered_segments)}")
    print(f"  Колонны: {pillar_polygons_count}")
    print(f"  Окна: {windows_count}")
    print(f"  Двери: {doors_count}")
    print(f"  Junctions: {junctions_count}")
    print(f"  Толщина стен (минимальная толщина двери): {wall_thickness:.1f} px")
    print(f"  Использованных сторон проемов: {sum(len(sides) for sides in used_opening_sides.values())}")
    print(f"  Использованных junctions: {len(used_junctions)}")
    
    print(f"\nГотово! Векторная визуализация с учетом junctions создана: {output_path}")
    print("Откройте файл в браузере или векторном редакторе для просмотра")

if __name__ == '__main__':
    visualize_polygons_junctions_aware()
