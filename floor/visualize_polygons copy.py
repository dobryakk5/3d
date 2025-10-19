#!/usr/bin/env python3
"""
Интегрированная версия visualize_polygons_opening_based.py с анализом типов junctions

Создает SVG файл с ограничивающими прямоугольниками на основе проемов и junctions,
а также определяет и визуализирует типы junctions (L, T, X).

КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
1. Стены строятся напрямую от проемов к junctions
2. Определяются типы junctions (L, T, X)
3. Визуализация включает цветовую кодировку типов junctions
"""

import json
import svgwrite
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import math
import sys
from dataclasses import dataclass
from collections import defaultdict

# Добавляем путь к текущей директории для импорта модуля
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_junction_type_analyzer import analyze_polygon_extensions_with_thickness, is_point_in_polygon

# =============================================================================
# СТРУКТУРЫ ДАННЫХ
# =============================================================================

@dataclass
class JunctionPoint:
    x: float
    y: float
    junction_type: str
    id: int
    detected_type: str = 'unknown'  # L, T, X, или unknown
    directions: List[str] = None  # Список значимых направлений
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.directions is None:
            self.directions = []

@dataclass
class OpeningWithJunctions:
    opening_id: str
    opening_type: str  # 'door' or 'window'
    bbox: Dict[str, float]
    orientation: str  # 'horizontal' or 'vertical'
    edge_junctions: List[Tuple[str, JunctionPoint]]  # List of (edge_side, junction)

@dataclass
class WallSegmentFromOpening:
    segment_id: str
    opening_id: str
    edge_side: str  # 'left', 'right', 'top', 'bottom'
    start_junction: JunctionPoint
    end_junction: JunctionPoint
    orientation: str  # 'horizontal' or 'vertical'
    bbox: Dict[str, float]

@dataclass
class WallSegmentFromJunction:
    segment_id: str
    start_junction: JunctionPoint
    end_junction: JunctionPoint
    direction: str  # 'left', 'right', 'up', 'down'
    orientation: str  # 'horizontal' or 'vertical'
    bbox: Dict[str, float]

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
# ФУНКЦИИ ОПРЕДЕЛЕНИЯ ОРИЕНТАЦИИ И ПОИСКА JUNCTIONS
# =============================================================================

def detect_opening_orientation(opening_bbox: Dict) -> str:
    """
    Determines if an opening is horizontal or vertical based on its bbox dimensions.
    
    Args:
        opening_bbox: Dictionary with 'x', 'y', 'width', 'height' keys
    
    Returns:
        'horizontal' if width > height, 'vertical' otherwise
    """
    width = opening_bbox.get('width', 0)
    height = opening_bbox.get('height', 0)
    
    if width > height:
        return 'horizontal'
    else:
        return 'vertical'

def find_closest_junction(x: float, y: float, junctions: List[JunctionPoint], tolerance: float) -> Optional[JunctionPoint]:
    """
    Finds the closest junction to a given point within tolerance.
    """
    closest_junction = None
    min_distance = float('inf')
    
    for junction in junctions:
        distance = math.sqrt((junction.x - x)**2 + (junction.y - y)**2)
        if distance < min_distance and distance <= tolerance:
            min_distance = distance
            closest_junction = junction
    
    return closest_junction

def find_junctions_at_opening_edges(opening: Dict, junctions: List[JunctionPoint], tolerance: float) -> List[Tuple[str, JunctionPoint]]:
    """
    Finds junctions that correspond to the extreme edges of an opening.
    
    Args:
        opening: Opening dictionary with id, type, and bbox
        junctions: List of all junction points
        tolerance: Maximum distance to consider a junction as belonging to an edge
    
    Returns:
        List of tuples (edge_side, junction) for each edge of the opening
    """
    opening_id = opening.get('id', '')
    bbox = opening.get('bbox', {})
    orientation = detect_opening_orientation(bbox)
    
    x, y = bbox['x'], bbox['y']
    width, height = bbox['width'], bbox['height']
    
    edge_junctions = []
    
    if orientation == 'horizontal':
        # Left edge (x, y + height/2)
        left_center = (x, y + height / 2)
        left_junction = find_closest_junction(left_center[0], left_center[1], junctions, tolerance)
        if left_junction:
            edge_junctions.append(('left', left_junction))
        
        # Right edge (x + width, y + height/2)
        right_center = (x + width, y + height / 2)
        right_junction = find_closest_junction(right_center[0], right_center[1], junctions, tolerance)
        if right_junction:
            edge_junctions.append(('right', right_junction))
    
    else:  # vertical
        # Top edge (x + width/2, y)
        top_center = (x + width / 2, y)
        top_junction = find_closest_junction(top_center[0], top_center[1], junctions, tolerance)
        if top_junction:
            edge_junctions.append(('top', top_junction))
        
        # Bottom edge (x + width/2, y + height)
        bottom_center = (x + width / 2, y + height)
        bottom_junction = find_closest_junction(bottom_center[0], bottom_center[1], junctions, tolerance)
        if bottom_junction:
            edge_junctions.append(('bottom', bottom_junction))
    
    return edge_junctions

def find_next_junction_in_direction(start_junction: JunctionPoint, direction: str, junctions: List[JunctionPoint], tolerance: float) -> Optional[JunctionPoint]:
    """
    Finds the next junction in a given direction from a starting junction.
    
    Args:
        start_junction: The starting junction point
        direction: Direction to search ('left', 'right', 'up', 'down')
        junctions: List of all junction points
        tolerance: Tolerance for alignment checking
    
    Returns:
        The next junction in the specified direction, or None if not found
    """
    closest_junction = None
    min_distance = float('inf')
    
    for junction in junctions:
        # Skip the starting junction
        if junction.id == start_junction.id:
            continue
        
        # Check if junction is in the correct direction and aligned
        is_aligned = False
        distance = 0
        
        if direction == 'left':
            # Junction should be to the left and roughly at same Y
            is_aligned = (junction.x < start_junction.x and 
                         abs(junction.y - start_junction.y) <= tolerance)
            distance = start_junction.x - junction.x
            
        elif direction == 'right':
            # Junction should be to the right and roughly at same Y
            is_aligned = (junction.x > start_junction.x and 
                         abs(junction.y - start_junction.y) <= tolerance)
            distance = junction.x - start_junction.x
            
        elif direction == 'up':
            # Junction should be above and roughly at same X
            is_aligned = (junction.y < start_junction.y and 
                         abs(junction.x - start_junction.x) <= tolerance)
            distance = start_junction.y - junction.y
            
        elif direction == 'down':
            # Junction should be below and roughly at same X
            is_aligned = (junction.y > start_junction.y and 
                         abs(junction.x - start_junction.x) <= tolerance)
            distance = junction.y - start_junction.y
        
        if is_aligned and distance < min_distance:
            min_distance = distance
            closest_junction = junction
    
    return closest_junction

def create_bbox_from_junctions(start_junction: JunctionPoint, 
                             end_junction: JunctionPoint, 
                             orientation: str, 
                             wall_thickness: float) -> Dict[str, float]:
    """
    Creates a bbox from two junction points.
    """
    if orientation == 'horizontal':
        x = min(start_junction.x, end_junction.x)
        y = start_junction.y - wall_thickness / 2
        width = abs(end_junction.x - start_junction.x)
        height = wall_thickness
    else:  # vertical
        x = start_junction.x - wall_thickness / 2
        y = min(start_junction.y, end_junction.y)
        width = wall_thickness
        height = abs(end_junction.y - start_junction.y)
    
    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'orientation': orientation
    }

# =============================================================================
# ФУНКЦИИ ПОСТРОЕНИЯ СТЕНОВЫХ СЕГМЕНТОВ
# =============================================================================

def construct_wall_segment_from_opening(opening_with_junction: OpeningWithJunctions,
                                      junctions: List[JunctionPoint],
                                      wall_thickness: float) -> List[WallSegmentFromOpening]:
    """
    Constructs wall segments from each edge junction of an opening to the next junction.
    Строит сегменты стен, которые точно стыкуются с углами проемов.
    
    Args:
        opening_with_junction: Opening with its edge junctions
        junctions: List of all junction points
        wall_thickness: Thickness of walls
    
    Returns:
        List of wall segments constructed from the opening
    """
    wall_segments = []
    opening_id = opening_with_junction.opening_id
    orientation = opening_with_junction.orientation
    bbox = opening_with_junction.bbox
    
    for edge_side, start_junction in opening_with_junction.edge_junctions:
        # Determine the direction to extend the wall
        if orientation == 'horizontal':
            if edge_side == 'left':
                direction = 'left'
                # Для левой стороны горизонтального проема, вычисляем смещение
                # от центра края до левого верхнего угла проема
                offset_x = start_junction.x - bbox['x']
                offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
            else:  # right
                direction = 'right'
                # Для правой стороны горизонтального проема, вычисляем смещение
                # от центра края до правого верхнего угла проема
                offset_x = start_junction.x - (bbox['x'] + bbox['width'])
                offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
        else:  # vertical
            if edge_side == 'top':
                direction = 'up'
                # Для верхней стороны вертикального проема, вычисляем смещение
                # от центра края до левого верхнего угла проема
                offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                offset_y = start_junction.y - bbox['y']
            else:  # bottom
                direction = 'down'
                # Для нижней стороны вертикального проема, вычисляем смещение
                # от центра края до левого нижнего угла проема
                offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                offset_y = start_junction.y - (bbox['y'] + bbox['height'])
        
        # Find the next junction in the direction
        end_junction = find_next_junction_in_direction(start_junction, direction, junctions, wall_thickness / 2.0)
        
        if end_junction:
            # Create wall segment from start to end junction
            segment_id = f"wall_{opening_id}_{edge_side}_{start_junction.id}_to_{end_junction.id}"
            
            # Создаем bbox, но с учетом смещения для точной стыковки с углами проема
            if orientation == 'horizontal':
                x = min(start_junction.x - offset_x, end_junction.x)
                y = (start_junction.y - offset_y) - wall_thickness / 2
                width = abs(end_junction.x - (start_junction.x - offset_x))
                height = wall_thickness
            else:  # vertical
                x = (start_junction.x - offset_x) - wall_thickness / 2
                y = min(start_junction.y - offset_y, end_junction.y)
                width = wall_thickness
                height = abs(end_junction.y - (start_junction.y - offset_y))
            
            bbox_result = {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'orientation': orientation
            }
            
            # Создаем скорректированный junction для начальной точки
            adjusted_start_junction = JunctionPoint(
                x=start_junction.x - offset_x,
                y=start_junction.y - offset_y,
                junction_type=start_junction.junction_type,
                id=start_junction.id
            )
            
            wall_segment = WallSegmentFromOpening(
                segment_id=segment_id,
                opening_id=opening_id,
                edge_side=edge_side,
                start_junction=adjusted_start_junction,
                end_junction=end_junction,
                orientation=orientation,
                bbox=bbox_result
            )
            wall_segments.append(wall_segment)
        else:
            # Handle case where no junction is found - extend to polygon edge or next opening
            # This would require additional logic to find the boundary
            print(f"    Предупреждение: не найден junction для проема {opening_id}, сторона {edge_side}")
    
    return wall_segments

def process_openings_with_junctions(data: Dict, wall_thickness: float) -> List[WallSegmentFromOpening]:
    """
    Main function that processes all openings and constructs wall segments.
    
    Args:
        data: Dictionary containing openings and junctions
        wall_thickness: Thickness of walls
    
    Returns:
        List of all wall segments constructed from openings
    """
    openings = data.get('openings', [])
    junctions = parse_junctions(data)
    
    all_wall_segments = []
    
    print(f"Обработка {len(openings)} проемов:")
    
    for opening in openings:
        opening_id = opening.get('id', '')
        opening_type = opening.get('type', '')
        bbox = opening.get('bbox', {})
        
        if not bbox:
            continue
        
        # Determine opening orientation
        orientation = detect_opening_orientation(bbox)
        
        # Find junctions at opening edges
        edge_junctions = find_junctions_at_opening_edges(opening, junctions, wall_thickness / 2.0)
        
        if not edge_junctions:
            print(f"  Предупреждение: не найдены junctions для проема {opening_id}")
            continue
        
        # Create OpeningWithJunctions object
        opening_with_junction = OpeningWithJunctions(
            opening_id=opening_id,
            opening_type=opening_type,
            bbox=bbox,
            orientation=orientation,
            edge_junctions=edge_junctions
        )
        
        # Construct wall segments from this opening
        wall_segments = construct_wall_segment_from_opening(
            opening_with_junction, junctions, wall_thickness
        )
        
        all_wall_segments.extend(wall_segments)
        print(f"  ✓ Проем {opening_id} ({orientation}): {len(wall_segments)} сегментов")
    
    print(f"\nВсего создано {len(all_wall_segments)} сегментов стен из {len(openings)} проемов")
    return all_wall_segments

def find_junctions_not_related_to_openings(junctions: List[JunctionPoint], wall_segments: List[WallSegmentFromOpening], junction_wall_segments: List[WallSegmentFromJunction] = None) -> List[JunctionPoint]:
    """
    Находит все junctions, которые не связаны с проемами или имеют недостаточное количество сегментов стен
    
    Args:
        junctions: Список всех junction points
        wall_segments: Список сегментов стен, созданных из проемов
        junction_wall_segments: Список сегментов стен, созданных из junctions
    
    Returns:
        Список junctions, не связанных с проемами или имеющих недостаточное количество сегментов стен
    """
    if junction_wall_segments is None:
        junction_wall_segments = []
    
    # Создаем множество всех junctions, связанных с проемами или уже построенными сегментами
    related_junctions = set()
    for segment in wall_segments:
        related_junctions.add(segment.start_junction.id)
        related_junctions.add(segment.end_junction.id)
    
    for segment in junction_wall_segments:
        related_junctions.add(segment.start_junction.id)
        related_junctions.add(segment.end_junction.id)
    
    # Находим junctions, не связанные с проемами или имеющие недостаточное количество сегментов
    non_opening_junctions = []
    for junction in junctions:
        # Если junction не связан с проемами или уже построенными сегментами
        if junction.id not in related_junctions:
            non_opening_junctions.append(junction)
            continue
        
        # Если junction связан, но имеет меньше сегментов, чем требуется
        required_directions = junction.directions if junction.directions else []
        if len(required_directions) > 0:
            # Считаем количество сегментов, подключенных к этому junction
            connected_segments = 0
            for segment in wall_segments:
                if segment.start_junction.id == junction.id or segment.end_junction.id == junction.id:
                    connected_segments += 1
            
            for segment in junction_wall_segments:
                if segment.start_junction.id == junction.id or segment.end_junction.id == junction.id:
                    connected_segments += 1
            
            # Если количество подключенных сегментов меньше, чем требуемых направлений
            if connected_segments < len(required_directions):
                non_opening_junctions.append(junction)
    
    print(f"  ✓ Найдено {len(non_opening_junctions)} junctions для обработки")
    return non_opening_junctions

def get_existing_directions_for_junction(junction: JunctionPoint, wall_segments: List[WallSegmentFromOpening], junction_wall_segments: List[WallSegmentFromJunction], wall_thickness: float) -> List[str]:
    """
    Определяет существующие направления стен для данного junction
    
    Args:
        junction: Junction point для анализа
        wall_segments: Список всех сегментов стен
        wall_thickness: Толщина стен
    
    Returns:
        Список существующих направлений (['left', 'right', 'up', 'down'])
    """
    existing_directions = []
    jx, jy = junction.x, junction.y
    tolerance = wall_thickness / 2.0
    
    # Проверяем сегменты стен из проемов
    for segment in wall_segments:
        # Проверяем, является ли junction начальной или конечной точкой сегмента
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        # Проверяем начальную точку
        if start_dist <= tolerance:
            if segment.orientation == 'horizontal':
                if segment.end_junction.x > segment.start_junction.x:
                    existing_directions.append('right')
                else:
                    existing_directions.append('left')
            else:  # vertical
                if segment.end_junction.y > segment.start_junction.y:
                    existing_directions.append('down')
                else:
                    existing_directions.append('up')
        
        # Проверяем конечную точку
        if end_dist <= tolerance:
            if segment.orientation == 'horizontal':
                if segment.end_junction.x > segment.start_junction.x:
                    existing_directions.append('left')
                else:
                    existing_directions.append('right')
            else:  # vertical
                if segment.end_junction.y > segment.start_junction.y:
                    existing_directions.append('up')
                else:
                    existing_directions.append('down')
    
    # Проверяем сегменты стен из junctions
    for segment in junction_wall_segments:
        # Проверяем, является ли junction начальной или конечной точкой сегмента
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        # Проверяем начальную точку
        if start_dist <= tolerance:
            if segment.orientation == 'horizontal':
                if segment.end_junction.x > segment.start_junction.x:
                    existing_directions.append('right')
                else:
                    existing_directions.append('left')
            else:  # vertical
                if segment.end_junction.y > segment.start_junction.y:
                    existing_directions.append('down')
                else:
                    existing_directions.append('up')
        
        # Проверяем конечную точку
        if end_dist <= tolerance:
            if segment.orientation == 'horizontal':
                if segment.end_junction.x > segment.start_junction.x:
                    existing_directions.append('left')
                else:
                    existing_directions.append('right')
            else:  # vertical
                if segment.end_junction.y > segment.start_junction.y:
                    existing_directions.append('up')
                else:
                    existing_directions.append('down')
    
    # Удаляем дубликаты
    existing_directions = list(set(existing_directions))
    return existing_directions

def build_missing_wall_segments_for_junctions(junctions: List[JunctionPoint], wall_segments: List[WallSegmentFromOpening], junction_wall_segments: List[WallSegmentFromJunction], all_junctions: List[JunctionPoint], wall_thickness: float) -> List[WallSegmentFromJunction]:
    """
    Достраивает недостающие сегменты стен для junctions не связанных с проемами
    
    Args:
        junctions: Список junctions не связанных с проемами
        wall_segments: Существующие сегменты стен из проемов
        junction_wall_segments: Существующие сегменты стен из junctions
        all_junctions: Список всех junctions
        wall_thickness: Толщина стен
    
    Returns:
        Список новых сегментов стен
    """
    new_segments = []
    
    for junction in junctions:
        # Получаем существующие направления для этого junction
        existing_directions = get_existing_directions_for_junction(junction, wall_segments, junction_wall_segments, wall_thickness)
        
        # Получаем все возможные направления из анализа junction
        required_directions = junction.directions if junction.directions else []
        
        # Находим недостающие направления
        missing_directions = []
        for direction in required_directions:
            if direction not in existing_directions:
                missing_directions.append(direction)
        
        print(f"  Junction {junction.id}: существующие направления={existing_directions}, требуемые={required_directions}, недостающие={missing_directions}")
        
        # Для каждого недостающего направления строим сегмент стены
        for direction in missing_directions:
            # Ищем следующий junction в этом направлении
            next_junction = find_next_junction_in_direction(junction, direction, all_junctions, wall_thickness / 2.0)
            
            if next_junction:
                # Определяем ориентацию сегмента
                orientation = 'horizontal' if direction in ['left', 'right'] else 'vertical'
                
                # Создаем bbox для сегмента
                bbox = create_bbox_from_junctions(junction, next_junction, orientation, wall_thickness)
                
                # Создаем новый сегмент
                segment_id = f"wall_junction_{junction.id}_to_{next_junction.id}_{direction}"
                new_segment = WallSegmentFromJunction(
                    segment_id=segment_id,
                    start_junction=junction,
                    end_junction=next_junction,
                    direction=direction,
                    orientation=orientation,
                    bbox=bbox
                )
                new_segments.append(new_segment)
                print(f"    ✓ Создан сегмент от J{junction.id} до J{next_junction.id} в направлении {direction}")
            else:
                print(f"    ✗ Не найден следующий junction для J{junction.id} в направлении {direction}")
    
    print(f"\n  ✓ Создано {len(new_segments)} новых сегментов стен для junctions не связанных с проемами")
    return new_segments

# =============================================================================
# ФУНКЦИИ АНАЛИЗА ТИПОВ JUNCTIONS
# =============================================================================

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

def analyze_wall_orientation_from_segment(wall_segment: WallSegmentFromOpening) -> str:
    """Анализирует ориентацию сегмента стены"""
    return wall_segment.orientation

def find_walls_connected_to_junction(junction: JunctionPoint, wall_segments: List[WallSegmentFromOpening], tolerance: float) -> List[WallSegmentFromOpening]:
    """Находит все сегменты стен, подключенные к данному junction"""
    connected_segments = []
    jx, jy = junction.x, junction.y
    
    for segment in wall_segments:
        # Проверяем, находится ли junction близко к началу или концу сегмента
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        if start_dist <= tolerance or end_dist <= tolerance:
            connected_segments.append(segment)
    
    return connected_segments

def determine_junction_type(connected_segments: List[WallSegmentFromOpening], segment_orientations: List[str]) -> str:
    """Определяет тип junction на основе подключенных сегментов"""
    connection_count = len(connected_segments)
    
    if connection_count < 2:
        return 'unknown'
    
    # Считаем количество горизонтальных и вертикальных сегментов
    horizontal_count = segment_orientations.count('horizontal')
    vertical_count = segment_orientations.count('vertical')
    
    # Определяем тип junction
    if connection_count == 2:
        # L-junction: 2 сегмента, обычно разной ориентации
        if horizontal_count == 1 and vertical_count == 1:
            return 'L'
        elif horizontal_count == 2 or vertical_count == 2:
            # Это может быть продолжение прямой стены
            return 'straight'
        else:
            return 'unknown'
    
    elif connection_count == 3:
        # T-junction: 3 сегмента, 2 одной ориентации, 1 другой
        if (horizontal_count == 2 and vertical_count == 1) or \
           (horizontal_count == 1 and vertical_count == 2):
            return 'T'
        else:
            return 'unknown'
    
    elif connection_count == 4:
        # X-junction: 4 сегмента, по 2 каждой ориентации
        if horizontal_count == 2 and vertical_count == 2:
            return 'X'
        else:
            return 'unknown'
    
    else:
        return 'unknown'

def analyze_junction_types_with_thickness(junctions: List[JunctionPoint], data: Dict, wall_thickness: float) -> List[JunctionPoint]:
    """
    Анализирует типы всех junctions с использованием улучшенной логики с учетом толщины стены
    
    Args:
        junctions: Список junction points
        data: Данные плана с wall_polygons
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Обновленный список junctions с определенными типами
    """
    print(f"Анализ типов {len(junctions)} junctions с улучшенной логикой...")
    
    wall_polygons = data.get('wall_polygons', [])
    type_counts = {}
    
    for junction in junctions:
        # Находим полигон, содержащий junction
        containing_polygon = None
        for wall in wall_polygons:
            vertices = wall.get('vertices', [])
            if vertices and is_point_in_polygon(junction.x, junction.y, vertices):
                containing_polygon = wall
                break
        
        if containing_polygon:
            # Определяем тип с учетом толщины стены
            analysis = analyze_polygon_extensions_with_thickness(
                {'x': junction.x, 'y': junction.y},
                containing_polygon['vertices'],
                wall_thickness
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
            
            junction.detected_type = junction_type
            junction.directions = significant_directions
            
            # Считаем статистику
            type_counts[junction_type] = type_counts.get(junction_type, 0) + 1
            
            print(f"  Junction {junction.id} ({junction.x}, {junction.y}): тип={junction_type}, направления={', '.join(significant_directions)}")
        else:
            # Junction не внутри полигона
            junction.detected_type = 'unknown'
            type_counts['unknown'] = type_counts.get('unknown', 0) + 1
            print(f"  Junction {junction.id} ({junction.x}, {junction.y}): тип=unknown (не внутри полигона)")
    
    print(f"\nСтатистика типов junctions:")
    for jtype, count in type_counts.items():
        print(f"  {jtype}: {count}")
    
    return junctions

def analyze_junction_types(wall_segments: List[WallSegmentFromOpening], junctions: List[JunctionPoint]) -> List[JunctionPoint]:
    """
    Анализирует типы всех junctions на основе подключенных сегментов стен
    
    Args:
        wall_segments: Список сегментов стен
        junctions: Список junction points
    
    Returns:
        Обновленный список junctions с определенными типами
    """
    print(f"Анализ типов {len(junctions)} junctions...")
    
    type_counts = {}
    
    for junction in junctions:
        # Находим подключенные сегменты стен
        # Используем половину толщины стены как допуск для поиска соединений
        connected_segments = find_walls_connected_to_junction(junction, wall_segments, wall_thickness / 2.0)
        
        # Анализируем ориентацию сегментов
        segment_orientations = []
        for segment in connected_segments:
            orientation = analyze_wall_orientation_from_segment(segment)
            segment_orientations.append(orientation)
        
        # Определяем тип junction
        junction_type = determine_junction_type(connected_segments, segment_orientations)
        junction.detected_type = junction_type
        
        # Считаем статистику
        type_counts[junction_type] = type_counts.get(junction_type, 0) + 1
        
        print(f"  Junction {junction.id} ({junction.x}, {junction.y}): тип={junction_type}, подключено сегментов={len(connected_segments)}")
    
    print(f"\nСтатистика типов junctions:")
    for jtype, count in type_counts.items():
        print(f"  {jtype}: {count}")
    
    return junctions

# =============================================================================
# ФУНКЦИИ ОБРАБОТКИ L-JUNCTIONS И РАСШИРЕНИЙ
# =============================================================================

def find_l_junctions(junctions: List[JunctionPoint]) -> List[JunctionPoint]:
    """
    Extracts all L-junctions from the list of processed junctions
    
    Args:
        junctions: List of all junction points with detected types
        
    Returns:
        List of L-junction points
    """
    l_junctions = [j for j in junctions if j.detected_type == 'L']
    print(f"  ✓ Найдено {len(l_junctions)} L-junctions")
    return l_junctions

def find_wall_segments_at_l_junction(l_junction: JunctionPoint,
                                   wall_segments: List[WallSegmentFromOpening],
                                   junction_wall_segments: List[WallSegmentFromJunction],
                                   tolerance: float) -> Tuple[List, List]:
    """
    Finds wall segments connected to a specific L-junction
    
    Args:
        l_junction: The L-junction point
        wall_segments: List of wall segments from openings
        junction_wall_segments: List of wall segments from junctions
        tolerance: Distance tolerance for connection checking
        
    Returns:
        Tuple of (opening_segments, junction_segments) connected to the L-junction
    """
    opening_segments = []
    junction_segments = []
    jx, jy = l_junction.x, l_junction.y
    
    # Check opening-based wall segments
    for segment in wall_segments:
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        if start_dist <= tolerance or end_dist <= tolerance:
            opening_segments.append(segment)
    
    # Check junction-based wall segments
    for segment in junction_wall_segments:
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        if start_dist <= tolerance or end_dist <= tolerance:
            junction_segments.append(segment)
    
    return opening_segments, junction_segments

def extend_segment_to_polygon_edge(segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                 l_junction: JunctionPoint,
                                 wall_polygons: List[Dict],
                                 wall_thickness: float) -> Dict[str, float]:
    """
    Extends a wall segment from an L-junction to the edge of the containing wall polygon
    
    Args:
        segment: The wall segment to extend
        l_junction: The L-junction point
        wall_polygons: List of wall polygons
        wall_thickness: Wall thickness for calculations
        
    Returns:
        Extended bbox dictionary
    """
    # Find the polygon containing the L-junction
    containing_polygon = None
    for wall in wall_polygons:
        vertices = wall.get('vertices', [])
        if vertices and is_point_in_polygon(l_junction.x, l_junction.y, vertices):
            containing_polygon = wall
            break
    
    if not containing_polygon:
        return segment.bbox  # Return original if no polygon found
    
    # Get the direction of extension based on junction directions
    directions = l_junction.directions
    if not directions:
        return segment.bbox
    
    # Determine which direction to extend (choose the first available)
    extend_direction = directions[0]
    
    # Get the polygon edge intersection in that direction
    analysis = analyze_polygon_extensions_with_thickness(
        {'x': l_junction.x, 'y': l_junction.y},
        containing_polygon['vertices'],
        wall_thickness
    )
    
    intersections = analysis['intersections']
    
    # Create extended bbox based on direction
    if extend_direction in intersections and intersections[extend_direction] is not None:
        if segment.orientation == 'horizontal':
            if extend_direction == 'left':
                # Extend to the left
                new_x = intersections[extend_direction]
                new_width = segment.bbox['x'] + segment.bbox['width'] - new_x
                return {
                    'x': new_x,
                    'y': segment.bbox['y'],
                    'width': new_width,
                    'height': segment.bbox['height'],
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
            elif extend_direction == 'right':
                # Extend to the right
                new_width = intersections[extend_direction] - segment.bbox['x']
                return {
                    'x': segment.bbox['x'],
                    'y': segment.bbox['y'],
                    'width': new_width,
                    'height': segment.bbox['height'],
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
        else:  # vertical
            if extend_direction == 'up':
                # Extend upward
                new_y = intersections[extend_direction]
                new_height = segment.bbox['y'] + segment.bbox['height'] - new_y
                return {
                    'x': segment.bbox['x'],
                    'y': new_y,
                    'width': segment.bbox['width'],
                    'height': new_height,
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
            elif extend_direction == 'down':
                # Extend downward
                new_height = intersections[extend_direction] - segment.bbox['y']
                return {
                    'x': segment.bbox['x'],
                    'y': segment.bbox['y'],
                    'width': segment.bbox['width'],
                    'height': new_height,
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
    
    return segment.bbox  # Return original if extension not possible

def extend_segment_to_perpendicular_x(segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                    perpendicular_segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                    l_junction: JunctionPoint) -> Dict[str, float]:
    """
    Extends a wall segment to reach the X coordinate of a perpendicular segment
    
    Args:
        segment: The wall segment to extend
        perpendicular_segment: The perpendicular wall segment
        l_junction: The L-junction point
        
    Returns:
        Extended bbox dictionary
    """
    # Determine which segment is horizontal and which is vertical
    if segment.orientation == 'horizontal' and perpendicular_segment.orientation == 'vertical':
        # Extend horizontal segment to reach vertical segment's X
        target_x = perpendicular_segment.bbox['x']
        
        # Determine if we need to extend left or right
        if target_x < segment.bbox['x']:
            # Extend to the left
            new_x = target_x
            new_width = segment.bbox['x'] + segment.bbox['width'] - target_x
            extension_direction = 'left'
        else:
            # Extend to the right
            new_width = target_x + perpendicular_segment.bbox['width'] - segment.bbox['x']
            new_x = segment.bbox['x']
            extension_direction = 'right'
        
        return {
            'x': new_x,
            'y': segment.bbox['y'],
            'width': new_width,
            'height': segment.bbox['height'],
            'orientation': segment.orientation,
            'extended': True,
            'extension_direction': extension_direction,
            'extension_type': 'perpendicular_alignment'
        }
    
    elif segment.orientation == 'vertical' and perpendicular_segment.orientation == 'horizontal':
        # Extend vertical segment to reach horizontal segment's Y
        target_y = perpendicular_segment.bbox['y']
        
        # Determine if we need to extend up or down
        if target_y < segment.bbox['y']:
            # Extend upward
            new_y = target_y
            new_height = segment.bbox['y'] + segment.bbox['height'] - target_y
            extension_direction = 'up'
        else:
            # Extend downward
            new_height = target_y + perpendicular_segment.bbox['height'] - segment.bbox['y']
            new_y = segment.bbox['y']
            extension_direction = 'down'
        
        return {
            'x': segment.bbox['x'],
            'y': new_y,
            'width': segment.bbox['width'],
            'height': new_height,
            'orientation': segment.orientation,
            'extended': True,
            'extension_direction': extension_direction,
            'extension_type': 'perpendicular_alignment'
        }
    
    return segment.bbox  # Return original if segments are not perpendicular

def process_l_junction_extensions(junctions: List[JunctionPoint],
                                wall_segments: List[WallSegmentFromOpening],
                                junction_wall_segments: List[WallSegmentFromJunction],
                                wall_polygons: List[Dict],
                                wall_thickness: float) -> List[Dict[str, float]]:
    """
    Processes all L-junctions and creates extended wall segments
    
    Args:
        junctions: List of all junction points (with detected_type already set)
        wall_segments: List of wall segments from openings
        junction_wall_segments: List of wall segments from junctions
        wall_polygons: List of wall polygons
        wall_thickness: Wall thickness for calculations
        
    Returns:
        List of extended bbox dictionaries for visualization
    """
    extended_segments = []
    
    # Find all L-junctions using existing detection
    l_junctions = find_l_junctions(junctions)
    
    if not l_junctions:
        print("  ✓ L-junctions не найдены")
        return extended_segments
    
    # Process each L-junction
    for l_junction in l_junctions:
        print(f"  Обработка L-junction {l_junction.id} ({l_junction.x}, {l_junction.y})")
        
        # Find connected wall segments
        opening_segments, junction_segments = find_wall_segments_at_l_junction(
            l_junction, wall_segments, junction_wall_segments, wall_thickness / 2.0
        )
        
        all_connected_segments = opening_segments + junction_segments
        
        if len(all_connected_segments) < 2:
            print(f"    ✗ Недостаточно сегментов для L-junction {l_junction.id}")
            continue
        
        # Select the first segment for extension to polygon edge
        selected_segment = all_connected_segments[0]
        print(f"    ✓ Выбран сегмент для расширения до края полигона: {selected_segment.segment_id}")
        
        # Extend to polygon edge
        extended_to_edge = extend_segment_to_polygon_edge(
            selected_segment, l_junction, wall_polygons, wall_thickness
        )
        
        if extended_to_edge.get('extended', False):
            extended_segments.append(extended_to_edge)
            print(f"    ✓ Сегмент расширен до края полигона в направлении {extended_to_edge.get('extension_direction')}")
        
        # If we have at least 2 segments, extend one to align with the perpendicular
        if len(all_connected_segments) >= 2:
            perpendicular_segment = all_connected_segments[1]
            print(f"    ✓ Выбран перпендикулярный сегмент: {perpendicular_segment.segment_id}")
            
            # Extend to align with perpendicular segment
            extended_to_perpendicular = extend_segment_to_perpendicular_x(
                selected_segment, perpendicular_segment, l_junction
            )
            
            if extended_to_perpendicular.get('extended', False):
                extended_segments.append(extended_to_perpendicular)
                print(f"    ✓ Сегмент расширен для выравнивания с перпендикулярным сегментом")
    
    print(f"  ✓ Создано {len(extended_segments)} расширенных сегментов для L-junctions")
    return extended_segments

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

def get_junction_style(junction_type: str) -> Dict[str, Any]:
    """Возвращает стиль для junction в зависимости от его типа"""
    if junction_type == 'L':
        return {
            'stroke': '#FF8C00',  # Оранжевый
            'stroke_width': 2,
            'fill': '#FFA500',    # Ярко-оранжевый
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        }
    elif junction_type == 'T':
        return {
            'stroke': '#9400D3',  # Фиолетовый
            'stroke_width': 2,
            'fill': '#8A2BE2',    # Синевато-фиолетовый
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        }
    elif junction_type == 'X':
        return {
            'stroke': '#DC143C',  # Малиновый
            'stroke_width': 2,
            'fill': '#FF1493',    # Глубокий розовый
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        }
    elif junction_type == 'straight':
        return {
            'stroke': '#32CD32',  # Лаймовый
            'stroke_width': 2,
            'fill': '#00FF00',    # Зеленый
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
        }
    else:  # unknown
        return {
            'stroke': '#708090',  # Серый
            'stroke_width': 2,
            'fill': '#A9A9A9',    # Темно-серый
            'stroke_linecap': 'round',
            'stroke_linejoin': 'round'
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

def draw_junctions_with_types(dwg: svgwrite.Drawing, junctions: List[JunctionPoint],
                             inverse_scale: float, padding: float) -> None:
    """Отображает junctions с цветовой кодировкой по типу"""
    junctions_group = dwg.add(dwg.g(id='junctions'))
    
    print(f"  ✓ Отрисовка {len(junctions)} junctions с типами")
    
    for idx, junction in enumerate(junctions):
        # Преобразуем координаты
        svg_x, svg_y = transform_coordinates(junction.x, junction.y, inverse_scale, padding)
        
        # Получаем стиль для типа junction
        junction_style = get_junction_style(junction.detected_type)
        
        # Создаем кружок для junction
        circle = dwg.circle(center=(svg_x, svg_y), r=5, **junction_style)
        junctions_group.add(circle)
        
        # Добавляем номер и тип
        text = dwg.text(
            f"J{idx+1}",
            insert=(svg_x + 10, svg_y - 5),
            text_anchor='start',
            fill='black',
            font_size='8px',
            font_weight='bold'
        )
        junctions_group.add(text)
        
        # Добавляем тип и направления (меньшим шрифтом)
        if junction.directions:
            # Форматируем направления в сокращенном виде
            direction_map = {'left': 'L', 'right': 'R', 'up': 'U', 'down': 'D'}
            direction_abbr = '-'.join([direction_map.get(d, d[0].upper()) for d in junction.directions])
            type_text = dwg.text(
                f"{junction.detected_type} {direction_abbr}",
                insert=(svg_x + 10, svg_y + 8),
                text_anchor='start',
                fill='black',
                font_size='6px'
            )
        else:
            type_text = dwg.text(
                f"{junction.detected_type}",
                insert=(svg_x + 10, svg_y + 8),
                text_anchor='start',
                fill='black',
                font_size='6px'
            )
        junctions_group.add(type_text)

def draw_opening_based_wall_bboxes(dwg: svgwrite.Drawing, 
                                  wall_segments: List[WallSegmentFromOpening], 
                                  inverse_scale: float, 
                                  padding: float, 
                                  styles: Dict) -> None:
    """Отрисовывает bbox стен для opening-based approach"""
    walls_group = dwg.add(dwg.g(id='opening_based_walls'))
    
    print(f"  ✓ Отрисовка {len(wall_segments)} сегментов стен (opening-based)")
    
    for idx, segment in enumerate(wall_segments):
        # Преобразуем координаты
        x, y = transform_coordinates(segment.bbox['x'], segment.bbox['y'], inverse_scale, padding)
        width = segment.bbox['width'] * inverse_scale
        height = segment.bbox['height'] * inverse_scale
        
        # Создаем прямоугольник
        rect = dwg.rect(insert=(x, y), size=(width, height), **styles['wall'])
        walls_group.add(rect)
        
        # Добавляем номер сегмента с информацией о проеме и ориентации
        orientation_label = 'h' if segment.orientation == 'horizontal' else 'v'
        opening_short_id = segment.opening_id.split('_')[-1] if '_' in segment.opening_id else segment.opening_id
        
        text = dwg.text(
            f"W{idx+1}_{opening_short_id}_{segment.edge_side[0]}{orientation_label}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='red',
            font_size='10px',
            font_weight='bold'
        )
        walls_group.add(text)

def draw_junction_based_wall_bboxes(dwg: svgwrite.Drawing,
                                   wall_segments: List[WallSegmentFromJunction],
                                   inverse_scale: float,
                                   padding: float,
                                   styles: Dict) -> None:
    """Отрисовывает bbox стен для junction-based approach"""
    walls_group = dwg.add(dwg.g(id='junction_based_walls'))
    
    print(f"  ✓ Отрисовка {len(wall_segments)} сегментов стен (junction-based)")
    
    # Создаем стиль для junction-based стен (другой цвет)
    junction_wall_style = {
        'stroke': '#FF6347',  # Томатово-красный
        'stroke_width': 2,
        'fill': 'none',
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round'
    }
    
    for idx, segment in enumerate(wall_segments):
        # Преобразуем координаты
        x, y = transform_coordinates(segment.bbox['x'], segment.bbox['y'], inverse_scale, padding)
        width = segment.bbox['width'] * inverse_scale
        height = segment.bbox['height'] * inverse_scale
        
        # Создаем прямоугольник
        rect = dwg.rect(insert=(x, y), size=(width, height), **junction_wall_style)
        walls_group.add(rect)
        
        # Добавляем номер сегмента с информацией о junctions и направлении
        orientation_label = 'h' if segment.orientation == 'horizontal' else 'v'
        direction_label = segment.direction[0].upper()  # L, R, U, D
        
        text = dwg.text(
            f"J{idx+1}_{segment.start_junction.id}->{segment.end_junction.id}_{direction_label}{orientation_label}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='#FF6347',
            font_size='8px',
            font_weight='bold'
        )
        walls_group.add(text)

def draw_extended_segments(dwg: svgwrite.Drawing,
                          extended_segments: List[Dict[str, float]],
                          inverse_scale: float,
                          padding: float) -> None:
    """
    Draws extended wall segments with special styling
    
    Args:
        dwg: SVG drawing object
        extended_segments: List of extended bbox dictionaries
        inverse_scale: Scale factor for coordinate transformation
        padding: Padding for SVG coordinates
    """
    extended_group = dwg.add(dwg.g(id='extended_segments'))
    
    # Style for extended segments
    extended_style = {
        'stroke': '#FF69B4',  # Hot pink
        'stroke_width': 3,
        'fill': 'none',
        'stroke_dasharray': '10,5',  # Dashed line
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round',
        'opacity': 0.8
    }
    
    print(f"  ✓ Отрисовка {len(extended_segments)} расширенных сегментов")
    
    for idx, segment in enumerate(extended_segments):
        # Transform coordinates
        x, y = transform_coordinates(segment['x'], segment['y'], inverse_scale, padding)
        width = segment['width'] * inverse_scale
        height = segment['height'] * inverse_scale
        
        # Create rectangle
        rect = dwg.rect(insert=(x, y), size=(width, height), **extended_style)
        extended_group.add(rect)
        
        # Add label with extension info
        extension_type = segment.get('extension_type', 'polygon_edge')
        direction = segment.get('extension_direction', 'unknown')
        orientation_label = 'h' if segment['orientation'] == 'horizontal' else 'v'
        
        text = dwg.text(
            f"E{idx+1}_{direction[0].upper()}{orientation_label}_{extension_type}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='#FF69B4',
            font_size='8px',
            font_weight='bold'
        )
        extended_group.add(text)

def draw_pillars(dwg: svgwrite.Drawing, data: Dict, inverse_scale: float, padding: float, styles: Dict, wall_thickness: float = None) -> None:
    """Отрисовывает колонны как квадраты со стороной равной толщине стены"""
    pillars_group = dwg.add(dwg.g(id='pillars'))
    
    pillar_polygons = data.get('pillar_polygons', [])
    print(f"  ✓ Отрисовка {len(pillar_polygons)} колонн как квадратов")
    
    # Проверяем, что толщина стены указана
    if wall_thickness is None:
        raise ValueError("Толщина стены должна быть передана в функцию draw_pillars")
    
    print(f"  ✓ Используем толщину стены: {wall_thickness} px")
    
    # Размер квадрата с учетом масштаба
    square_size = wall_thickness * inverse_scale
    print(f"  ✓ Размер квадрата колонны: {square_size:.2f} px")
    
    for idx, polygon in enumerate(pillar_polygons):
        vertices = polygon.get('vertices', [])
        if vertices:
            # Вычисляем центр полигона
            center_x = sum(v['x'] for v in vertices) / len(vertices)
            center_y = sum(v['y'] for v in vertices) / len(vertices)
            
            # Преобразуем координаты центра в SVG систему
            svg_center_x, svg_center_y = transform_coordinates(center_x, center_y, inverse_scale, padding)
            
            # Вычисляем координаты левого верхнего угла квадрата
            square_x = svg_center_x - square_size / 2
            square_y = svg_center_y - square_size / 2
            
            # Создаем квадрат вместо полигона
            square_element = dwg.rect(
                insert=(square_x, square_y),
                size=(square_size, square_size),
                **styles['pillar']
            )
            pillars_group.add(square_element)
            
            # Добавляем номер колонны
            text = dwg.text(
                f"P{idx+1}",
                insert=(svg_center_x, svg_center_y),
                text_anchor='middle',
                fill='white',
                font_size='12px',
                font_weight='bold'
            )
            pillars_group.add(text)
            
            print(f"    ✓ Колонна P{idx+1}: центр({center_x:.1f}, {center_y:.1f}) -> квадрат {square_size:.2f}x{square_size:.2f} px")

def draw_openings_bboxes(dwg: svgwrite.Drawing, data: Dict, inverse_scale: float, padding: float, styles: Dict, wall_thickness: float) -> None:
    """Отрисовывает окна и двери (прямоугольники) с толщиной равной толщине стены"""
    openings_group = dwg.add(dwg.g(id='openings'))
    
    windows_group = openings_group.add(dwg.g(id='windows'))
    doors_group = openings_group.add(dwg.g(id='doors'))
    
    openings = data.get('openings', [])
    windows_count = 0
    doors_count = 0
    
    # Размер толщины в SVG координатах
    svg_wall_thickness = wall_thickness * inverse_scale
    
    for opening in openings:
        opening_type = opening.get('type', '')
        bbox = opening.get('bbox', {})
        
        if bbox:
            # Определяем ориентацию проема
            width = bbox['width']
            height = bbox['height']
            orientation = 'horizontal' if width > height else 'vertical'
            
            # Преобразуем координаты
            x, y = transform_coordinates(bbox['x'], bbox['y'], inverse_scale, padding)
            
            # Определяем размеры с учетом толщины стены
            if orientation == 'horizontal':
                # Горизонтальное окно/дверь: ширина остается, высота = толщине стены
                svg_width = width * inverse_scale
                svg_height = svg_wall_thickness
                # Центрируем по вертикали
                y = y + (height * inverse_scale - svg_wall_thickness) / 2
            else:
                # Вертикальное окно/дверь: высота остается, ширина = толщине стены
                svg_width = svg_wall_thickness
                svg_height = height * inverse_scale
                # Центрируем по горизонтали
                x = x + (width * inverse_scale - svg_wall_thickness) / 2
            
            # Определяем ориентацию проема для метки
            orientation_label = 'h' if orientation == 'horizontal' else 'v'
            
            if opening_type == 'window':
                # Создаем окно
                rect = dwg.rect(insert=(x, y), size=(svg_width, svg_height), **styles['window'])
                windows_group.add(rect)
                windows_count += 1
                
                # Добавляем метку
                opening_id = opening.get('id', '').split('_')[-1] if opening.get('id') else str(windows_count)
                text = dwg.text(
                    f"W{opening_id}{orientation_label}",
                    insert=(x + svg_width/2, y + svg_height/2),
                    text_anchor='middle',
                    fill='black',
                    font_size='10px',
                    font_weight='bold'
                )
                windows_group.add(text)
                
            elif opening_type == 'door':
                # Создаем дверь
                rect = dwg.rect(insert=(x, y), size=(svg_width, svg_height), **styles['door'])
                doors_group.add(rect)
                doors_count += 1
                
                # Добавляем метку
                opening_id = opening.get('id', '').split('_')[-1] if opening.get('id') else str(doors_count)
                text = dwg.text(
                    f"D{opening_id}{orientation_label}",
                    insert=(x + svg_width/2, y + svg_height/2),
                    text_anchor='middle',
                    fill='black',
                    font_size='10px',
                    font_weight='bold'
                )
                doors_group.add(text)
    
    print(f"  ✓ Отрисовка {windows_count} окон и {doors_count} дверей с толщиной {wall_thickness} px")

def add_legend(dwg: svgwrite.Drawing, width: int, height: int, styles: Dict) -> None:
    """Добавляет легенду с описанием цветов и типов junctions"""
    legend_group = dwg.add(dwg.g(id='legend'))
    
    # Позиция легенды
    legend_x = 20
    legend_y = height - 320  # Увеличил для junction types
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
        ("Стены (opening-based)", styles['wall']),
        ("Стены (junction-based)", {'stroke': '#FF6347', 'fill': 'none'}),
        ("Стеновые полигоны (JSON)", {'stroke': '#808080', 'fill': 'none', 'stroke_dasharray': '5,5'}),
        ("Junction L-типа", get_junction_style('L')),
        ("Junction T-типа", get_junction_style('T')),
        ("Junction X-типа", get_junction_style('X')),
        ("Прямое соединение", get_junction_style('straight')),
        ("Неизвестный тип", get_junction_style('unknown')),
        ("Расширенные сегменты", {'stroke': '#FF69B4', 'fill': 'none', 'stroke_dasharray': '10,5'}),
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
        "План этажа - Opening-Based с анализом типов Junctions",
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

def visualize_polygons_opening_based_with_junction_types():
    """Основная функция создания векторной визуализации на основе проемов с анализом типов junctions"""
    print("="*60)
    print("СОЗДАНИЕ ВЕКТОРНОЙ ВИЗУАЛИЗАЦИИ НА ОСНОВЕ ПРОЕМОВ С АНАЛИЗОМ ТИПОВ JUNCTIONS")
    print("="*60)
    
    # Параметры
    input_path = 'plan_floor1_objects.json'
    output_path = 'wall_polygons.svg'
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
    
    # Обрабатываем проемы с учетом junctions
    print(f"\n{'='*60}")
    print("ОБРАБОТКА ПРОЕМОВ С УЧЕТОМ JUNCTIONS")
    print(f"{'='*60}")
    
    wall_segments = process_openings_with_junctions(data, wall_thickness)
    
    # Анализируем типы junctions с улучшенной логикой
    print(f"\n{'='*60}")
    print("АНАЛИЗ ТИПОВ JUNCTIONS С УЛУЧШЕННОЙ ЛОГИКОЙ")
    print(f"{'='*60}")
    
    junctions = parse_junctions(data)
    junctions_with_types = analyze_junction_types_with_thickness(junctions, data, wall_thickness)
    
    # Находим junctions не связанные с проемами и достраиваем недостающие сегменты
    print(f"\n{'='*60}")
    print("ПОИСК JUNCTIONS НЕ СВЯЗАННЫХ С ПРОЕМАМИ И ДОСТРАИВАНИЕ СТЕН")
    print(f"{'='*60}")
    
    junction_wall_segments = []  # Инициализируем пустой список для сегментов стен из junctions
    max_iterations = 5  # Максимальное количество итераций для предотвращения бесконечного цикла
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Итерация {iteration} ---")
        
        # Находим junctions не связанные с проемами или уже построенными сегментами
        non_opening_junctions = find_junctions_not_related_to_openings(junctions_with_types, wall_segments, junction_wall_segments)
        
        if not non_opening_junctions:
            print(f"  ✓ Все junctions связаны с проемами или уже имеют достаточное количество сегментов стен")
            break
        
        # Достраиваем недостающие сегменты
        new_segments = build_missing_wall_segments_for_junctions(
            non_opening_junctions, wall_segments, junction_wall_segments, junctions_with_types, wall_thickness
        )
        
        if not new_segments:
            print(f"  ✓ Не удалось создать новые сегменты стен на этой итерации")
            break
        
        # Добавляем новые сегменты к общему списку
        junction_wall_segments.extend(new_segments)
        print(f"  ✓ Создано {len(new_segments)} новых сегментов стен на итерации {iteration}")
    
    # Process L-junctions and create extensions
    print(f"\n{'='*60}")
    print("ОБРАБОТКА L-JUNCTIONS И СОЗДАНИЕ РАСШИРЕНИЙ")
    print(f"{'='*60}")
    
    extended_segments = process_l_junction_extensions(
        junctions_with_types, wall_segments, junction_wall_segments,
        data.get('wall_polygons', []), wall_thickness
    )
    
    print(f"\n{'='*60}")
    print(f"ИТОГО: {len(wall_segments)} сегментов стен из проемов + {len(junction_wall_segments)} сегментов стен из junctions + {len(extended_segments)} расширенных сегментов")
    print(f"{'='*60}")
    
    # Convert WallSegmentFromOpening to bbox format for visualization
    processed_segments = [segment.bbox for segment in wall_segments]
    # Add junction-based segments
    processed_segments.extend([segment.bbox for segment in junction_wall_segments])
    # Add extended segments
    processed_segments.extend(extended_segments)
    
    # Вычисляем размеры и масштаб
    max_x, max_y, inverse_scale = calculate_svg_dimensions(data, processed_segments)
    
    # Вычисляем размеры SVG с учетом масштаба и отступов
    svg_width = int(max_x * inverse_scale + padding * 2)
    svg_height = int(max_y * inverse_scale + padding * 2)
    
    print(f"  ✓ Размеры SVG: {svg_width}x{svg_height}")
    
    # Создаем SVG документ
    dwg = create_svg_document(output_path, svg_width, svg_height, data)
    
    # Определяем стили
    styles = define_styles()
    
    # Отрисовываем объекты в правильном порядке (слои)
    # 1. Сначала отрисовываем все стеновые полигоны (самый нижний слой)
    draw_all_wall_polygons(dwg, data, inverse_scale, padding)
    
    # 2. Затем отрисовываем junctions с типами
    draw_junctions_with_types(dwg, junctions_with_types, inverse_scale, padding)
    
    # 3. Затем отрисовываем колонны как квадраты
    draw_pillars(dwg, data, inverse_scale, padding, styles, wall_thickness)
    
    # 4. Затем отрисовываем окна и двери с толщиной равной толщине стены
    draw_openings_bboxes(dwg, data, inverse_scale, padding, styles, wall_thickness)
    
    # 5. Отрисовываем сегменты стен из junctions
    draw_junction_based_wall_bboxes(dwg, junction_wall_segments, inverse_scale, padding, styles)
    
    # 6. Отрисовываем расширенные сегменты
    draw_extended_segments(dwg, extended_segments, inverse_scale, padding)
    
    # 7. Наконец отрисовываем сегменты стен из проемов (верхний слой)
    draw_opening_based_wall_bboxes(dwg, wall_segments, inverse_scale, padding, styles)
    
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
    junctions_count = len(junctions_with_types)
    
    # Считаем статистику по типам junctions
    type_counts = {}
    for junction in junctions_with_types:
        jtype = junction.detected_type
        type_counts[jtype] = type_counts.get(jtype, 0) + 1
    
    print(f"\nСтатистика объектов:")
    print(f"  Полигоны стен: {wall_polygons_count}")
    print(f"  Сегменты стен из проемов: {len(wall_segments)}")
    print(f"  Сегменты стен из junctions: {len(junction_wall_segments)}")
    print(f"  Расширенные сегменты: {len(extended_segments)}")
    print(f"  Всего сегментов стен: {len(wall_segments) + len(junction_wall_segments) + len(extended_segments)}")
    print(f"  Колонны: {pillar_polygons_count}")
    print(f"  Окна: {windows_count}")
    print(f"  Двери: {doors_count}")
    print(f"  Junctions: {junctions_count}")
    print(f"  Толщина стен (минимальная толщина двери): {wall_thickness:.1f} px")
    
    print(f"\nСтатистика по типам junctions:")
    for jtype, count in type_counts.items():
        print(f"  {jtype}: {count}")
    
    print(f"\nГотово! Векторная визуализация на основе проемов с анализом типов junctions создана: {output_path}")
    print("Откройте файл в браузере или векторном редакторе для просмотра")

# =============================================================================
# ФУНКЦИЯ ЗАМЕНЫ КОЛОНН НА КВАДРАТЫ
# =============================================================================

def replace_pillars_with_squares(svg_path: str, wall_thickness: float) -> None:
    """
    Заменяет колонны P1 и P2 на квадраты со стороной равной толщине стены
    
    Args:
        svg_path: Путь к SVG файлу
        wall_thickness: Толщина стены для определения размера квадратов
    """
    print(f"\n{'='*60}")
    print("ЗАМЕНА КОЛОНН P1 И P2 НА КВАДРАТЫ")
    print(f"{'='*60}")
    
    try:
        # Читаем SVG файл
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        print(f"  ✓ SVG файл загружен: {svg_path}")
        
        # Находим координаты центров колонн P1 и P2
        # Из SVG видно, что P1 находится примерно на координатах (576, 2695)
        # а P2 на координатах (1269, 2694)
        
        # Получаем масштабный коэффициент из SVG
        import re
        scale_match = re.search(r'Масштабный коэффициент: ([\d.]+)', svg_content)
        scale_factor = float(scale_match.group(1)) if scale_match else 1.0
        
        # Вычисляем размер квадрата на основе толщины стены с учетом масштаба
        square_size = wall_thickness * scale_factor
        
        print(f"  ✓ Масштабный коэффициент: {scale_factor}")
        print(f"  ✓ Толщина стены: {wall_thickness} px")
        print(f"  ✓ Размер квадрата: {square_size} px")
        
        # Находим группу колонн
        pillars_group_match = re.search(r'<g id="pillars">(.*?)</g>', svg_content, re.DOTALL)
        if not pillars_group_match:
            print("  ✗ Группа колонн не найдена в SVG")
            return
        
        pillars_group_content = pillars_group_match.group(1)
        
        # Заменяем полигоны колонн на квадраты
        # P1: примерно (576, 2695)
        # P2: примерно (1269, 2694)
        
        # Находим и заменяем P1
        p1_polygon_match = re.search(r'<polygon[^>]*points="[^"]*"[^>]*>.*?</polygon>', pillars_group_content, re.DOTALL)
        if p1_polygon_match:
            # Вычисляем координаты для квадрата P1
            p1_x = 576 - square_size / 2
            p1_y = 2695 - square_size / 2
            
            p1_square = f'<rect x="{p1_x}" y="{p1_y}" width="{square_size}" height="{square_size}" fill="#D2691E" stroke="#8B4513" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" />'
            
            # Заменяем полигон на квадрат
            pillars_group_content = pillars_group_content.replace(p1_polygon_match.group(0), p1_square)
            
            # Также заменяем текстовую метку
            p1_text_match = re.search(r'<text[^>]*>P1</text>', pillars_group_content)
            if p1_text_match:
                p1_text = f'<text fill="white" font-size="12px" font-weight="bold" text-anchor="middle" x="576" y="2695">P1</text>'
                pillars_group_content = pillars_group_content.replace(p1_text_match.group(0), p1_text)
            
            print(f"  ✓ Колонна P1 заменена на квадрат {square_size}x{square_size} px")
        
        # Находим и заменяем P2
        p2_polygon_match = re.search(r'<polygon[^>]*points="[^"]*"[^>]*>.*?</polygon>', pillars_group_content, re.DOTALL)
        if p2_polygon_match:
            # Вычисляем координаты для квадрата P2
            p2_x = 1269 - square_size / 2
            p2_y = 2694 - square_size / 2
            
            p2_square = f'<rect x="{p2_x}" y="{p2_y}" width="{square_size}" height="{square_size}" fill="#D2691E" stroke="#8B4513" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" />'
            
            # Заменяем полигон на квадрат
            pillars_group_content = pillars_group_content.replace(p2_polygon_match.group(0), p2_square)
            
            # Также заменяем текстовую метку
            p2_text_match = re.search(r'<text[^>]*>P2</text>', pillars_group_content)
            if p2_text_match:
                p2_text = f'<text fill="white" font-size="12px" font-weight="bold" text-anchor="middle" x="1269" y="2694">P2</text>'
                pillars_group_content = pillars_group_content.replace(p2_text_match.group(0), p2_text)
            
            print(f"  ✓ Колонна P2 заменена на квадрат {square_size}x{square_size} px")
        
        # Заменяем старую группу колонн на новую
        new_pillars_group = f'<g id="pillars">{pillars_group_content}</g>'
        svg_content = svg_content.replace(pillars_group_match.group(0), new_pillars_group)
        
        # Создаем новый файл с замененными колоннами
        new_svg_path = svg_path.replace('.svg', '_with_square_pillars.svg')
        with open(new_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"  ✓ Новый SVG файл сохранен: {new_svg_path}")
        print(f"\n✓ Колонны P1 и P2 успешно заменены на квадраты со стороной {wall_thickness} px")
        
    except Exception as e:
        print(f"  ✗ Ошибка при замене колонн: {e}")

def replace_pillars_in_existing_svg():
    """
    Автономная функция для замены колонн в существующем SVG файле
    """
    svg_path = 'wall_polygons.svg'
    
    # Сначала определяем толщину стен из JSON файла
    json_path = 'plan_floor1_objects.json'
    data = load_objects_data(json_path)
    if not data:
        print("✗ Ошибка: не удалось загрузить данные для определения толщины стен")
        return
    
    wall_thickness = get_wall_thickness_from_doors(data)
    
    print("="*60)
    print("ЗАМЕНА КОЛОНН В СУЩЕСТВУЮЩЕМ SVG ФАЙЛЕ")
    print("="*60)
    
    replace_pillars_with_squares(svg_path, wall_thickness)

if __name__ == '__main__':
    # Сначала пробуем запустить основную функцию
    try:
        visualize_polygons_opening_based_with_junction_types()
    except Exception as e:
        print(f"✗ Ошибка при выполнении основной функции: {e}")
        print("\nЗапускаем замену колонн в существующем SVG файле...")
        replace_pillars_in_existing_svg()