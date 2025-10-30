#!/usr/bin/env python3
"""
Интегрированная версия visualize_polygons_opening_based.py с анализом типов junctions
и хранением всех координат в JSON

Создает SVG файл с ограничивающими прямоугольниками на основе проемов и junctions,
а также определяет и визуализирует типы junctions (L, T, X).

КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
1. Стены строятся напрямую от проемов к junctions
2. Определяются типы junctions (L, T, X)
3. Визуализация включает цветовую кодировку типов junctions
4. Все координаты сохраняются в JSON для последующего использования
"""

import json
import svgwrite
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import math
import sys
import datetime
from dataclasses import dataclass
from collections import defaultdict

# Добавляем путь к текущей директории для импорта модуля
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_junction_type_analyzer import analyze_polygon_extensions_with_thickness, is_point_in_polygon
from visualize_polygons_align import align_walls_by_openings

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
# ФУНКЦИИ ДЛЯ РАБОТЫ С JSON
# =============================================================================

def initialize_json_data(input_path: str, wall_thickness: float) -> Dict:
    """Инициализирует структуру JSON для хранения данных"""
    return {
        "metadata": {
            "source_file": input_path,
            "wall_thickness": wall_thickness,
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0"
        },
        "junctions": [],
        "wall_segments_from_openings": [],
        "wall_segments_from_junctions": [],
        "openings": [],
        "pillar_squares": [],
        "statistics": {
            "total_junctions": 0,
            "total_wall_segments_from_openings": 0,
            "total_wall_segments_from_junctions": 0,
            "total_openings": 0,
            "total_pillar_squares": 0,
            "extended_segments": 0
        }
    }

def add_junction_to_json(json_data: Dict, junction: JunctionPoint) -> None:
    """Добавляет junction в JSON структуру"""
    junction_data = {
        "id": junction.id,
        "x": junction.x,
        "y": junction.y,
        "junction_type": junction.junction_type,
        "detected_type": junction.detected_type,
        "directions": junction.directions,
        "confidence": junction.confidence,
        "connected_wall_segments": []  # Инициализация массива для привязки сегментов стен
    }
    json_data["junctions"].append(junction_data)
    json_data["statistics"]["total_junctions"] += 1

def add_junction_wall_binding(json_data: Dict, junction_id: int, segment_id: str,
                            segment_type: str, connection_type: str) -> None:
    """
    Добавляет привязку сегмента стены к junction
    
    Args:
        json_data: JSON структура для сохранения данных
        junction_id: ID junction
        segment_id: ID сегмента стены
        segment_type: Тип сегмента ('from_openings' или 'from_junctions')
        connection_type: Тип подключения ('start' или 'end')
    """
    # Находим junction по ID
    for junction in json_data["junctions"]:
        if junction["id"] == junction_id:
            # Проверяем, нет ли уже такой привязки
            existing_binding = None
            for binding in junction["connected_wall_segments"]:
                if (binding["segment_id"] == segment_id and
                    binding["segment_type"] == segment_type):
                    existing_binding = binding
                    break
            
            # Если привязка уже существует, обновляем её
            if existing_binding:
                existing_binding["connection_type"] = connection_type
            else:
                # Добавляем новую привязку
                junction["connected_wall_segments"].append({
                    "segment_id": segment_id,
                    "segment_type": segment_type,
                    "connection_type": connection_type
                })
            break

def add_wall_segment_from_opening_to_json(json_data: Dict, segment: WallSegmentFromOpening) -> None:
    """Добавляет сегмент стены от проема в JSON структуру"""
    segment_data = {
        "segment_id": segment.segment_id,
        "opening_id": segment.opening_id,
        "edge_side": segment.edge_side,
        "start_junction_id": segment.start_junction.id,
        "end_junction_id": segment.end_junction.id,
        "orientation": segment.orientation,
        "bbox": segment.bbox,
        "alignment_info": segment.bbox.get("alignment_info", {})
    }
    json_data["wall_segments_from_openings"].append(segment_data)
    json_data["statistics"]["total_wall_segments_from_openings"] += 1
    
    # Добавляем привязки к junctions
    add_junction_wall_binding(json_data, segment.start_junction.id, segment.segment_id,
                            "from_openings", "start")
    add_junction_wall_binding(json_data, segment.end_junction.id, segment.segment_id,
                            "from_openings", "end")

def add_wall_segment_from_junction_to_json(json_data: Dict, segment: WallSegmentFromJunction) -> None:
    """Добавляет сегмент стены между junctions в JSON структуру"""
    segment_data = {
        "segment_id": segment.segment_id,
        "start_junction_id": segment.start_junction.id,
        "end_junction_id": segment.end_junction.id,
        "direction": segment.direction,
        "orientation": segment.orientation,
        "bbox": segment.bbox
    }
    json_data["wall_segments_from_junctions"].append(segment_data)
    json_data["statistics"]["total_wall_segments_from_junctions"] += 1
    
    # Добавляем привязки к junctions
    add_junction_wall_binding(json_data, segment.start_junction.id, segment.segment_id,
                            "from_junctions", "start")
    add_junction_wall_binding(json_data, segment.end_junction.id, segment.segment_id,
                            "from_junctions", "end")

def update_existing_junctions_with_wall_bindings(json_data: Dict) -> None:
    """
    Обновляет существующие junctions привязками к сегментам стен
    Вызывается после создания всех сегментов стен
    
    Args:
        json_data: JSON структура с данными
    """
    print(f"  Обновление junctions привязками к сегментам стен...")
    
    # Проходим по всем сегментам стен из проемов
    for segment in json_data["wall_segments_from_openings"]:
        segment_id = segment["segment_id"]
        start_junction_id = segment["start_junction_id"]
        end_junction_id = segment["end_junction_id"]
        
        # Добавляем привязки для начального junction
        add_junction_wall_binding(json_data, start_junction_id, segment_id,
                                "from_openings", "start")
        
        # Добавляем привязки для конечного junction
        add_junction_wall_binding(json_data, end_junction_id, segment_id,
                                "from_openings", "end")
    
    # Проходим по всем сегментам стен из junctions
    for segment in json_data["wall_segments_from_junctions"]:
        segment_id = segment["segment_id"]
        start_junction_id = segment["start_junction_id"]
        end_junction_id = segment["end_junction_id"]
        
        # Добавляем привязки для начального junction
        add_junction_wall_binding(json_data, start_junction_id, segment_id,
                                "from_junctions", "start")
        
        # Добавляем привязки для конечного junction
        add_junction_wall_binding(json_data, end_junction_id, segment_id,
                                "from_junctions", "end")
    
    print(f"  ✓ Обновление junctions завершено")

def add_opening_to_json(json_data: Dict, opening: Dict, edge_junctions: List[Tuple[str, JunctionPoint]]) -> None:
    """Добавляет проем в JSON структуру"""
    opening_data = {
        "id": opening.get('id', ''),
        "type": opening.get('type', ''),
        "bbox": opening.get('bbox', {}),
        "orientation": detect_opening_orientation(opening.get('bbox', {})),
        "edge_junctions": [
            {
                "edge_side": edge_side,
                "junction_id": junction.id
            }
            for edge_side, junction in edge_junctions
        ]
    }
    json_data["openings"].append(opening_data)
    json_data["statistics"]["total_openings"] += 1

def save_json_data(json_data: Dict, output_path: str) -> None:
    """Сохраняет JSON данные в файл"""
    print(f"Сохранение данных в JSON: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ Данные успешно сохранены")

def add_pillar_square_to_json(json_data: Dict, pillar_id: str, center_x: float, center_y: float,
                             wall_thickness: float, original_vertices: List[Dict] = None) -> None:
    """
    Добавляет квадрат колонны в JSON структуру
    
    Args:
        json_data: JSON структура для сохранения данных
        pillar_id: Идентификатор колонны
        center_x: X-координата центра колонны
        center_y: Y-координата центра колонны
        wall_thickness: Толщина стены (размер квадрата)
        original_vertices: Оригинальные вершины полигона колонны
    """
    # Вычисляем координаты квадрата
    half_size = wall_thickness / 2
    square_x = center_x - half_size
    square_y = center_y - half_size
    
    pillar_square = {
        "id": pillar_id,
        "type": "pillar_square",
        "center_x": center_x,
        "center_y": center_y,
        "bbox": {
            "x": square_x,
            "y": square_y,
            "width": wall_thickness,
            "height": wall_thickness
        },
        "wall_thickness": wall_thickness,
        "original_vertices": original_vertices or []
    }
    
    # Добавляем в JSON, если раздел еще не существует
    if "pillar_squares" not in json_data:
        json_data["pillar_squares"] = []
        json_data["statistics"]["total_pillar_squares"] = 0
    
    json_data["pillar_squares"].append(pillar_square)
    json_data["statistics"]["total_pillar_squares"] += 1

def create_svg_from_json(json_path: str, svg_output_path: str, original_data: Dict) -> None:
    """Создает SVG файл на основе сохраненных JSON данных"""
    print(f"Создание SVG из JSON: {json_path}")
    
    # Загружаем данные из JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Вычисляем размеры SVG
    max_x, max_y = 0, 0
    
    # Проверяем сегменты стен
    for segment in json_data["wall_segments_from_openings"]:
        bbox = segment["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
    
    for segment in json_data["wall_segments_from_junctions"]:
        bbox = segment["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
    
    # Проверяем проемы
    for opening in json_data["openings"]:
        bbox = opening["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
    
    # Проверяем полигоны колонн
    for polygon in original_data.get('pillar_polygons', []):
        for vertex in polygon.get('vertices', []):
            max_x = max(max_x, vertex.get('x', 0))
            max_y = max(max_y, vertex.get('y', 0))
    
    # Проверяем junctions
    for junction in json_data["junctions"]:
        max_x = max(max_x, junction["x"])
        max_y = max(max_y, junction["y"])
    
    # Получаем масштабный коэффициент из метаданных
    scale_factor = original_data.get('metadata', {}).get('scale_factor', 1.0)
    inverse_scale = 1.0 / scale_factor
    
    # Добавляем отступы
    padding = 50
    svg_width = int(max_x * inverse_scale + padding * 2)
    svg_height = int(max_y * inverse_scale + padding * 2)
    
    # Создаем SVG документ
    dwg = svgwrite.Drawing(svg_output_path, size=(svg_width, svg_height), profile='full')
    dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill='white'))
    
    # Определяем стили
    styles = define_styles()
    
    # Отрисовываем все стеновые полигоны (самый нижний слой)
    draw_all_wall_polygons(dwg, original_data, inverse_scale, padding)
    
    # Отрисовываем junctions с типами
    junctions_group = dwg.add(dwg.g(id='junctions'))
    for junction in json_data["junctions"]:
        svg_x, svg_y = transform_coordinates(junction["x"], junction["y"], inverse_scale, padding)
        junction_style = get_junction_style(junction["detected_type"])
        circle = dwg.circle(center=(svg_x, svg_y), r=5, **junction_style)
        junctions_group.add(circle)
        
        # Добавляем номер
        text = dwg.text(
            f"J{junction['id']}",
            insert=(svg_x + 10, svg_y - 5),
            text_anchor='start',
            fill='black',
            font_size='8px',
            font_weight='bold'
        )
        junctions_group.add(text)
        
        # Добавляем тип и направления
        if junction["directions"]:
            direction_map = {'left': 'L', 'right': 'R', 'up': 'U', 'down': 'D'}
            direction_abbr = '-'.join([direction_map.get(d, d[0].upper()) for d in junction["directions"]])
            type_text = dwg.text(
                f"{junction['detected_type']} {direction_abbr}",
                insert=(svg_x + 10, svg_y + 8),
                text_anchor='start',
                fill='black',
                font_size='6px'
            )
        else:
            type_text = dwg.text(
                f"{junction['detected_type']}",
                insert=(svg_x + 10, svg_y + 8),
                text_anchor='start',
                fill='black',
                font_size='6px'
            )
        junctions_group.add(type_text)
    
    # Отрисовываем колонны как квадраты
    draw_pillars(dwg, original_data, inverse_scale, padding, styles, json_data["metadata"]["wall_thickness"])
    
    # Отрисовываем окна и двери с толщиной равной толщине стены
    draw_openings_bboxes(dwg, original_data, inverse_scale, padding, styles, json_data["metadata"]["wall_thickness"])
    
    # Отрисовываем сегменты стен от junctions (нижний слой)
    junction_walls_group = dwg.add(dwg.g(id='junction_based_walls'))
    junction_wall_style = {
        'stroke': '#FF6347',
        'stroke_width': 2,
        'fill': 'none',
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round'
    }
    
    for segment in json_data["wall_segments_from_junctions"]:
        bbox = segment["bbox"]
        x, y = transform_coordinates(bbox["x"], bbox["y"], inverse_scale, padding)
        width, height = bbox["width"] * inverse_scale, bbox["height"] * inverse_scale
        
        rect = dwg.rect(insert=(x, y), size=(width, height), **junction_wall_style)
        junction_walls_group.add(rect)
        
        # Добавляем номер сегмента с информацией о junctions и направлении
        orientation_label = 'h' if segment["orientation"] == 'horizontal' else 'v'
        direction_label = segment["direction"][0].upper()  # L, R, U, D
        
        text = dwg.text(
            f"J{segment['start_junction_id']}->{segment['end_junction_id']}_{direction_label}{orientation_label}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='#FF6347',
            font_size='8px',
            font_weight='bold'
        )
        junction_walls_group.add(text)
    
    # Отрисовываем сегменты стен от проемов (верхний слой)
    opening_walls_group = dwg.add(dwg.g(id='opening_based_walls'))
    for segment in json_data["wall_segments_from_openings"]:
        bbox = segment["bbox"]
        x, y = transform_coordinates(bbox["x"], bbox["y"], inverse_scale, padding)
        width, height = bbox["width"] * inverse_scale, bbox["height"] * inverse_scale
        
        rect = dwg.rect(insert=(x, y), size=(width, height), **styles['wall'])
        opening_walls_group.add(rect)
        
        # Добавляем номер сегмента с информацией о проеме и ориентации
        orientation_label = 'h' if segment["orientation"] == 'horizontal' else 'v'
        opening_short_id = segment["opening_id"].split('_')[-1] if '_' in segment["opening_id"] else segment["opening_id"]
        
        text = dwg.text(
            f"W{segment['segment_id']}_{opening_short_id}_{segment['edge_side'][0]}{orientation_label}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='red',
            font_size='10px',
            font_weight='bold'
        )
        opening_walls_group.add(text)
    
    # Добавляем легенду и заголовок
    add_legend(dwg, svg_width, svg_height, styles)
    add_title(dwg, svg_width, original_data)
    
    # Сохраняем SVG
    dwg.save()
    print(f"  ✓ SVG сохранен: {svg_output_path}")

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
    best_distance = float('inf')
    
    for junction in junctions:
        distance = math.sqrt((junction.x - x)**2 + (junction.y - y)**2)
        if distance < best_distance and distance <= tolerance:
            best_distance = distance
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

def find_next_junction_in_direction(start_junction: JunctionPoint, direction: str, junctions: List[JunctionPoint], tolerance: float, min_distance: float = 0.0) -> Optional[JunctionPoint]:
    """
    Finds the next junction in a given direction from a starting junction.
    
    Args:
        start_junction: The starting junction point
        direction: Direction to search ('left', 'right', 'up', 'down')
        junctions: List of all junction points
        tolerance: Tolerance for alignment checking (perpendicular axis)
        min_distance: Minimum allowed distance along the search axis. Junctions
                      closer than this are ignored.
    
    Returns:
        The next junction in the specified direction, or None if not found
    """
    closest_junction = None
    best_distance = float('inf')
    
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
        
        # Пропускаем слишком близкие узлы (меньше минимально допустимого расстояния)
        if is_aligned and distance < min_distance:
            continue

        if is_aligned and distance < best_distance:
            best_distance = distance
            closest_junction = junction
    
    return closest_junction

def find_candidate_junctions_in_direction(start_junction: JunctionPoint, direction: str, junctions: List[JunctionPoint], tolerance: float, max_candidates: int = 10) -> List[JunctionPoint]:
    """
    Находит несколько ближайших junctions в заданном направлении, отсортированных по расстоянию.

    Args:
        start_junction: начальный узел
        direction: 'left' | 'right' | 'up' | 'down'
        junctions: все узлы
        tolerance: допуск выравнивания по перпендикулярной оси
        max_candidates: максимальное количество кандидатов

    Returns:
        Список узлов-кандидатов, отсортированных по возрастанию расстояния
    """
    candidates = []
    for junction in junctions:
        if junction.id == start_junction.id:
            continue

        aligned = False
        distance = 0.0

        if direction == 'left':
            aligned = (junction.x < start_junction.x and abs(junction.y - start_junction.y) <= tolerance)
            if aligned:
                distance = start_junction.x - junction.x
        elif direction == 'right':
            aligned = (junction.x > start_junction.x and abs(junction.y - start_junction.y) <= tolerance)
            if aligned:
                distance = junction.x - start_junction.x
        elif direction == 'up':
            aligned = (junction.y < start_junction.y and abs(junction.x - start_junction.x) <= tolerance)
            if aligned:
                distance = start_junction.y - junction.y
        elif direction == 'down':
            aligned = (junction.y > start_junction.y and abs(junction.x - start_junction.x) <= tolerance)
            if aligned:
                distance = junction.y - start_junction.y

        if aligned:
            candidates.append((distance, junction))

    candidates.sort(key=lambda t: t[0])
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return [j for _, j in candidates]

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

def find_nearby_opening_in_direction(opening: Dict, direction: str, all_openings: List[Dict], wall_thickness: float) -> Optional[Dict]:
    """
    Ищет проем в пределах половины толщины стены в заданном направлении от текущего проема
    
    Args:
        opening: Текущий проем с bbox
        direction: Направление поиска ('left', 'right', 'up', 'down')
        all_openings: Список всех проемов
        wall_thickness: Толщина стены
    
    Returns:
        Найденный проем или None, если проем не найден
    """
    bbox = opening.get('bbox', {})
    if not bbox:
        return None
    
    x, y = bbox['x'], bbox['y']
    width, height = bbox['width'], bbox['height']
    half_thickness = wall_thickness / 2.0
    
    # Ищем подходящие проемы
    closest_opening = None
    min_distance = float('inf')
    
    for other_opening in all_openings:
        # Пропускаем тот же проем
        if other_opening.get('id') == opening.get('id'):
            continue
        
        other_bbox = other_opening.get('bbox', {})
        if not other_bbox:
            continue
        
        other_x, other_y = other_bbox['x'], other_bbox['y']
        other_width, other_height = other_bbox['width'], other_bbox['height']
        
        # Проверяем, находится ли проем в правильном направлении
        is_in_direction = False
        distance = 0
        
        # ИСПРАВЛЕНИЕ: Добавляем проверку на выравнивание по оси
        is_aligned = False
        
        if direction in ['left', 'right']:
            # Для горизонтальных проемов проверяем выравнивание по оси Y
            y_overlap = not (other_y + other_height < y or other_y > y + height)
            is_aligned = y_overlap
        else:  # up, down
            # Для вертикальных проемов проверяем выравнивание по оси X
            x_overlap = not (other_x + other_width < x or other_x > x + width)
            is_aligned = x_overlap
        
        if direction == 'left':
            # Проем должен быть слева
            if other_x + other_width < x:
                is_in_direction = True
                distance = x - (other_x + other_width)
            # ДОБАВЛЕНО: Проверяем, являются ли проемы смежными или перекрывающимися
            elif other_x + other_width >= x and other_x + other_width <= x + wall_thickness / 2.0:
                is_in_direction = True
                distance = 0  # Считаем расстояние нулевым для смежных/перекрывающихся проемов
        elif direction == 'right':
            # Проем должен быть справа
            if other_x > x + width:
                is_in_direction = True
                distance = other_x - (x + width)
            # ДОБАВЛЕНО: Проверяем, являются ли проемы смежными или перекрывающимися
            elif other_x <= x + width and other_x >= x + width - wall_thickness / 2.0:
                is_in_direction = True
                distance = 0  # Считаем расстояние нулевым для смежных/перекрывающихся проемов
        elif direction == 'up':
            # Проем должен быть выше
            if other_y + other_height < y:
                is_in_direction = True
                distance = y - (other_y + other_height)
        elif direction == 'down':
            # Проем должен быть ниже
            if other_y > y + height:
                is_in_direction = True
                distance = other_y - (y + height)
        
        # ИСПРАВЛЕНИЕ: Учитываем выравнивание при определении соседнего проема
        if is_in_direction and not is_aligned:
            is_in_direction = False
        
        if is_in_direction:
            # Проверяем, что расстояние не превышает половину толщины стены
            if distance <= half_thickness:
                if distance < min_distance:
                    min_distance = distance
                    closest_opening = other_opening
            # ДОБАВЛЕНО: Проверяем, являются ли проемы смежными (касаются друг друга)
            elif distance == 0:
                # Проемы смежные
                if distance < min_distance:
                    min_distance = distance
                    closest_opening = other_opening
    
    return closest_opening

def find_nearest_junction_in_direction_from_point(x: float, y: float, direction: str, junctions: List[JunctionPoint], tolerance: float) -> Optional[JunctionPoint]:
    """
    Ищет ближайший junction в заданном направлении от точки
    
    Args:
        x, y: Начальные координаты
        direction: Направление поиска ('left', 'right', 'up', 'down')
        junctions: Список всех junction points
        tolerance: Допуск для выравнивания
    
    Returns:
        Ближайший junction или None, если не найден
    """
    closest_junction = None
    min_distance = float('inf')
    
    for junction in junctions:
        # Проверяем, находится ли junction в правильном направлении
        is_aligned = False
        distance = 0
        
        if direction == 'left':
            is_aligned = (junction.x < x and abs(junction.y - y) <= tolerance)
            distance = x - junction.x
        elif direction == 'right':
            is_aligned = (junction.x > x and abs(junction.y - y) <= tolerance)
            distance = junction.x - x
        elif direction == 'up':
            is_aligned = (junction.y < y and abs(junction.x - x) <= tolerance)
            distance = y - junction.y
        elif direction == 'down':
            is_aligned = (junction.y > y and abs(junction.x - x) <= tolerance)
            distance = junction.y - y
        
        if is_aligned and distance < min_distance:
            min_distance = distance
            closest_junction = junction
    
    return closest_junction

def extend_opening_to_junction(opening: Dict, direction: str, junction: JunctionPoint) -> Dict:
    """
    Расширяет или обрезает проем до junction в заданном направлении
    
    Args:
        opening: Проем с bbox
        direction: Направление расширения ('left', 'right', 'up', 'down')
        junction: Junction до которого нужно расширить/обрезать
    
    Returns:
        Измененный bbox проема
    """
    bbox = opening.get('bbox', {}).copy()
    if not bbox:
        return bbox
    
    x, y = bbox['x'], bbox['y']
    width, height = bbox['width'], bbox['height']
    
    if direction == 'left':
        # Расширяем влево до junction.x
        new_width = x + width - junction.x
        bbox['x'] = junction.x
        bbox['width'] = new_width
    elif direction == 'right':
        # Расширяем вправо до junction.x
        new_width = junction.x - x
        bbox['width'] = new_width
    elif direction == 'up':
        # Расширяем вверх до junction.y
        new_height = y + height - junction.y
        bbox['y'] = junction.y
        bbox['height'] = new_height
    elif direction == 'down':
        # Расширяем вниз до junction.y
        new_height = junction.y - y
        bbox['height'] = new_height
    
    return bbox

# =============================================================================
# ФУНКЦИИ ПОСТРОЕНИЯ СТЕНОВЫХ СЕГМЕНТОВ
# =============================================================================

def construct_wall_segment_from_opening(opening_with_junction: OpeningWithJunctions,
                                      junctions: List[JunctionPoint],
                                      wall_thickness: float,
                                      all_openings: List[Dict] = None,
                                      json_data: Dict = None,
                                      wall_polygons: List[Dict] = None) -> List[WallSegmentFromOpening]:
    """
    Constructs wall segments from each edge junction of an opening to the next junction.
    Строит сегменты стен, которые точно стыкуются с углами проемов.
    
    Args:
        opening_with_junction: Opening with its edge junctions
        junctions: List of all junction points
        wall_thickness: Thickness of walls
        json_data: JSON структура для сохранения данных
    
    Returns:
        List of wall segments constructed from the opening
    """
    wall_segments = []
    opening_id = opening_with_junction.opening_id
    orientation = opening_with_junction.orientation
    bbox = opening_with_junction.bbox
    
    print(f"    DEBUG: Обработка проема {opening_id}, ориентация: {orientation}")
    
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
        
        # Определяем точку на краю проема для поиска junction
        if direction == 'left':
            search_x, search_y = bbox['x'], bbox['y'] + bbox['height'] / 2
        elif direction == 'right':
            search_x, search_y = bbox['x'] + bbox['width'], bbox['y'] + bbox['height'] / 2
        elif direction == 'up':
            search_x, search_y = bbox['x'] + bbox['width'] / 2, bbox['y']
        else:  # down
            search_x, search_y = bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height']
        
        # Ищем следующий junction в направлении от стартового узла, но
        # игнорируем слишком близкие узлы: distance < wall_thickness/2
        
        # Теперь проверяем, есть ли рядом другой проем
        nearby_opening = None
        if all_openings:
            # Создаем временный объект проема для проверки
            temp_opening = {
                'id': opening_id,
                'bbox': bbox
            }
            nearby_opening = find_nearby_opening_in_direction(temp_opening, direction, all_openings, wall_thickness)
            print(f"    DEBUG: Проверка проема {opening_id} в направлении {direction}, найден соседний проем: {nearby_opening.get('id') if nearby_opening else 'None'}")
        
        if nearby_opening:
            # Если есть соседний проем, не создаем стену
            print(f"    Найден соседний проем {nearby_opening.get('id')}, стена не создается")
            continue  # Не создаем стену, если есть соседний проем
        else:
            print(f"    Соседний проем не найден, создаем стену")
        
        # Находим ближайший подходящий узел (не ближе половины толщины стены)
        end_junction = find_next_junction_in_direction(
            start_junction, direction, junctions,
            wall_thickness / 2.0,
            min_distance=wall_thickness / 2.0
        )

        if end_junction:
            # ИСПРАВЛЕНИЕ: Пересчитываем offset после того, как bbox мог быть изменен
            # Это необходимо, т.к. bbox мог быть сдвинут до junction выше
            if orientation == 'horizontal':
                if edge_side == 'left':
                    offset_x = start_junction.x - bbox['x']
                    offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
                else:  # right
                    offset_x = start_junction.x - (bbox['x'] + bbox['width'])
                    offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
            else:  # vertical
                if edge_side == 'top':
                    offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                    offset_y = start_junction.y - bbox['y']
                else:  # bottom
                    offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                    offset_y = start_junction.y - (bbox['y'] + bbox['height'])

            # Create wall segment from start to end junction
            segment_id = f"wall_{opening_id}_{edge_side}_{start_junction.id}_to_{end_junction.id}"

            # Создаем bbox, но с учетом смещения для точной стыковки с углами проема
            if orientation == 'horizontal':
                # ИСПРАВЛЕНИЕ: Убедимся, что стена начинается точно от junction
                # Для правой стороны проема, всегда начинаем от junction
                if edge_side == 'right':
                    start_x = start_junction.x
                    # УБРАНО: Коррекция на 1 пиксель для избежания перекрытия на SVG
                    # start_x += 1.0
                else:
                    # УБРАНО: Коррекция на 1 пиксель для левой стороны проема
                    # start_x = start_junction.x - 1.0
                    start_x = start_junction.x
                    
                x = min(start_x, end_junction.x)
                y = (start_junction.y - offset_y) - wall_thickness / 2
                width = abs(end_junction.x - start_x)
                height = wall_thickness
            else:  # vertical
                # ИСПРАВЛЕНИЕ: Убедимся, что стена начинается точно от junction
                # Для нижней стороны проема, всегда начинаем от junction
                if edge_side == 'bottom':
                    start_y = start_junction.y
                    # УБРАНО: Коррекция на 1 пиксель для избежания перекрытия на SVG
                    # start_y += 1.0
                    # ИСПРАВЛЕНИЕ: Для нижней стороны проема, end_junction должен быть скорректирован на offset_y
                    adjusted_end_y = end_junction.y + offset_y
                else:
                    # УБРАНО: Коррекция на 1 пиксель для верхней стороны проема
                    # start_y = start_junction.y - 1.0
                    start_y = start_junction.y
                    # ИСПРАВЛЕНИЕ: Для верхней стороны проема, end_junction должен быть скорректирован на offset_y
                    adjusted_end_y = end_junction.y + offset_y

                x = (start_junction.x - offset_x) - wall_thickness / 2
                y = min(start_y, adjusted_end_y)
                width = wall_thickness
                # ИСПРАВЛЕНИЕ: Используем скорректированную координату для end_junction
                height = abs(adjusted_end_y - start_y)
            
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
            
            # Добавляем сегмент в JSON
            if json_data:
                add_wall_segment_from_opening_to_json(json_data, wall_segment)
        else:
            # Нет подходящего junction: сначала пробуем край содержащего полигона, затем fallback
            print(f"    Нет подходящего junction — попытка дотянуть до края полигона (проем {opening_id}, {edge_side})")

            def find_containing_polygon(px: float, py: float) -> Optional[Dict]:
                if not wall_polygons:
                    return None
                for poly in wall_polygons:
                    verts = poly.get('vertices', [])
                    if verts and is_point_in_polygon(px, py, verts):
                        return poly
                return None

            end_coord = None
            inter_key = None
            # Для выбора полигона используем точку на краю проема, смещенную на половину толщины в нужную сторону
            # Это соответствует правилу "от двери на половину толщины стены".
            dir_map = {
                ('horizontal', 'left'): 'left',
                ('horizontal', 'right'): 'right',
                ('vertical', 'top'): 'up',
                ('vertical', 'bottom'): 'down',
            }
            inter_key = dir_map.get((orientation, edge_side))

            sample_x, sample_y = start_junction.x, start_junction.y
            if inter_key is not None:
                # Базовая точка на самом краю проема
                if direction == 'left':
                    base_x, base_y = bbox['x'], bbox['y'] + bbox['height'] / 2.0
                elif direction == 'right':
                    base_x, base_y = bbox['x'] + bbox['width'], bbox['y'] + bbox['height'] / 2.0
                elif direction == 'up':
                    base_x, base_y = bbox['x'] + bbox['width'] / 2.0, bbox['y']
                else:  # 'down'
                    base_x, base_y = bbox['x'] + bbox['width'] / 2.0, bbox['y'] + bbox['height']

                # Смещаемся на половину толщины в направлении сегмента
                half = wall_thickness / 2.0
                dx = dy = 0.0
                if inter_key == 'up':
                    dy = -half
                elif inter_key == 'down':
                    dy = half
                elif inter_key == 'left':
                    dx = -half
                elif inter_key == 'right':
                    dx = half
                sample_x, sample_y = base_x + dx, base_y + dy

            # 1) Пробуем полигон по смещенной от двери точке
            containing_poly = find_containing_polygon(sample_x, sample_y)
            if containing_poly is None:
                # 2) Если не нашли — пробуем по исходному junction
                containing_poly = find_containing_polygon(start_junction.x, start_junction.y)

            if containing_poly is not None and inter_key is not None:
                # Анализ пересечений для выбранной точки (лучи из смещенной от двери точки)
                analysis = analyze_polygon_extensions_with_thickness(
                    {'x': sample_x, 'y': sample_y},
                    containing_poly.get('vertices', []),
                    wall_thickness,
                )
                intersections = analysis.get('intersections', {})
                end_coord = intersections.get(inter_key)

            # Доп. попытка: сместиться от J на половину толщины в нужном направлении и поискать содержащий полигон
            if end_coord is None and wall_polygons and inter_key is not None:
                dx = dy = 0.0
                half = wall_thickness / 2.0
                if inter_key == 'up':
                    dy = -half
                elif inter_key == 'down':
                    dy = half
                elif inter_key == 'left':
                    dx = -half
                elif inter_key == 'right':
                    dx = half
                shifted_poly = find_containing_polygon(start_junction.x + dx, start_junction.y + dy)
                if shifted_poly is not None:
                    analysis2 = analyze_polygon_extensions_with_thickness(
                        {'x': start_junction.x + dx, 'y': start_junction.y + dy},
                        shifted_poly.get('vertices', []),
                        wall_thickness,
                    )
                    inter2 = analysis2.get('intersections', {})
                    if inter_key in inter2 and inter2[inter_key] is not None:
                        end_coord = inter2[inter_key]
                        print(f"    ✓ Найден край полигона после смещения от J на half-thickness: {inter_key}={end_coord:.1f}")

            if end_coord is None and wall_polygons and inter_key is not None:
                # Попробуем найти пересечение луча со всеми полигонами, даже если точка не внутри полигона,
                # используя в качестве базы смещенную от двери точку
                def ray_intersection_any_polygon(px: float, py: float, direction_key: str,
                                                 band_half: float = 0.0, samples: int = 1) -> Optional[float]:
                    eps = 1e-6
                    best = None
                    # Сканируем в поперечной полосе шириной band_half (±) для повышения устойчивости
                    offsets = [0.0]
                    if band_half > eps and samples > 1:
                        step = (2.0 * band_half) / (samples - 1)
                        offsets = [(-band_half + i * step) for i in range(samples)]

                    for off in offsets:
                        px_off, py_off = px, py
                        if direction_key in ('up', 'down'):
                            px_off = px + off  # двигаем по X поперёк вертикального луча
                        else:
                            py_off = py + off  # двигаем по Y поперёк горизонтального луча

                        for poly in wall_polygons:
                            verts = poly.get('vertices', [])
                            if not verts:
                                continue
                            n = len(verts)
                            for i in range(n):
                                x1, y1 = verts[i]['x'], verts[i]['y']
                                x2, y2 = verts[(i + 1) % n]['x'], verts[(i + 1) % n]['y']

                                if direction_key in ('up', 'down'):
                                    # Пересечение вертикальной прямой x=px_off с ребром (не вертикальным)
                                    if min(x1, x2) - eps <= px_off <= max(x1, x2) + eps and abs(x2 - x1) > eps:
                                        t = (px_off - x1) / (x2 - x1)
                                        if 0 - eps <= t <= 1 + eps:
                                            yi = y1 + t * (y2 - y1)
                                            if direction_key == 'up' and yi < py_off - eps:
                                                if best is None or yi > best:
                                                    best = yi
                                            elif direction_key == 'down' and yi > py_off + eps:
                                                if best is None or yi < best:
                                                    best = yi
                                else:  # left/right
                                    # Пересечение горизонтальной прямой y=py_off с ребром (не горизонтальным)
                                    if min(y1, y2) - eps <= py_off <= max(y1, y2) + eps and abs(y2 - y1) > eps:
                                        t = (py_off - y1) / (y2 - y1)
                                        if 0 - eps <= t <= 1 + eps:
                                            xi = x1 + t * (x2 - x1)
                                            if direction_key == 'left' and xi < px_off - eps:
                                                if best is None or xi > best:
                                                    best = xi
                                            elif direction_key == 'right' and xi > px_off + eps:
                                                if best is None or xi < best:
                                                    best = xi
                    return best

                end_coord = ray_intersection_any_polygon(
                    sample_x, sample_y, inter_key,
                    band_half=wall_thickness / 2.0, samples=5
                )

            if end_coord is not None:
                print(f"    ✓ Край полигона найден: {inter_key}={end_coord:.1f}. Строим до него")
                # Пересчитываем смещения относительно текущего bbox
                if orientation == 'horizontal':
                    if edge_side == 'left':
                        offset_x = start_junction.x - bbox['x']
                        offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
                    else:
                        offset_x = start_junction.x - (bbox['x'] + bbox['width'])
                        offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
                    start_x = start_junction.x
                    # Делаем сегмент на 1px длиннее в сторону направления
                    pixel_extend = 1.0
                    if inter_key == 'right':
                        end_x = end_coord + pixel_extend
                    elif inter_key == 'left':
                        end_x = end_coord - pixel_extend
                    else:
                        end_x = end_coord
                    synthetic_end = JunctionPoint(x=end_x, y=start_junction.y, junction_type=start_junction.junction_type, id=-(100000 + start_junction.id))
                    x = min(start_x, end_x)
                    y = (start_junction.y - offset_y) - wall_thickness / 2
                    width = abs(end_x - start_x)
                    height = wall_thickness
                else:
                    if edge_side == 'top':
                        offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                        offset_y = start_junction.y - bbox['y']
                    else:
                        offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                        offset_y = start_junction.y - (bbox['y'] + bbox['height'])
                    start_y = start_junction.y
                    # Делаем сегмент на 1px длиннее в сторону направления
                    pixel_extend = 1.0
                    if inter_key == 'down':
                        end_y = end_coord + pixel_extend
                    elif inter_key == 'up':
                        end_y = end_coord - pixel_extend
                    else:
                        end_y = end_coord
                    synthetic_end = JunctionPoint(x=start_junction.x, y=end_y, junction_type=start_junction.junction_type, id=-(100000 + start_junction.id))
                    x = (start_junction.x - offset_x) - wall_thickness / 2
                    adjusted_end_y = end_y + offset_y
                    y = min(start_y, adjusted_end_y)
                    width = wall_thickness
                    height = abs(adjusted_end_y - start_y)

                bbox_result = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'orientation': orientation
                }

                adjusted_start_junction = JunctionPoint(
                    x=start_junction.x - offset_x,
                    y=start_junction.y - offset_y,
                    junction_type=start_junction.junction_type,
                    id=start_junction.id
                )

                segment_id = f"wall_{opening_id}_{edge_side}_{start_junction.id}_to_{synthetic_end.id}_edge"
                wall_segment = WallSegmentFromOpening(
                    segment_id=segment_id,
                    opening_id=opening_id,
                    edge_side=edge_side,
                    start_junction=adjusted_start_junction,
                    end_junction=synthetic_end,
                    orientation=orientation,
                    bbox=bbox_result
                )
                wall_segments.append(wall_segment)
                if json_data:
                    add_wall_segment_from_opening_to_json(json_data, wall_segment)
            else:
                print(f"    Нет пересечения с краем полигона — строим fallback сегмент длиной wall_thickness")

                # Пересчитываем смещения относительно текущего bbox
                if orientation == 'horizontal':
                    if edge_side == 'left':
                        offset_x = start_junction.x - bbox['x']
                        offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
                    else:
                        offset_x = start_junction.x - (bbox['x'] + bbox['width'])
                        offset_y = start_junction.y - (bbox['y'] + bbox['height'] / 2)
                    start_x = start_junction.x
                    end_x = start_x + (-wall_thickness if edge_side == 'left' else wall_thickness)
                    synthetic_end = JunctionPoint(x=end_x, y=start_junction.y, junction_type=start_junction.junction_type, id=-(100000 + start_junction.id))
                    x = min(start_x, end_x)
                    y = (start_junction.y - offset_y) - wall_thickness / 2
                    width = abs(end_x - start_x)
                    height = wall_thickness
                else:
                    if edge_side == 'top':
                        offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                        offset_y = start_junction.y - bbox['y']
                    else:
                        offset_x = start_junction.x - (bbox['x'] + bbox['width'] / 2)
                        offset_y = start_junction.y - (bbox['y'] + bbox['height'])
                    start_y = start_junction.y
                    end_y = start_y + (-wall_thickness if edge_side == 'top' else wall_thickness)
                    synthetic_end = JunctionPoint(x=start_junction.x, y=end_y, junction_type=start_junction.junction_type, id=-(100000 + start_junction.id))
                    x = (start_junction.x - offset_x) - wall_thickness / 2
                    adjusted_end_y = end_y + offset_y
                    y = min(start_y, adjusted_end_y)
                    width = wall_thickness
                    height = abs(adjusted_end_y - start_y)

                bbox_result = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'orientation': orientation
                }

                adjusted_start_junction = JunctionPoint(
                    x=start_junction.x - offset_x,
                    y=start_junction.y - offset_y,
                    junction_type=start_junction.junction_type,
                    id=start_junction.id
                )

                segment_id = f"wall_{opening_id}_{edge_side}_{start_junction.id}_to_{synthetic_end.id}_fallback"
                wall_segment = WallSegmentFromOpening(
                    segment_id=segment_id,
                    opening_id=opening_id,
                    edge_side=edge_side,
                    start_junction=adjusted_start_junction,
                    end_junction=synthetic_end,
                    orientation=orientation,
                    bbox=bbox_result
                )
                wall_segments.append(wall_segment)

                if json_data:
                    add_wall_segment_from_opening_to_json(json_data, wall_segment)
    
    return wall_segments

def process_openings_with_junctions(data: Dict, wall_thickness: float, json_data: Dict = None) -> List[WallSegmentFromOpening]:
    """
    Main function that processes all openings and constructs wall segments.
    
    Args:
        data: Dictionary containing openings and junctions
        wall_thickness: Thickness of walls
        json_data: JSON структура для сохранения данных
    
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

        print(f"  Вызов construct_wall_segment_from_opening для проема {opening_id}")
        # Construct wall segments from this opening
        wall_segments = construct_wall_segment_from_opening(
            opening_with_junction, junctions, wall_thickness, openings, json_data, data.get('wall_polygons', [])
        )

        # Добавляем проем в JSON с обновленным bbox после обработки
        if json_data:
            # Используем обновленный bbox из opening_with_junction
            opening_updated = opening.copy()
            opening_updated['bbox'] = opening_with_junction.bbox
            add_opening_to_json(json_data, opening_updated, edge_junctions)

        all_wall_segments.extend(wall_segments)
        print(f"  ✓ Проем {opening_id} ({orientation}): {len(wall_segments)} сегментов")
    
    print(f"\nВсего создано {len(all_wall_segments)} сегментов стен из {len(openings)} проемов")
    return all_wall_segments

def process_pillars_to_squares(data: Dict, wall_thickness: float, json_data: Dict = None) -> None:
    """
    Обрабатывает колонны из исходных данных и создает квадраты для них в JSON
    
    Args:
        data: Исходные данные с колоннами
        wall_thickness: Толщина стены для определения размера квадратов
        json_data: JSON структура для сохранения данных
    """
    print(f"\n{'='*60}")
    print("ОБРАБОТКА КОЛОНН И СОЗДАНИЕ КВАДРАТОВ")
    print(f"{'='*60}")
    
    # Получаем колонны из разных разделов
    pillar_polygons = data.get('pillar_polygons', [])
    pillars = data.get('pillars', [])
    
    print(f"  Найдено {len(pillar_polygons)} полигонов колонн")
    print(f"  Найдено {len(pillars)} колонн")
    
    # Используем только pillar_polygons для избежания дублирования
    # Так как pillars содержит те же данные, что и pillar_polygons
    print(f"  Используем только pillar_polygons для избежания дублирования")
    
    # Обрабатываем полигоны колонн
    for idx, polygon in enumerate(pillar_polygons):
        polygon_id = polygon.get('id', f'pillar_polygon_{idx+1}')
        vertices = polygon.get('vertices', [])
        
        if not vertices:
            print(f"  ✗ Полигон {polygon_id} не имеет вершин")
            continue
        
        # Вычисляем центр полигона
        center_x = sum(v['x'] for v in vertices) / len(vertices)
        center_y = sum(v['y'] for v in vertices) / len(vertices)
        
        # Создаем квадрат колонны
        pillar_id = f"P{idx+1}"
        add_pillar_square_to_json(json_data, pillar_id, center_x, center_y, wall_thickness, vertices)
        
        print(f"  ✓ Создан квадрат для колонны {pillar_id}: центр({center_x:.1f}, {center_y:.1f}), размер {wall_thickness}x{wall_thickness}")
    
    print(f"\n  ✓ Всего создано квадратов колонн: {json_data['statistics']['total_pillar_squares']}")

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

def segment_exists(start_junction: JunctionPoint, end_junction: JunctionPoint, existing_segments: List[Union[WallSegmentFromOpening, WallSegmentFromJunction]]) -> bool:
    """
    Проверяет, существует ли уже сегмент между двумя junctions
    
    Args:
        start_junction: Начальный junction
        end_junction: Конечный junction
        existing_segments: Список существующих сегментов
        
    Returns:
        True если сегмент уже существует, иначе False
    """
    for segment in existing_segments:
        # Проверяем прямое направление
        if (segment.start_junction.id == start_junction.id and
            segment.end_junction.id == end_junction.id):
            return True
        # Проверяем обратное направление
        if (segment.start_junction.id == end_junction.id and
            segment.end_junction.id == start_junction.id):
            return True
    return False

def build_missing_wall_segments_for_junctions(junctions: List[JunctionPoint],
                                             wall_segments: List[WallSegmentFromOpening],
                                             junction_wall_segments: List[WallSegmentFromJunction],
                                             all_junctions: List[JunctionPoint],
                                             wall_thickness: float,
                                             json_data: Dict = None,
                                             wall_polygons: List[Dict] = None) -> List[WallSegmentFromJunction]:
    """
    Достраивает недостающие сегменты стен для junctions не связанных с проемами
    
    Args:
        junctions: Список junctions не связанных с проемами
        wall_segments: Существующие сегменты стен из проемов
        junction_wall_segments: Существующие сегменты стен из junctions
        all_junctions: Список всех junctions
        wall_thickness: Толщина стен
        json_data: JSON структура для сохранения данных
    
    Returns:
        Список новых сегментов стен
    """
    new_segments = []

    # Вспомогательные функции для поиска полигона и пересечения в направлении
    def find_containing_polygon_for_point(px: float, py: float) -> Optional[Dict]:
        if not wall_polygons:
            return None
        for wall in wall_polygons:
            vertices = wall.get('vertices', [])
            if vertices and is_point_in_polygon(px, py, vertices):
                return wall
        return None

    def get_polygon_edge_intersection(j: JunctionPoint, direction: str) -> Optional[float]:
        polygon = find_containing_polygon_for_point(j.x, j.y)
        if not polygon:
            return None
        analysis = analyze_polygon_extensions_with_thickness({'x': j.x, 'y': j.y}, polygon['vertices'], wall_thickness)
        intersections = analysis.get('intersections', {})
        return intersections.get(direction)

    def try_bridge_small_gap(junc: JunctionPoint, dirn: str) -> bool:
        """
        Пытается вытянуть ближайший bbox стены, ориентированный в указанную сторону,
        если зазор <= толщины стены. Меняет только длину сегмента, толщину не меняет.
        """
        align_tol = wall_thickness * 0.75
        max_gap = wall_thickness * 1.25
        best = None  # (gap, segment, orientation)

        def aligned_x(seg):
            return abs((seg.bbox['x'] + seg.bbox['width'] / 2) - junc.x) <= align_tol

        def aligned_y(seg):
            return abs((seg.bbox['y'] + seg.bbox['height'] / 2) - junc.y) <= align_tol

        # Собираем кандидатов по ориентации и направлению
        all_segs = wall_segments + junction_wall_segments
        for seg in all_segs:
            if dirn in ['up', 'down'] and seg.orientation == 'vertical' and aligned_x(seg):
                top = seg.bbox['y']
                bottom = seg.bbox['y'] + seg.bbox['height']
                if dirn == 'down':
                    # Сегмент ниже junction: верхняя грань >= j.y
                    if top >= junc.y:
                        gap = top - junc.y
                        if 0 <= gap <= max_gap:
                            if best is None or gap < best[0]:
                                best = (gap, seg, 'vertical')
                else:  # up
                    # Сегмент выше junction: нижняя грань <= j.y
                    if bottom <= junc.y:
                        gap = junc.y - bottom
                        if 0 <= gap <= max_gap:
                            if best is None or gap < best[0]:
                                best = (gap, seg, 'vertical')
            elif dirn in ['left', 'right'] and seg.orientation == 'horizontal' and aligned_y(seg):
                left = seg.bbox['x']
                right = seg.bbox['x'] + seg.bbox['width']
                if dirn == 'right':
                    # Сегмент справа от junction: левая грань >= j.x
                    if left >= junc.x:
                        gap = left - junc.x
                        if 0 <= gap <= max_gap:
                            if best is None or gap < best[0]:
                                best = (gap, seg, 'horizontal')
                else:  # left
                    # Сегмент слева от junction: правая грань <= j.x
                    if right <= junc.x:
                        gap = junc.x - right
                        if 0 <= gap <= max_gap:
                            if best is None or gap < best[0]:
                                best = (gap, seg, 'horizontal')

        if best is None:
            return False

        gap, seg, orient = best
        original = seg.bbox.copy()
        if orient == 'vertical':
            # Продлеваем вертикальный сегмент по Y-только
            if dirn == 'down':
                # Поднять верхнюю грань до j.y
                new_y = junc.y
                seg.bbox['height'] = (seg.bbox['y'] + seg.bbox['height']) - new_y
                seg.bbox['y'] = new_y
            else:  # up
                # Опустить нижнюю грань до j.y
                bottom = seg.bbox['y'] + seg.bbox['height']
                seg.bbox['height'] = junc.y - seg.bbox['y']
                # y не меняется
            print(f"    ✓ Вытянули вертикальный сегмент {seg.segment_id} на {gap:.1f}px к J{junc.id} ({dirn}); bbox {original} -> {seg.bbox}")
            return True
        else:
            # Продлеваем горизонтальный сегмент по X-только
            if dirn == 'right':
                # Сдвигаем левую грань до j.x
                right = seg.bbox['x'] + seg.bbox['width']
                seg.bbox['width'] = right - junc.x
                seg.bbox['x'] = junc.x
            else:  # left
                # Сдвигаем правую грань до j.x
                seg.bbox['width'] = junc.x - seg.bbox['x']
                # x не меняется
            print(f"    ✓ Вытянули горизонтальный сегмент {seg.segment_id} на {gap:.1f}px к J{junc.id} ({dirn}); bbox {original} -> {seg.bbox}")
            return True
    
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
            # Перебираем кандидатов по близости, пока не найдем подходящего
            candidates = find_candidate_junctions_in_direction(
                junction, direction, all_junctions, wall_thickness / 2.0
            )
            created = False

            # Локальный предел по полигону: не соединять через разрыв полигона
            polygon_edge_coord = get_polygon_edge_intersection(junction, direction)
            for next_junction in candidates:
                # Проверяем, не существует ли уже сегмент между этими junctions
                all_existing_segments = wall_segments + junction_wall_segments + new_segments
                if segment_exists(junction, next_junction, all_existing_segments):
                    print(f"    ✗ Сегмент между J{junction.id} и J{next_junction.id} уже существует, пропускаем")
                    continue

                # Дополнительная проверка: убедимся, что next_junction также логично соединять
                next_required_directions = next_junction.directions if next_junction.directions else []

                # Определяем противоположное направление
                opposite_direction = {
                    'left': 'right',
                    'right': 'left',
                    'up': 'down',
                    'down': 'up'
                }.get(direction, direction)

                # Приоритет: если у next_junction есть противоположное направление — отлично
                # Иначе допускаем, если у него есть любая значимая компонента по той же оси (горизонталь/вертикаль)
                same_axis_ok = False
                if direction in ['left', 'right']:
                    same_axis_ok = any(d in next_required_directions for d in ['left', 'right'])
                else:
                    same_axis_ok = any(d in next_required_directions for d in ['up', 'down'])

                if opposite_direction not in next_required_directions and not same_axis_ok:
                    print(f"    ✗ Кандидат J{next_junction.id} не подтверждает соединение (нет {opposite_direction} и нет осевой компоненты), пробуем следующего")
                    continue

                # Проверка: не пересекаем ли границу текущего полигона по пути (ограничиваемся краем полигона)
                if polygon_edge_coord is not None:
                    if direction == 'right' and next_junction.x > polygon_edge_coord:
                        print(f"    ✗ Кандидат J{next_junction.id} за пределами полигона (right до {polygon_edge_coord:.1f}), пробуем следующего")
                        continue
                    if direction == 'left' and next_junction.x < polygon_edge_coord:
                        print(f"    ✗ Кандидат J{next_junction.id} за пределами полигона (left до {polygon_edge_coord:.1f}), пробуем следующего")
                        continue
                    if direction == 'down' and next_junction.y > polygon_edge_coord:
                        print(f"    ✗ Кандидат J{next_junction.id} за пределами полигона (down до {polygon_edge_coord:.1f}), пробуем следующего")
                        continue
                    if direction == 'up' and next_junction.y < polygon_edge_coord:
                        print(f"    ✗ Кандидат J{next_junction.id} за пределами полигона (up до {polygon_edge_coord:.1f}), пробуем следующего")
                        continue

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

                # Добавляем сегмент в JSON
                if json_data:
                    add_wall_segment_from_junction_to_json(json_data, new_segment)

                print(f"    ✓ Создан сегмент от J{junction.id} до J{next_junction.id} в направлении {direction}")
                created = True
                break

            if not created:
                # Попытка: вытянуть ближайший bbox стены, ориентированный в эту сторону
                # (увеличиваем длину, не меняя толщину)
                def try_bridge_small_gap(junc: JunctionPoint, dirn: str) -> bool:
                    align_tol = wall_thickness / 2.0
                    max_gap = wall_thickness
                    best = None  # (gap, segment, orientation)

                    def aligned_x(seg):
                        return abs((seg.bbox['x'] + seg.bbox['width'] / 2) - junc.x) <= align_tol

                    def aligned_y(seg):
                        return abs((seg.bbox['y'] + seg.bbox['height'] / 2) - junc.y) <= align_tol

                    # Собираем кандидатов по ориентации и направлению
                    all_segs = wall_segments + junction_wall_segments
                    for seg in all_segs:
                        if dirn in ['up', 'down'] and seg.orientation == 'vertical' and aligned_x(seg):
                            top = seg.bbox['y']
                            bottom = seg.bbox['y'] + seg.bbox['height']
                            if dirn == 'down':
                                # Сегмент ниже junction: верхняя грань >= j.y
                                if top >= junc.y:
                                    gap = top - junc.y
                                    if 0 <= gap <= max_gap:
                                        if best is None or gap < best[0]:
                                            best = (gap, seg, 'vertical')
                            else:  # up
                                # Сегмент выше junction: нижняя грань <= j.y
                                if bottom <= junc.y:
                                    gap = junc.y - bottom
                                    if 0 <= gap <= max_gap:
                                        if best is None or gap < best[0]:
                                            best = (gap, seg, 'vertical')
                        elif dirn in ['left', 'right'] and seg.orientation == 'horizontal' and aligned_y(seg):
                            left = seg.bbox['x']
                            right = seg.bbox['x'] + seg.bbox['width']
                            if dirn == 'right':
                                # Сегмент справа от junction: левая грань >= j.x
                                if left >= junc.x:
                                    gap = left - junc.x
                                    if 0 <= gap <= max_gap:
                                        if best is None or gap < best[0]:
                                            best = (gap, seg, 'horizontal')
                            else:  # left
                                # Сегмент слева от junction: правая грань <= j.x
                                if right <= junc.x:
                                    gap = junc.x - right
                                    if 0 <= gap <= max_gap:
                                        if best is None or gap < best[0]:
                                            best = (gap, seg, 'horizontal')

                    if best is None:
                        return False

                    gap, seg, orient = best
                    original = seg.bbox.copy()
                    if orient == 'vertical':
                        # Продлеваем вертикальный сегмент по Y-только
                        if dirn == 'down':
                            # Поднять верхнюю грань до j.y
                            new_y = junc.y
                            seg.bbox['height'] = (seg.bbox['y'] + seg.bbox['height']) - new_y
                            seg.bbox['y'] = new_y
                        else:  # up
                            # Опустить нижнюю грань до j.y
                            bottom = seg.bbox['y'] + seg.bbox['height']
                            seg.bbox['height'] = junc.y - seg.bbox['y']
                            # y не меняется
                        print(f"    ✓ Вытянули вертикальный сегмент {seg.segment_id} на {gap:.1f}px к J{junc.id} ({dirn}); bbox {original} -> {seg.bbox}")
                        return True
                    else:
                        # Продлеваем горизонтальный сегмент по X-только
                        if dirn == 'right':
                            # Сдвигаем левую грань до j.x
                            right = seg.bbox['x'] + seg.bbox['width']
                            seg.bbox['width'] = right - junc.x
                            seg.bbox['x'] = junc.x
                        else:  # left
                            # Сдвигаем правую грань до j.x
                            seg.bbox['width'] = junc.x - seg.bbox['x']
                            # x не меняется
                        print(f"    ✓ Вытянули горизонтальный сегмент {seg.segment_id} на {gap:.1f}px к J{junc.id} ({dirn}); bbox {original} -> {seg.bbox}")
                        return True

                if try_bridge_small_gap(junction, direction):
                    created = True
                # Если не удалось вытянуть ближайший bbox, и если от junction действительно идет полигон,
                # продолжаем до края полигона, создавая сегмент до границы
                elif polygon_edge_coord is not None:
                    print(f"    → Нет подходящего узла; создаем сегмент до края полигона в направлении {direction}")

                    # Создаем виртуальную конечную точку на границе полигона
                    if direction in ['left', 'right']:
                        end_x = polygon_edge_coord
                        end_y = junction.y
                        orientation = 'horizontal'
                    else:
                        end_x = junction.x
                        end_y = polygon_edge_coord
                        orientation = 'vertical'

                    # Генерируем уникальный ID для виртуального узла на границе полигона
                    # Используем текущее количество узлов в json_data, если доступно
                    virtual_id = None
                    if json_data and isinstance(json_data.get('junctions'), list):
                        existing_ids = [j.get('id') for j in json_data['junctions'] if isinstance(j, dict) and 'id' in j]
                        max_id = max(existing_ids) if existing_ids else 0
                        virtual_id = max_id + 1
                    else:
                        # Запасной вариант: отталкиваться от max id среди all_junctions
                        max_id = max([j.id for j in all_junctions]) if all_junctions else 0
                        virtual_id = max_id + 1

                    virtual_end = JunctionPoint(
                        x=end_x,
                        y=end_y,
                        junction_type='polygon_edge',
                        id=virtual_id,
                        detected_type='wall_end',
                        directions=[],
                        confidence=1.0,
                    )

                    # Добавляем виртуальный узел в JSON, чтобы привязки работали корректно
                    if json_data is not None:
                        add_junction_to_json(json_data, virtual_end)
                    # Включаем виртуальный узел в общий список узлов для последующих проходов (бриджинг и т.п.)
                    try:
                        all_junctions.append(virtual_end)
                    except Exception:
                        pass

                    # Создаем bbox для сегмента до края полигона
                    bbox = create_bbox_from_junctions(junction, virtual_end, orientation, wall_thickness)

                    segment_id = f"wall_junction_{junction.id}_to_edge_{direction}"
                    edge_segment = WallSegmentFromJunction(
                        segment_id=segment_id,
                        start_junction=junction,
                        end_junction=virtual_end,
                        direction=direction,
                        orientation=orientation,
                        bbox=bbox,
                    )
                    new_segments.append(edge_segment)

                    if json_data is not None:
                        add_wall_segment_from_junction_to_json(json_data, edge_segment)

                    print(f"    ✓ Создан сегмент от J{junction.id} до края полигона ({direction})")
                else:
                    print(f"    ✗ Не удалось подобрать подходящий узел для J{junction.id} в направлении {direction}")

        # Дополнительный проход: если по какому-либо направлению (включая те, что не были 'значимыми')
        # есть малый зазор <= толщины стены, пытаемся вытянуть ближайший ориентированный сегмент
        # Это покрывает случай J19→down, где gap < wall_thickness
        for direction_all in ['left', 'right', 'up', 'down']:
            # Пропускаем, если уже существует сегмент в этом направлении
            existing_directions = get_existing_directions_for_junction(junction, wall_segments, junction_wall_segments, wall_thickness)
            if direction_all in existing_directions:
                continue
            if try_bridge_small_gap(junction, direction_all):
                # После успешного вытягивания обновляем existing_directions и продолжаем к следующему направлению
                continue
    
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

def override_junction_types_with_connectivity(
    junctions: List[JunctionPoint],
    wall_segments: List[WallSegmentFromOpening],
    junction_wall_segments: List[WallSegmentFromJunction],
    wall_thickness: float,
) -> List[JunctionPoint]:
    """
    Если полигонный анализ дал unknown или только 1 направление, смотреть подключенные к узлу сегменты;
    если есть по одному горизонтальному и вертикальному — маркировать как L.

    Args:
        junctions: Узлы с типами по полигонному анализу
        wall_segments: Сегменты стен от проёмов
        junction_wall_segments: Сегменты стен между узлами
        wall_thickness: Толщина стены (для допусков)

    Returns:
        Обновлённый список узлов
    """
    updated = 0
    for j in junctions:
        # Только если ещё не L и информации по полигонам недостаточно
        if j.detected_type != 'L' and (not j.directions or len(j.directions) <= 1):
            # Собираем подключенные сегменты около узла
            opening_segs, junction_segs = find_wall_segments_at_l_junction(
                j, wall_segments, junction_wall_segments, wall_thickness / 2.0
            )
            all_segs = opening_segs + junction_segs
            if not all_segs:
                continue
            horiz = sum(1 for s in all_segs if getattr(s, 'orientation', None) == 'horizontal')
            vert = sum(1 for s in all_segs if getattr(s, 'orientation', None) == 'vertical')

            # Если есть хотя бы один горизонтальный и один вертикальный сегмент — это L
            if horiz >= 1 and vert >= 1:
                prev = j.detected_type
                j.detected_type = 'L'
                # Обновляем направления по существующим сегментам, чтобы расширения знали, куда тянуть
                try:
                    j.directions = get_existing_directions_for_junction(
                        j, wall_segments, junction_wall_segments, wall_thickness
                    )
                except Exception:
                    # В крайнем случае оставляем как есть
                    pass
                print(f"  ✓ Переопределение типа J{j.id}: {prev} -> L (по подключенным сегментам: H={horiz}, V={vert})")
                updated += 1
    if updated:
        print(f"  ✓ Переопределили тип на L по связям: {updated} узлов")
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
                                 wall_thickness: float) -> bool:
    """
    Измененная функция: расширяет оригинальный сегмент стены до края содержащего его полигона
    
    Args:
        segment: The wall segment to extend (будет изменен)
        l_junction: The L-junction point
        wall_polygons: List of wall polygons
        wall_thickness: Wall thickness for calculations
        
    Returns:
        True если сегмент был расширен, иначе False
    """
    # Add debug output for J10
    if l_junction.id == 10:
        print(f"    DEBUG J10: extend_segment_to_polygon_edge called for {segment.segment_id}")
        print(f"    DEBUG J10: segment bbox before={segment.bbox}")
        print(f"    DEBUG J10: junction directions={l_junction.directions}")
    
    # Find the polygon containing the L-junction
    containing_polygon = None
    for wall in wall_polygons:
        vertices = wall.get('vertices', [])
        if vertices and is_point_in_polygon(l_junction.x, l_junction.y, vertices):
            containing_polygon = wall
            break
    
    if not containing_polygon:
        if l_junction.id == 10:
            print(f"    DEBUG J10: No containing polygon found")
        return False
    
    # Get the direction of extension based on junction directions
    directions = l_junction.directions
    if not directions:
        if l_junction.id == 10:
            print(f"    DEBUG J10: No directions found")
        return False
    
    # Determine which direction to extend (choose the first available)
    extend_direction = directions[0]
    
    if l_junction.id == 10:
        print(f"    DEBUG J10: extend_direction={extend_direction}")
    
    # Get the polygon edge intersection in that direction
    analysis = analyze_polygon_extensions_with_thickness(
        {'x': l_junction.x, 'y': l_junction.y},
        containing_polygon['vertices'],
        wall_thickness
    )
    
    intersections = analysis['intersections']
    
    if l_junction.id == 10:
        print(f"    DEBUG J10: intersections={intersections}")
    
    # Modify the original segment bbox based on direction
    if extend_direction in intersections and intersections[extend_direction] is not None:
        original_bbox = segment.bbox.copy()  # Сохраняем оригинал для отладки
        
        if segment.orientation == 'horizontal':
            if extend_direction == 'left':
                # Extend to the left
                new_x = intersections[extend_direction]
                new_width = segment.bbox['x'] + segment.bbox['width'] - new_x
                segment.bbox['x'] = new_x
                segment.bbox['width'] = new_width
                print(f"    ✓ Сегмент {segment.segment_id} расширен влево до {new_x}")
                if l_junction.id == 10:
                    print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
                return True
            elif extend_direction == 'right':
                # Extend to the right
                new_width = intersections[extend_direction] - segment.bbox['x']
                segment.bbox['width'] = new_width
                print(f"    ✓ Сегмент {segment.segment_id} расширен вправо до {intersections[extend_direction]}")
                if l_junction.id == 10:
                    print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
                return True
        else:  # vertical
            if extend_direction == 'up':
                # Extend upward
                new_y = intersections[extend_direction]
                new_height = segment.bbox['y'] + segment.bbox['height'] - new_y
                segment.bbox['y'] = new_y
                segment.bbox['height'] = new_height
                print(f"    ✓ Сегмент {segment.segment_id} расширен вверх до {new_y}")
                if l_junction.id == 10:
                    print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
                return True
            elif extend_direction == 'down':
                # Extend downward
                new_height = intersections[extend_direction] - segment.bbox['y']
                segment.bbox['height'] = new_height
                print(f"    ✓ Сегмент {segment.segment_id} расширен вниз до {intersections[extend_direction]}")
                if l_junction.id == 10:
                    print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
                return True
    
    if l_junction.id == 10:
        print(f"    DEBUG J10: No extension performed")
    return False

def extend_segment_to_perpendicular_x(segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                     perpendicular_segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                     l_junction: JunctionPoint,
                                     overshoot: float = 0.0) -> bool:
    """
    Измененная функция: расширяет оригинальный сегмент стены до координаты перпендикулярного сегмента
    
    Args:
        segment: The wall segment to extend (будет изменен)
        perpendicular_segment: The perpendicular wall segment
        l_junction: The L-junction point
        
    Returns:
        True если сегмент был расширен, иначе False
    """
    # Add debug output for J10
    if l_junction.id == 10:
        print(f"    DEBUG J10: segment={segment.segment_id}, perpendicular={perpendicular_segment.segment_id}")
        print(f"    DEBUG J10: segment bbox={segment.bbox}")
        print(f"    DEBUG J10: perpendicular bbox={perpendicular_segment.bbox}")
    
    # Сохраняем оригинальные границы для отладки
    original_bbox = segment.bbox.copy()

    # Выбор целевой стороны bbox перпендикулярной стены по направлению от узла
    def target_x_by_direction(direction: str, rect: Dict[str, float]) -> float:
        left = rect['x']
        right = rect['x'] + rect['width']
        return left if direction == 'left' else right

    def target_y_by_direction(direction: str, rect: Dict[str, float]) -> float:
        top = rect['y']
        bottom = rect['y'] + rect['height']
        return top if direction == 'up' else bottom

    # Выбираем цель в зависимости от ориентаций (тянем до дальней стороны bbox)
    if segment.orientation == 'horizontal' and perpendicular_segment.orientation == 'vertical':
        seg_left = segment.bbox['x']
        seg_right = segment.bbox['x'] + segment.bbox['width']
        # Направление: если узел ближе к правой грани сегмента — тянем вправо, иначе влево
        direction = 'right' if abs(l_junction.x - seg_right) <= abs(l_junction.x - seg_left) else 'left'
        target = target_x_by_direction(direction, perpendicular_segment.bbox)

        if direction == 'left' and target < seg_left - 1e-6:
            new_x = target - overshoot
            segment.bbox['x'] = new_x
            segment.bbox['width'] = seg_right - new_x
            print(f"    ✓ Сегмент {segment.segment_id} расширен влево до {target} (bbox перпендикуляра)")
            if l_junction.id == 10:
                print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
            return True
        elif direction == 'right' and target > seg_right + 1e-6:
            new_width = (target - seg_left) + overshoot
            segment.bbox['width'] = new_width
            print(f"    ✓ Сегмент {segment.segment_id} расширен вправо до {target} (bbox перпендикуляра)")
            if l_junction.id == 10:
                print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
            return True
        else:
            if l_junction.id == 10:
                print("    DEBUG J10: target X within span or no extension needed")
            return False

    if segment.orientation == 'vertical' and perpendicular_segment.orientation == 'horizontal':
        seg_top = segment.bbox['y']
        seg_bottom = segment.bbox['y'] + segment.bbox['height']
        direction = 'down' if abs(l_junction.y - seg_bottom) <= abs(l_junction.y - seg_top) else 'up'
        target = target_y_by_direction(direction, perpendicular_segment.bbox)

        if direction == 'up' and target < seg_top - 1e-6:
            new_y = target - overshoot
            segment.bbox['y'] = new_y
            segment.bbox['height'] = seg_bottom - new_y
            print(f"    ✓ Сегмент {segment.segment_id} расширен вверх до {target} (bbox перпендикуляра)")
            if l_junction.id == 10:
                print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
            return True
        elif direction == 'down' and target > seg_bottom + 1e-6:
            new_height = (target - seg_top) + overshoot
            segment.bbox['height'] = new_height
            print(f"    ✓ Сегмент {segment.segment_id} расширен вниз до {target} (bbox перпендикуляра)")
            if l_junction.id == 10:
                print(f"    DEBUG J10: bbox changed from {original_bbox} to {segment.bbox}")
            return True
        else:
            if l_junction.id == 10:
                print("    DEBUG J10: target Y within span or no extension needed")
            return False

    if l_junction.id == 10:
        print(f"    DEBUG J10: No perpendicular extension performed - orientations: {segment.orientation} vs {perpendicular_segment.orientation}")
    return False

def process_l_junction_extensions(junctions: List[JunctionPoint],
                                wall_segments: List[WallSegmentFromOpening],
                                junction_wall_segments: List[WallSegmentFromJunction],
                                wall_polygons: List[Dict],
                                wall_thickness: float) -> int:
    """
    Измененная функция: обрабатывает все L-junctions и расширяет оригинальные сегменты стен
    
    Args:
        junctions: List of all junction points (with detected_type already set)
        wall_segments: List of wall segments from openings
        junction_wall_segments: List of wall segments from junctions
        wall_polygons: List of wall polygons
        wall_thickness: Wall thickness for calculations
        
    Returns:
        Количество расширенных сегментов
    """
    extended_count = 0
    
    # Find all L-junctions using existing detection
    l_junctions = find_l_junctions(junctions)
    
    if not l_junctions:
        print("  ✓ L-junctions не найдены")
        return extended_count
    
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
        
        # Выбираем сегмент для расширения: предпочитаем сегмент от проема,
        # чтобы закрыть соединение проем ↔ перпендикулярная стена
        selected_segment = None
        for s in opening_segments:
            selected_segment = s
            break
        if selected_segment is None:
            selected_segment = all_connected_segments[0]
        original_bbox = selected_segment.bbox.copy()
        print(f"    ✓ Выбран сегмент для расширения: {selected_segment.segment_id}")
        print(f"    ✓ Оригинальные границы segment bbox: {original_bbox}")
        
        # Новый порядок: сначала тянем до перпендикулярного bbox (дальняя сторона),
        # затем при необходимости — до границы полигона как запасной вариант.
        extended_to_edge = False
        extended_to_perpendicular = False

        # Ищем перпендикулярный сегмент: сначала среди подключенных,
        # если не нашли — среди всех сегментов по выравниванию
        perpendicular_segment = None
        for s in all_connected_segments:
            if s is selected_segment:
                continue
            if s.orientation != selected_segment.orientation:
                perpendicular_segment = s
                break
        if perpendicular_segment is None:
            # Поиск среди всех сегментов по выравниванию относительно узла
            align_tol = wall_thickness * 0.75
            all_segs = wall_segments + junction_wall_segments
            if selected_segment.orientation == 'horizontal':
                # Ищем вертикальный соосный по X
                for s in all_segs:
                    if s.orientation == 'vertical' and abs((s.bbox['x'] + s.bbox['width']/2) - l_junction.x) <= align_tol:
                        perpendicular_segment = s
                        break
            else:
                # Ищем горизонтальный соосный по Y
                for s in all_segs:
                    if s.orientation == 'horizontal' and abs((s.bbox['y'] + s.bbox['height']/2) - l_junction.y) <= align_tol:
                        perpendicular_segment = s
                        break

        if perpendicular_segment is not None:
            print(f"    ✓ Выбран перпендикулярный сегмент: {perpendicular_segment.segment_id}")
            print(f"    → Шаг 1: Расширение до перпендикулярного bbox (дальняя сторона)...")

            extended_to_perpendicular = extend_segment_to_perpendicular_x(
                selected_segment, perpendicular_segment, l_junction
            )

            if extended_to_perpendicular:
                extended_count += 1
                print(f"    ✓ После расширения до перпендикуляра: {selected_segment.bbox}")

        # Если перпендикулярный сегмент не дал результата — пробуем границу полигона
        if not extended_to_perpendicular:
            print(f"    → Шаг 2: Расширение до границы полигона (fallback)...")
            extended_to_edge = extend_segment_to_polygon_edge(
                selected_segment, l_junction, wall_polygons, wall_thickness
            )
            if extended_to_edge:
                extended_count += 1
                print(f"    ✓ После расширения до границы полигона: {selected_segment.bbox}")
        
        # Итоговая информация о расширении
        if extended_to_perpendicular or extended_to_edge:
            print(f"    ✓ Итог: Сегмент {selected_segment.segment_id} расширен")
            print(f"      оригинал: {original_bbox}")
            print(f"      результат: {selected_segment.bbox}")
        else:
            print(f"    ✗ Сегмент {selected_segment.segment_id} не был расширен")
    
    print(f"  ✓ Расширено {extended_count} сегментов для L-junctions")
    return extended_count

def bridge_small_gaps_around_all_junctions(junctions: List[JunctionPoint],
                                          wall_segments: List[WallSegmentFromOpening],
                                          junction_wall_segments: List[WallSegmentFromJunction],
                                          wall_thickness: float) -> int:
    """
    Проход по всем junctions: если между junction и ближайшей соосной стеной
    зазор <= толщины стены, вытягивает ближайший bbox стены по длине (толщина не меняется).

    Возвращает количество модифицированных сегментов.
    """
    changed = 0
    align_tol = wall_thickness * 0.75
    max_gap = wall_thickness * 1.25

    def aligned_x(seg, jx):
        return abs((seg.bbox['x'] + seg.bbox['width'] / 2) - jx) <= align_tol

    def aligned_y(seg, jy):
        return abs((seg.bbox['y'] + seg.bbox['height'] / 2) - jy) <= align_tol

    all_segs = wall_segments + junction_wall_segments

    for j in junctions:
        for dirn in ['left', 'right', 'up', 'down']:

            best = None  # (gap, seg, orientation)
            for seg in all_segs:
                if dirn in ['up', 'down'] and seg.orientation == 'vertical' and aligned_x(seg, j.x):
                    top = seg.bbox['y']
                    bottom = seg.bbox['y'] + seg.bbox['height']
                    if dirn == 'down' and top >= j.y:
                        gap = top - j.y
                        if 0 <= gap <= max_gap and (best is None or gap < best[0]):
                            best = (gap, seg, 'vertical')
                    elif dirn == 'up' and bottom <= j.y:
                        gap = j.y - bottom
                        if 0 <= gap <= max_gap and (best is None or gap < best[0]):
                            best = (gap, seg, 'vertical')
                elif dirn in ['left', 'right'] and seg.orientation == 'horizontal' and aligned_y(seg, j.y):
                    left = seg.bbox['x']
                    right = seg.bbox['x'] + seg.bbox['width']
                    if dirn == 'right' and left >= j.x:
                        gap = left - j.x
                        if 0 <= gap <= max_gap and (best is None or gap < best[0]):
                            best = (gap, seg, 'horizontal')
                    elif dirn == 'left' and right <= j.x:
                        gap = j.x - right
                        if 0 <= gap <= max_gap and (best is None or gap < best[0]):
                            best = (gap, seg, 'horizontal')

            if best is None:
                continue

            gap, seg, orient = best
            original = seg.bbox.copy()
            if orient == 'vertical':
                if dirn == 'down':
                    new_y = j.y
                    seg.bbox['height'] = (seg.bbox['y'] + seg.bbox['height']) - new_y
                    seg.bbox['y'] = new_y
                else:  # up
                    seg.bbox['height'] = j.y - seg.bbox['y']
                print(f"    ✓ [BRIDGE] Вытянули вертикальный сегмент {seg.segment_id} на {gap:.1f}px к J{j.id} ({dirn}); bbox {original} -> {seg.bbox}")
            else:
                if dirn == 'right':
                    right = seg.bbox['x'] + seg.bbox['width']
                    seg.bbox['width'] = right - j.x
                    seg.bbox['x'] = j.x
                else:  # left
                    seg.bbox['width'] = j.x - seg.bbox['x']
                print(f"    ✓ [BRIDGE] Вытянули горизонтальный сегмент {seg.segment_id} на {gap:.1f}px к J{j.id} ({dirn}); bbox {original} -> {seg.bbox}")
            changed += 1

    print(f"  ✓ Вытянули {changed} сегментов для закрытия малых зазоров")
    return changed

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

def draw_aligned_walls(dwg: svgwrite.Drawing, aligned_walls: List[Dict], inverse_scale: float, padding: float) -> None:
    """
    Отрисовывает выровненные стены с специальной стилизацией
    
    Args:
        dwg: SVG drawing object
        aligned_walls: List of aligned wall dictionaries
        inverse_scale: Scale factor for coordinate transformation
        padding: Padding for SVG coordinates
    """
    aligned_walls_group = dwg.add(dwg.g(id='aligned_walls'))
    
    # Стиль для выровненных стен
    aligned_wall_style = {
        'stroke': '#00CED1',  # Темно-бирюзовый
        'stroke_width': 4,
        'fill': 'none',
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round',
        'opacity': 0.7
    }
    
    print(f"  ✓ Отрисовка {len(aligned_walls)} выровненных стен")
    
    for idx, wall in enumerate(aligned_walls):
        # Получаем координаты
        x = wall.get('x', 0)
        y = wall.get('y', 0)
        width = wall.get('width', 0)
        height = wall.get('height', 0)
        
        # Преобразуем координаты
        svg_x, svg_y = transform_coordinates(x, y, inverse_scale, padding)
        svg_width = width * inverse_scale
        svg_height = height * inverse_scale
        
        # Создаем прямоугольник
        rect = dwg.rect(insert=(svg_x, svg_y), size=(svg_width, svg_height), **aligned_wall_style)
        aligned_walls_group.add(rect)
        
        # Добавляем метку с информацией о выравнивании
        wall_id = wall.get('id', f'A{idx+1}')
        alignment_info = wall.get('alignment_info', {})
        opening_id = alignment_info.get('opening_id', 'N/A')
        
        text = dwg.text(
            f"A{idx+1}_{opening_id}",
            insert=(svg_x + svg_width/2, svg_y + svg_height/2),
            text_anchor='middle',
            fill='#00CED1',
            font_size='8px',
            font_weight='bold'
        )
        aligned_walls_group.add(text)
        
        print(f"    ✓ Выровненная стена A{idx+1}: ({x:.1f}, {y:.1f}) {width:.1f}x{height:.1f} -> выровнена по проему {opening_id}")

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
        "План этажа - Opening-Based с улучшенным расширением L-junctions",
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
    output_json_path = 'wall_coordinates.json'
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
    
    # Инициализируем JSON структуру для хранения данных
    json_data = initialize_json_data(input_path, wall_thickness)
    
    # Обрабатываем проемы с учетом junctions
    print(f"\n{'='*60}")
    print("ОБРАБОТКА ПРОЕМОВ С УЧЕТОМ JUNCTIONS")
    print(f"{'='*60}")
    
    wall_segments = process_openings_with_junctions(data, wall_thickness, json_data)
    
    # Обрабатываем колонны и создаем квадраты
    process_pillars_to_squares(data, wall_thickness, json_data)
    
    # Анализируем типы junctions с улучшенной логикой
    print(f"\n{'='*60}")
    print("АНАЛИЗ ТИПОВ JUNCTIONS С УЛУЧШЕННОЙ ЛОГИКОЙ")
    print(f"{'='*60}")
    
    junctions = parse_junctions(data)
    junctions_with_types = analyze_junction_types_with_thickness(junctions, data, wall_thickness)
    # Гибридная коррекция типов: если по полигону тип unknown/одно направление,
    # но рядом есть один горизонтальный и один вертикальный сегмент — помечаем как L
    junctions_with_types = override_junction_types_with_connectivity(
        junctions_with_types, wall_segments, [], wall_thickness
    )
    
    # Добавляем все junctions в JSON
    for junction in junctions_with_types:
        add_junction_to_json(json_data, junction)
    
    # Находим junctions не связанные с проемами и достраиваем недостающие сегменты
    print(f"\n{'='*60}")
    print("ПОИСК JUNCTIONS НЕ СВЯЗАННЫХ С ПРОЕМАМИ И ДОСТРАИВАНИЕ СТЕН")
    print(f"{'='*60}")
    
    junction_wall_segments = []  # Инициализируем пустой список для сегментов стен из junctions
    # Увеличиваем ограничение итераций и обрабатываем все доступные узлы на каждой итерации,
    # пока появляются новые сегменты. Это предотвращает ситуацию, когда обрабатывается только часть junctions.
    max_iterations = max(20, len(junctions_with_types))
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Итерация {iteration} ---")

        # Находим junctions не связанные с проемами или уже построенными сегментами
        non_opening_junctions = find_junctions_not_related_to_openings(
            junctions_with_types, wall_segments, junction_wall_segments
        )

        if not non_opening_junctions:
            print("  ✓ Все junctions связаны с проемами или уже имеют достаточное количество сегментов стен")
            break

        # Обрабатываем все такие junctions за итерацию, избегая дубликатов
        added_this_iteration = 0
        for junction in non_opening_junctions:
            print(f"  Обработка junction {junction.id}...")

            new_segments = build_missing_wall_segments_for_junctions(
                [junction], wall_segments, junction_wall_segments, junctions_with_types, wall_thickness, json_data, data.get('wall_polygons', [])
            )

            if not new_segments:
                print(f"  ✗ Не удалось создать сегменты для junction {junction.id}")
                continue

            unique_segments = []
            for new_segment in new_segments:
                if not segment_exists(
                    new_segment.start_junction,
                    new_segment.end_junction,
                    wall_segments + junction_wall_segments + unique_segments,
                ):
                    unique_segments.append(new_segment)
                    print(
                        f"    ✓ Создан уникальный сегмент от J{new_segment.start_junction.id} до J{new_segment.end_junction.id}"
                    )
                else:
                    print(
                        f"    ✗ Сегмент от J{new_segment.start_junction.id} до J{new_segment.end_junction.id} уже существует, пропускаем"
                    )

            if unique_segments:
                junction_wall_segments.extend(unique_segments)
                added_this_iteration += len(unique_segments)
                print(
                    f"  ✓ Создано {len(unique_segments)} новых уникальных сегментов для junction {junction.id}"
                )

        if added_this_iteration == 0:
            print(f"  ✓ Новые сегменты стен не найдены на итерации {iteration}")
            break
        else:
            print(f"  ✓ Добавлено {added_this_iteration} сегментов на итерации {iteration}")
    
    # Перед обработкой L-junctions корректируем типы по подключенным сегментам
    junctions_with_types = override_junction_types_with_connectivity(
        junctions_with_types, wall_segments, junction_wall_segments, wall_thickness
    )

    # Process L-junctions and extend original segments
    print(f"\n{'='*60}")
    print("ОБРАБОТКА L-JUNCTIONS И РАСШИРЕНИЕ ОРИГИНАЛЬНЫХ СЕГМЕНТОВ")
    print(f"{'='*60}")
    
    extended_count = process_l_junction_extensions(
        junctions_with_types, wall_segments, junction_wall_segments,
        data.get('wall_polygons', []), wall_thickness
    )
    # Дополнительный проход: вытягивание сегментов для малых зазоров вокруг всех junctions
    print(f"\n{'='*60}")
    print("ЗАКРЫТИЕ МАЛЫХ ЗАЗОРОВ МЕЖДУ СОСЕДНИМИ СТЕНАМИ")
    print(f"{'='*60}")
    bridged = bridge_small_gaps_around_all_junctions(
        junctions_with_types, wall_segments, junction_wall_segments, wall_thickness
    )
    extended_count += bridged
    
    # Обновляем статистику расширенных сегментов
    json_data["statistics"]["extended_segments"] = extended_count
    
    # Обновляем JSON данные с расширенными сегментами
    print(f"\n{'='*60}")
    print("ОБНОВЛЕНИЕ JSON ДАННЫХ С РАСШИРЕННЫМИ СЕГМЕНТАМИ")
    print(f"{'='*60}")
    
    # Очищаем существующие списки сегментов в JSON
    json_data["wall_segments_from_openings"] = []
    json_data["wall_segments_from_junctions"] = []
    
    # Добавляем обновленные сегменты из проемов в JSON
    for segment in wall_segments:
        add_wall_segment_from_opening_to_json(json_data, segment)
        print(f"  ✓ Обновлен сегмент из проема: {segment.segment_id}")
    
    # Добавляем обновленные сегменты из junctions в JSON
    for segment in junction_wall_segments:
        add_wall_segment_from_junction_to_json(json_data, segment)
        print(f"  ✓ Обновлен сегмент из junction: {segment.segment_id}")
    
    print(f"  ✓ JSON данные обновлены с {extended_count} расширенными сегментами")
    
    # Добавляем привязки junctions к стенам
    print(f"\n{'='*60}")
    print("ДОБАВЛЕНИЕ ПРИВЯЗОК JUNCTIONS К СТЕНАМ")
    print(f"{'='*60}")
    
    update_existing_junctions_with_wall_bindings(json_data)
    
    # Выравнивание стен по проемам
    print(f"\n{'='*60}")
    print("ВЫРАВНИВАНИЕ СТЕН ПО ПРОЕМАМ")
    print(f"{'='*60}")
    
    # Проверяем на дубликаты перед выравниванием
    print(f"  Проверка на дубликаты сегментов:")
    print(f"    Сегменты из проемов: {len(wall_segments)}")
    print(f"    Сегменты из junctions: {len(junction_wall_segments)}")
    
    # Применяем выравнивание непосредственно к существующим сегментам
    apply_alignment_to_existing_segments(wall_segments, junction_wall_segments, data)
    
    print(f"  ✓ Выравнивание применено к {len(wall_segments)} сегментам из проемов и {len(junction_wall_segments)} сегментам из junctions")
    
    # Объединяем все сегменты стен в один список для визуализации
    all_wall_segments_for_visualization = []
    
    # Добавляем сегменты из проемов
    all_wall_segments_for_visualization.extend(wall_segments)
    
    # Добавляем сегменты из junctions
    all_wall_segments_for_visualization.extend(junction_wall_segments)
    
    # Сохраняем все данные в JSON
    save_json_data(json_data, output_json_path)
    
    # Создаем SVG на основе сохраненных JSON данных
    create_svg_from_json(output_json_path, output_path, data)
    
    # Добавляем объекты outline, foundation и street в основной JSON файл
    copy_to_main_json(input_path, output_json_path)
    
    print(f"\nГотово! Векторная визуализация с измененной логикой расширения сегментов создана: {output_path}")
    print(f"Данные координат сохранены в: {output_json_path}")

def apply_alignment_to_existing_segments(wall_segments: List[WallSegmentFromOpening],
                                        junction_wall_segments: List[WallSegmentFromJunction],
                                        data: Dict) -> None:
    """
    Применяет выравнивание непосредственно к существующим сегментам стен
    
    Args:
        wall_segments: Список сегментов стен из проемов
        junction_wall_segments: Список сегментов стен из junctions
        data: Данные плана с проемами
    """
    print(f"  Начинаем выравнивание {len(wall_segments)} сегментов из проемов и {len(junction_wall_segments)} сегментов из junctions")
    
    openings = data.get('openings', [])
    wall_polygons = data.get('wall_polygons', [])
    wall_thickness = get_wall_thickness_from_doors(data)
    
    # Конвертируем все сегменты в единый формат для обработки
    all_segments = []
    
    # Добавляем сегменты из проемов с ссылкой на оригинальный объект
    for segment in wall_segments:
        all_segments.append({
            'type': 'opening',
            'original': segment,
            'bbox': segment.bbox,
            'orientation': segment.orientation,
            'id': segment.segment_id
        })
    
    # Добавляем сегменты из junctions с ссылкой на оригинальный объект
    for segment in junction_wall_segments:
        all_segments.append({
            'type': 'junction',
            'original': segment,
            'bbox': segment.bbox,
            'orientation': segment.orientation,
            'id': segment.segment_id
        })
    
    print(f"  ✓ Конвертировано {len(all_segments)} сегментов в единый формат")
    
    # Группируем стены по ориентации
    horizontal_walls = []
    vertical_walls = []
    
    for wall_data in all_segments:
        orientation = wall_data.get('orientation', 'unknown')
        if orientation == 'horizontal':
            horizontal_walls.append(wall_data)
        elif orientation == 'vertical':
            vertical_walls.append(wall_data)
    
    print(f"  ✓ Разделено на {len(horizontal_walls)} горизонтальных и {len(vertical_walls)} вертикальных стен")
    
    # Обрабатываем горизонтальные стены
    print(f"  Обрабатываем горизонтальные стены...")
    horizontal_groups = group_walls_by_position(horizontal_walls, 'y', wall_thickness)
    print(f"  ✓ Найдено {len(horizontal_groups)} групп горизонтальных стен")
    
    for position, group_walls in horizontal_groups.items():
        print(f"    Группа на Y={position:.1f}: {len(group_walls)} стен")
        for wall in group_walls:
            print(f"      Стена {wall['id']}: bbox={wall['bbox']}")
        
        # Ищем проемы на этой же Y-координате
        related_openings = []
        for opening in openings:
            bbox = opening.get('bbox', {})
            if not bbox:
                continue
            
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            # Только горизонтальные проемы
            if width > height:
                opening_y = bbox['y']
                if abs(opening_y - position) <= wall_thickness:
                    related_openings.append(opening)
        
        print(f"      Найдено {len(related_openings)} связанных проемов")
        
        if related_openings and len(group_walls) > 1:
            # Выбираем опорный проем (ближайший к центру группы)
            center_x = sum(wall['bbox']['x'] + wall['bbox']['width']/2 for wall in group_walls) / len(group_walls)
            center_y = position
            
            closest_opening = None
            min_distance = float('inf')
            
            for opening in related_openings:
                bbox = opening.get('bbox', {})
                opening_center_x = bbox['x'] + bbox['width'] / 2
                opening_center_y = bbox['y'] + bbox['height'] / 2
                
                distance = abs(opening_center_x - center_x) + abs(opening_center_y - center_y)
                if distance < min_distance:
                    min_distance = distance
                    closest_opening = opening
            
            if closest_opening:
                # Выравниваем все стены по Y-координате опорного проема
                target_y = closest_opening['bbox']['y']
                
                for wall_data in group_walls:
                    original = wall_data['original']
                    original_y = wall_data['bbox']['y']
                    
                    # Модифицируем bbox в оригинальном объекте
                    original.bbox['y'] = target_y
                    original.bbox['alignment_info'] = {
                        'opening_id': closest_opening.get('id', 'unknown'),
                        'original_coords': {'x': wall_data['bbox']['x'], 'y': original_y},
                        'alignment_type': 'opening'
                    }
                    
                    print(f"    ✓ Стена {wall_data['id']}: Y {original_y:.1f} -> {target_y:.1f} (по проему {closest_opening.get('id')})")
        else:
            # НОВАЯ ЛОГИКА: выравнивание по полигонам стен
            print(f"      Проемы не найдены, ищем полигоны стен для выравнивания...")
            print(f"      Количество стен в группе: {len(group_walls)}")
            
            # Проверяем, нужно ли выравнивать эту группу
            if len(group_walls) > 1:
                # Находим лучший полигон для выравнивания
                best_polygon_id = find_best_wall_polygon_for_alignment(
                    group_walls, wall_polygons, wall_thickness
                )
                
                if best_polygon_id:
                    print(f"      Найден полигон для выравнивания: {best_polygon_id}")
                    # Выравниваем стены по полигону
                    aligned_walls = align_wall_group_to_polygon(
                        group_walls, best_polygon_id, wall_polygons, wall_thickness
                    )
                    # Обновляем оригинальные стены
                    for i, wall_data in enumerate(group_walls):
                        if i < len(aligned_walls):
                            original = wall_data['original']
                            original.bbox.update(aligned_walls[i])
                            original.bbox['alignment_info'] = aligned_walls[i]['alignment_info']
                else:
                    print(f"      Полигоны для выравнивания не найдены")
            else:
                print(f"      Группа содержит только одну стену, выравнивание не требуется")
    
    # Обрабатываем вертикальные стены
    print(f"  Обрабатываем вертикальные стены...")
    vertical_groups = group_walls_by_position(vertical_walls, 'x', wall_thickness)
    print(f"  ✓ Найдено {len(vertical_groups)} групп вертикальных стен")
    
    for position, group_walls in vertical_groups.items():
        print(f"    Группа на X={position:.1f}: {len(group_walls)} стен")
        for wall in group_walls:
            print(f"      Стена {wall['id']}: bbox={wall['bbox']}")
        
        # Ищем проемы на этой же X-координате
        related_openings = []
        for opening in openings:
            bbox = opening.get('bbox', {})
            if not bbox:
                continue
            
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            # Только вертикальные проемы
            if height > width:
                opening_x = bbox['x']
                if abs(opening_x - position) <= wall_thickness:
                    related_openings.append(opening)
        
        print(f"      Найдено {len(related_openings)} связанных проемов")
        
        if related_openings and len(group_walls) > 1:
            # Выбираем опорный проем (ближайший к центру группы)
            center_x = position
            center_y = sum(wall['bbox']['y'] + wall['bbox']['height']/2 for wall in group_walls) / len(group_walls)
            
            closest_opening = None
            min_distance = float('inf')
            
            for opening in related_openings:
                bbox = opening.get('bbox', {})
                opening_center_x = bbox['x'] + bbox['width'] / 2
                opening_center_y = bbox['y'] + bbox['height'] / 2
                
                distance = abs(opening_center_x - center_x) + abs(opening_center_y - center_y)
                if distance < min_distance:
                    min_distance = distance
                    closest_opening = opening
            
            if closest_opening:
                # Выравниваем все стены по X-координате опорного проема
                target_x = closest_opening['bbox']['x']
                
                for wall_data in group_walls:
                    original = wall_data['original']
                    original_x = wall_data['bbox']['x']
                    
                    # Модифицируем bbox в оригинальном объекте
                    original.bbox['x'] = target_x
                    original.bbox['alignment_info'] = {
                        'opening_id': closest_opening.get('id', 'unknown'),
                        'original_coords': {'x': original_x, 'y': wall_data['bbox']['y']},
                        'alignment_type': 'opening'
                    }
                    
                    print(f"    ✓ Стена {wall_data['id']}: X {original_x:.1f} -> {target_x:.1f} (по проему {closest_opening.get('id')})")
        else:
            # НОВАЯ ЛОГИКА: выравнивание по полигонам стен
            print(f"      Проемы не найдены, ищем полигоны стен для выравнивания...")
            print(f"      Количество стен в группе: {len(group_walls)}")
            print(f"      Количество полигонов стен: {len(wall_polygons)}")
            
            # Проверяем, нужно ли выравнивать эту группу
            if len(group_walls) > 1:
                # Находим лучший полигон для выравнивания
                best_polygon_id = find_best_wall_polygon_for_alignment(
                    group_walls, wall_polygons, wall_thickness
                )
                
                if best_polygon_id:
                    print(f"      Найден полигон для выравнивания: {best_polygon_id}")
                    # Выравниваем стены по полигону
                    aligned_walls = align_wall_group_to_polygon(
                        group_walls, best_polygon_id, wall_polygons, wall_thickness
                    )
                    # Обновляем оригинальные стены
                    for i, wall_data in enumerate(group_walls):
                        if i < len(aligned_walls):
                            original = wall_data['original']
                            original.bbox.update(aligned_walls[i])
                            original.bbox['alignment_info'] = aligned_walls[i]['alignment_info']
                else:
                    print(f"      Полигоны для выравнивания не найдены")
                    # ДОБАВЛЕНО: Дополнительная отладочная информация
                    print(f"      Проверяем пересечения для каждой стены в группе:")
                    for wall_data in group_walls:
                        intersections = count_wall_polygon_intersections(
                            wall_data['bbox'], wall_polygons, wall_thickness
                        )
                        print(f"        Стена {wall_data['id']}: пересечения={intersections}")
                        
                        # Additional debugging for problematic walls
                        if wall_data['id'] in ['wall_junction_11_to_15_down', 'wall_junction_13_to_15_up']:
                            print(f"          DEBUG: Детальная информация для проблемной стены {wall_data['id']}")
                            print(f"          DEBUG: bbox={wall_data['bbox']}")
                            print(f"          DEBUG: orientation={wall_data.get('orientation', 'unknown')}")
                            
                            # Check each polygon individually
                            for polygon in wall_polygons:
                                polygon_id = polygon.get('id', 'unknown')
                                vertices = polygon.get('vertices', [])
                                if vertices:
                                    relationship_type, score = check_wall_polygon_relationship(
                                        wall_data['bbox'], vertices, wall_thickness
                                    )
                                    print(f"          DEBUG: Отношение с {polygon_id}: {relationship_type} (score: {score})")
            else:
                print(f"      Группа содержит только одну стену, выравнивание не требуется")
    
    print(f"  ✓ Выравнивание завершено")

def line_intersects_polygon(line_start, line_end, polygon_vertices, tolerance=1.0):
    """
    Определяет, пересекает ли линия полигон
    
    Args:
        line_start: Начальная точка линии (x, y)
        line_end: Конечная точка линии (x, y)
        polygon_vertices: Список вершин полигона [{'x': x, 'y': y}, ...]
        tolerance: Допуск для определения пересечения
    
    Returns:
        True если линия пересекает полигон, иначе False
    """
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Проверяем пересечение с каждым ребром полигона
    for i in range(len(polygon_vertices)):
        v1 = polygon_vertices[i]
        v2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
        
        # Ребро полигона
        x3, y3 = v1['x'], v1['y']
        x4, y4 = v2['x'], v2['y']
        
        # Проверяем пересечение отрезков
        if line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4, tolerance):
            return True
    
    return False

def line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4, tolerance=1.0):
    """
    Проверяет пересечение двух отрезков с учетом допуска
    
    Args:
        x1, y1: Координаты начала первого отрезка
        x2, y2: Координаты конца первого отрезка
        x3, y3: Координаты начала второго отрезка
        x4, y4: Координаты конца второго отрезка
        tolerance: Допуск для определения пересечения
    
    Returns:
        True если отрезки пересекаются, иначе False
    """
    # Вычисляем определители
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Если отрезки параллельны
    if abs(den) < tolerance:
        return False
    
    # Вычисляем параметры пересечения
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
    # Проверяем, находится ли точка пересечения внутри обоих отрезков
    return 0 <= t <= 1 and 0 <= u <= 1

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Вычисляет расстояние от точки до линии
    
    Args:
        px, py: Координаты точки
        x1, y1: Координаты начала линии
        x2, y2: Координаты конца линии
    
    Returns:
        Расстояние от точки до линии
    """
    # Длина линии
    line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if line_length == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Параметрическое представление линии
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)))
    
    # Ближайшая точка на линии
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    
    # Расстояние до ближайшей точки
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def count_close_vertices(line_start, line_end, vertices, wall_thickness):
    """
    Подсчитывает количество вершин полигона, близких к линии стены
    
    Args:
        line_start: Начальная точка линии стены
        line_end: Конечная точка линии стены
        vertices: Список вершин полигона
        wall_thickness: Толщина стены
    
    Returns:
        Количество близких вершин
    """
    x1, y1 = line_start
    x2, y2 = line_end
    count = 0
    
    for vertex in vertices:
        vx, vy = vertex['x'], vertex['y']
        
        # Вычисляем расстояние от точки до линии
        distance = point_to_line_distance(vx, vy, x1, y1, x2, y2)
        
        if distance <= wall_thickness:
            count += 1
    
    return count
def check_wall_polygon_relationship(wall_bbox, polygon_vertices, wall_thickness):
    """
    Checks the relationship between a wall and a polygon
    
    Args:
        wall_bbox: Bbox of the wall {'x': x, 'y': y, 'width': w, 'height': h}
        polygon_vertices: List of vertices of the polygon
        wall_thickness: Thickness of the wall
    
    Returns:
        A tuple (relationship_type, score) where relationship_type is:
        - 'inside': Wall is inside the polygon
        - 'intersects': Wall intersects polygon edges
        - 'near': Wall is near the polygon (within tolerance)
        - 'contains': Wall contains polygon vertices
        - 'none': No relationship detected
    """
    # Get wall center and endpoints
    wall_center_x = wall_bbox['x'] + wall_bbox['width'] / 2
    wall_center_y = wall_bbox['y'] + wall_bbox['height'] / 2
    
    # Determine if wall is horizontal or vertical
    if wall_bbox['width'] > wall_bbox['height']:  # Horizontal wall
        line_start = (wall_bbox['x'], wall_center_y)
        line_end = (wall_bbox['x'] + wall_bbox['width'], wall_center_y)
    else:  # Vertical wall
        line_start = (wall_center_x, wall_bbox['y'])
        line_end = (wall_center_x, wall_bbox['y'] + wall_bbox['height'])
    
    # Check if center is inside polygon
    center_inside = is_point_in_polygon(wall_center_x, wall_center_y, polygon_vertices)
    if center_inside:
        return ('inside', 100)
    
    # Check for edge intersections
    if line_intersects_polygon(line_start, line_end, polygon_vertices, wall_thickness / 2):
        return ('intersects', 80)
    
    # Check for nearby vertices
    close_vertices = count_close_vertices(line_start, line_end, polygon_vertices, wall_thickness * 2)
    if close_vertices > 0:
        return ('near', 60 * close_vertices)
    
    # Check if wall contains polygon vertices
    contained_vertices = 0
    for vertex in polygon_vertices:
        vx, vy = vertex['x'], vertex['y']
        if (wall_bbox['x'] <= vx <= wall_bbox['x'] + wall_bbox['width'] and
            wall_bbox['y'] <= vy <= wall_bbox['y'] + wall_bbox['height']):
            contained_vertices += 1
    
    if contained_vertices > 0:
        return ('contains', 40 * contained_vertices)
    
    return ('none', 0)


def count_wall_polygon_intersections(wall_bbox, wall_polygons, wall_thickness):
    """
    Enhanced version that counts intersections and other relationships
    
    Args:
        wall_bbox: Bbox of the wall {'x': x, 'y': y, 'width': w, 'height': h}
        wall_polygons: List of wall polygons from JSON
        wall_thickness: Thickness of the wall
    
    Returns:
        List of tuples (polygon_id, score, relationship_type)
    """
    relationships = []
    
    # Get wall center and endpoints for debugging
    wall_center_x = wall_bbox['x'] + wall_bbox['width'] / 2
    wall_center_y = wall_bbox['y'] + wall_bbox['height'] / 2
    
    # Determine if wall is horizontal or vertical
    if wall_bbox['width'] > wall_bbox['height']:  # Horizontal wall
        line_start = (wall_bbox['x'], wall_center_y)
        line_end = (wall_bbox['x'] + wall_bbox['width'], wall_center_y)
        orientation = 'horizontal'
    else:  # Vertical wall
        line_start = (wall_center_x, wall_bbox['y'])
        line_end = (wall_center_x, wall_bbox['y'] + wall_bbox['height'])
        orientation = 'vertical'
    
    print(f"    DEBUG: Checking {orientation} wall at ({wall_center_x:.1f}, {wall_center_y:.1f})")
    
    # Check each polygon
    for polygon in wall_polygons:
        polygon_id = polygon.get('id', 'unknown')
        vertices = polygon.get('vertices', [])
        
        if not vertices:
            continue
        
        # Check relationship type
        relationship_type, score = check_wall_polygon_relationship(wall_bbox, vertices, wall_thickness)
        
        if relationship_type != 'none':
            relationships.append((polygon_id, score, relationship_type))
            print(f"      Found {relationship_type} relationship with {polygon_id} (score: {score})")
    
    # Sort by score (descending)
    relationships.sort(key=lambda x: x[1], reverse=True)
    
    print(f"    Total relationships found: {len(relationships)}")
    return relationships

def find_best_wall_polygon_for_alignment(wall_group, wall_polygons, wall_thickness):
    """
    Enhanced version that finds the best polygon for alignment
    
    Args:
        wall_group: Group of walls for alignment
        wall_polygons: List of wall polygons from JSON
        wall_thickness: Thickness of the wall
    
    Returns:
        ID of the best polygon for alignment or None if not found
    """
    if not wall_polygons:
        return None
    
    # Sum up relationship scores for all walls in the group
    total_scores = {}
    relationship_details = {}
    
    for wall_data in wall_group:
        wall_bbox = wall_data['bbox']
        wall_id = wall_data.get('id', 'unknown')
        
        print(f"  Checking relationships for wall {wall_id}")
        
        # Get relationships for this wall
        relationships = count_wall_polygon_intersections(wall_bbox, wall_polygons, wall_thickness)
        
        # Add to total scores
        for polygon_id, score, relationship_type in relationships:
            if polygon_id not in total_scores:
                total_scores[polygon_id] = 0
                relationship_details[polygon_id] = []
            
            total_scores[polygon_id] += score
            relationship_details[polygon_id].append((wall_id, relationship_type, score))
    
    # Find the polygon with the highest total score
    if not total_scores:
        print("  No relationships found for any walls in the group")
        return None
    
    best_polygon_id = max(total_scores.items(), key=lambda x: x[1])[0]
    best_score = total_scores[best_polygon_id]
    
    print(f"  Best polygon found: {best_polygon_id} with total score: {best_score}")
    print(f"  Relationship details:")
    for wall_id, relationship_type, score in relationship_details[best_polygon_id]:
        print(f"    {wall_id}: {relationship_type} (score: {score})")
    
    return best_polygon_id

def calculate_average_y_for_horizontal_walls(wall_group, polygon_vertices, wall_thickness):
    """
    Вычисляет среднюю Y-координату для выравнивания горизонтальных стен
    
    Args:
        wall_group: Группа горизонтальных стен
        polygon_vertices: Вершины полигона
        wall_thickness: Толщина стены
    
    Returns:
        Целевая Y-координата для выравнивания
    """
    # Определяем Y-координату стен в группе
    wall_y = wall_group[0]['bbox']['y'] + wall_group[0]['bbox']['height'] / 2
    
    # Находим все вершины полигона с близкой Y-координатой
    close_vertices = []
    for vertex in polygon_vertices:
        vy = vertex['y']
        if abs(vy - wall_y) <= wall_thickness * 2:  # Используем больший допуск
            close_vertices.append(vy)
    
    if close_vertices:
        # Вычисляем среднюю Y-координату
        avg_y = sum(close_vertices) / len(close_vertices)
        # Смещаем на половину толщины стены вверх
        target_y = avg_y - wall_thickness / 2
        return target_y
    else:
        # Если близких вершин нет, используем Y-координату стен
        return wall_group[0]['bbox']['y']

def calculate_average_x_for_vertical_walls(wall_group, polygon_vertices, wall_thickness):
    """
    Вычисляет среднюю X-координату для выравнивания вертикальных стен
    
    Args:
        wall_group: Группа вертикальных стен
        polygon_vertices: Вершины полигона
        wall_thickness: Толщина стены
    
    Returns:
        Целевая X-координата для выравнивания
    """
    # Определяем X-координату стен в группе
    wall_x = wall_group[0]['bbox']['x'] + wall_group[0]['bbox']['width'] / 2
    
    # Находим все вершины полигона с близкой X-координатой
    close_vertices = []
    for vertex in polygon_vertices:
        vx = vertex['x']
        if abs(vx - wall_x) <= wall_thickness * 2:  # Используем больший допуск
            close_vertices.append(vx)
    
    if close_vertices:
        # Вычисляем среднюю X-координату
        avg_x = sum(close_vertices) / len(close_vertices)
        # Смещаем на половину толщины стены влево
        target_x = avg_x - wall_thickness / 2
        return target_x
    else:
        # Если близких вершин нет, используем X-координату стен
        return wall_group[0]['bbox']['x']

def align_wall_group_to_polygon(wall_group, polygon_id, wall_polygons, wall_thickness):
    """
    Enhanced version that aligns a group of walls to a polygon
    
    Args:
        wall_group: Group of walls for alignment
        polygon_id: ID of the polygon for alignment
        wall_polygons: List of wall polygons from JSON
        wall_thickness: Thickness of the wall
    
    Returns:
        List of aligned walls
    """
    # Find the polygon by ID
    target_polygon = None
    for polygon in wall_polygons:
        if polygon.get('id') == polygon_id:
            target_polygon = polygon
            break
    
    if not target_polygon:
        print(f"      Polygon {polygon_id} not found")
        return []
    
    vertices = target_polygon.get('vertices', [])
    if not vertices:
        print(f"      Polygon {polygon_id} has no vertices")
        return []
    
    # Determine orientation of walls in the group
    orientation = wall_group[0].get('orientation', 'unknown')
    
    if orientation == 'horizontal':
        # For horizontal walls, align by Y-coordinate
        target_y = calculate_average_y_for_horizontal_walls(wall_group, vertices, wall_thickness)
        
        aligned_walls = []
        for wall_data in wall_group:
            wall_bbox = wall_data['bbox'].copy()
            original_y = wall_bbox['y']
            
            # Align Y-coordinate
            wall_bbox['y'] = target_y
            
            # Save alignment information
            wall_bbox['alignment_info'] = {
                'polygon_id': polygon_id,
                'original_coords': {'x': wall_bbox['x'], 'y': original_y},
                'alignment_type': 'polygon',
                'relationship_scores': []
            }
            
            # Add relationship scores for debugging
            relationships = count_wall_polygon_intersections(wall_bbox, [target_polygon], wall_thickness)
            for rel_polygon_id, score, relationship_type in relationships:
                wall_bbox['alignment_info']['relationship_scores'].append({
                    'polygon_id': rel_polygon_id,
                    'relationship_type': relationship_type,
                    'score': score
                })
            
            aligned_walls.append(wall_bbox)
            print(f"        Wall {wall_data.get('id', 'unknown')}: Y {original_y:.1f} -> {target_y:.1f}")
        
        return aligned_walls
    
    elif orientation == 'vertical':
        # For vertical walls, align by X-coordinate
        target_x = calculate_average_x_for_vertical_walls(wall_group, vertices, wall_thickness)
        
        aligned_walls = []
        for wall_data in wall_group:
            wall_bbox = wall_data['bbox'].copy()
            original_x = wall_bbox['x']
            
            # Align X-coordinate
            wall_bbox['x'] = target_x
            
            # Save alignment information
            wall_bbox['alignment_info'] = {
                'polygon_id': polygon_id,
                'original_coords': {'x': original_x, 'y': wall_bbox['y']},
                'alignment_type': 'polygon',
                'relationship_scores': []
            }
            
            # Add relationship scores for debugging
            relationships = count_wall_polygon_intersections(wall_bbox, [target_polygon], wall_thickness)
            for rel_polygon_id, score, relationship_type in relationships:
                wall_bbox['alignment_info']['relationship_scores'].append({
                    'polygon_id': rel_polygon_id,
                    'relationship_type': relationship_type,
                    'score': score
                })
            
            aligned_walls.append(wall_bbox)
            print(f"        Wall {wall_data.get('id', 'unknown')}: X {original_x:.1f} -> {target_x:.1f}")
        
        return aligned_walls
    
    else:
        print(f"      Unknown wall orientation: {orientation}")
        return []

def align_walls_by_openings_simple(walls, data):
    """
    Упрощенная функция выравнивания стен по проемам
    
    Args:
        walls: Список стен в формате bbox
        data: Данные плана с проемами
    
    Returns:
        Список выровненных стен
    """
    openings = data.get('openings', [])
    wall_thickness = get_wall_thickness_from_doors(data)
    
    # Группируем стены по ориентации
    horizontal_walls = []
    vertical_walls = []
    
    for wall in walls:
        orientation = wall.get('orientation', 'unknown')
        if orientation == 'horizontal':
            horizontal_walls.append(wall)
        elif orientation == 'vertical':
            vertical_walls.append(wall)
    
    aligned_walls = []
    
    # Обрабатываем горизонтальные стены
    horizontal_groups = group_walls_by_position(horizontal_walls, 'y', wall_thickness)
    for position, group_walls in horizontal_groups.items():
        # Ищем проемы на этой же Y-координате
        related_openings = []
        for opening in openings:
            bbox = opening.get('bbox', {})
            if not bbox:
                continue
            
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            # Только горизонтальные проемы
            if width > height:
                opening_y = bbox['y']
                if abs(opening_y - position) <= wall_thickness:
                    related_openings.append(opening)
        
        if related_openings and len(group_walls) > 1:
            # Выбираем опорный проем (ближайший к центру группы)
            center_x = sum(wall['x'] + wall['width']/2 for wall in group_walls) / len(group_walls)
            center_y = position
            
            closest_opening = None
            min_distance = float('inf')
            
            for opening in related_openings:
                bbox = opening.get('bbox', {})
                opening_center_x = bbox['x'] + bbox['width'] / 2
                opening_center_y = bbox['y'] + bbox['height'] / 2
                
                distance = abs(opening_center_x - center_x) + abs(opening_center_y - center_y)
                if distance < min_distance:
                    min_distance = distance
                    closest_opening = opening
            
            if closest_opening:
                # Выравниваем все стены по Y-координате опорного проема
                target_y = closest_opening['bbox']['y']
                
                for wall in group_walls:
                    aligned_wall = wall.copy()
                    original_y = wall['y']
                    aligned_wall['original_y'] = original_y
                    aligned_wall['y'] = target_y
                    aligned_wall['alignment_info'] = {
                        'opening_id': closest_opening.get('id', 'unknown'),
                        'original_coords': {'x': wall['x'], 'y': original_y}
                    }
                    aligned_walls.append(aligned_wall)
                    
                    print(f"    ✓ Стена {wall['id']}: Y {wall['y']:.1f} -> {target_y:.1f} (по проему {closest_opening.get('id')})")
    
    # Обрабатываем вертикальные стены
    vertical_groups = group_walls_by_position(vertical_walls, 'x', wall_thickness)
    for position, group_walls in vertical_groups.items():
        # Ищем проемы на этой же X-координате
        related_openings = []
        for opening in openings:
            bbox = opening.get('bbox', {})
            if not bbox:
                continue
            
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            # Только вертикальные проемы
            if height > width:
                opening_x = bbox['x']
                if abs(opening_x - position) <= wall_thickness:
                    related_openings.append(opening)
        
        if related_openings and len(group_walls) > 1:
            # Выбираем опорный проем (ближайший к центру группы)
            center_x = position
            center_y = sum(wall['y'] + wall['height']/2 for wall in group_walls) / len(group_walls)
            
            closest_opening = None
            min_distance = float('inf')
            
            for opening in related_openings:
                bbox = opening.get('bbox', {})
                opening_center_x = bbox['x'] + bbox['width'] / 2
                opening_center_y = bbox['y'] + bbox['height'] / 2
                
                distance = abs(opening_center_x - center_x) + abs(opening_center_y - center_y)
                if distance < min_distance:
                    min_distance = distance
                    closest_opening = opening
            
            if closest_opening:
                # Выравниваем все стены по X-координате опорного проема
                target_x = closest_opening['bbox']['x']
                
                for wall in group_walls:
                    aligned_wall = wall.copy()
                    original_x = wall['x']
                    aligned_wall['original_x'] = original_x
                    aligned_wall['x'] = target_x
                    aligned_wall['alignment_info'] = {
                        'opening_id': closest_opening.get('id', 'unknown'),
                        'original_coords': {'x': original_x, 'y': wall['y']}
                    }
                    aligned_walls.append(aligned_wall)
                    
                    print(f"    ✓ Стена {wall['id']}: X {wall['x']:.1f} -> {target_x:.1f} (по проему {closest_opening.get('id')})")
    
    return aligned_walls

def group_walls_by_position(walls, position_key, tolerance):
    """
    Группирует стены по позиции с учетом допуска
    
    Args:
        walls: Список стен
        position_key: 'x' или 'y'
        tolerance: Допуск для группировки
    
    Returns:
        Словарь {позиция: [стены]}
    """
    groups = {}
    ungrouped = walls.copy()
    
    while ungrouped:
        # Начинаем новую группу
        reference = ungrouped.pop(0)
        # Получаем позицию из bbox для нашей новой структуры данных
        if 'bbox' in reference:
            ref_pos = reference['bbox'][position_key]
        else:
            ref_pos = reference[position_key]  # Для обратной совместимости
        
        group_walls = [reference]
        
        # Ищем все стены в пределах допуска
        i = 0
        while i < len(ungrouped):
            wall = ungrouped[i]
            # Получаем позицию из bbox для нашей новой структуры данных
            if 'bbox' in wall:
                pos = wall['bbox'][position_key]
            else:
                pos = wall[position_key]  # Для обратной совместимости
            
            if abs(pos - ref_pos) <= tolerance:
                group_walls.append(ungrouped.pop(i))
            else:
                i += 1
        
        if len(group_walls) > 1:  # Только группы с несколькими стенами
            groups[ref_pos] = group_walls
    
    return groups

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

def copy_to_main_json(input_path: str, output_json_path: str) -> None:
    """
    Добавляет объекты outline, foundation и street из исходного JSON файла в основной wall_coordinates.json
    
    Args:
        input_path: Путь к исходному JSON файлу
        output_json_path: Путь к основному JSON файлу wall_coordinates.json
    """
    print(f"\n{'='*60}")
    print("КОПИРОВАНИЕ ОБЪЕКТОВ OUTLINE, FOUNDATION И STREET В ОСНОВНОЙ JSON")
    print(f"{'='*60}")
    
    try:
        # Загружаем исходные данные
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
        except UnicodeDecodeError:
            # Пробуем другие кодировки
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(input_path, 'r', encoding=encoding) as f:
                        source_data = json.load(f)
                    print(f"  ✓ Исходный файл загружен с кодировкой: {encoding}")
                    break
                except Exception:
                    continue
            else:
                print(f"  ✗ Не удалось загрузить исходный файл с поддерживаемыми кодировками")
                raise
        
        print(f"  ✓ Исходный файл загружен: {input_path}")
        
        # Загружаем основной JSON файл
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                main_data = json.load(f)
        except UnicodeDecodeError:
            # Пробуем другие кодировки
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(output_json_path, 'r', encoding=encoding) as f:
                        main_data = json.load(f)
                    print(f"  ✓ Основной JSON файл загружен с кодировкой: {encoding}")
                    break
                except Exception:
                    continue
            else:
                print(f"  ✗ Не удалось загрузить основной JSON файл с поддерживаемыми кодировками")
                raise
        
        print(f"  ✓ Основной JSON файл загружен: {output_json_path}")
        
        # Добавляем объекты outline, foundation и street в основной файл
        if source_data.get("building_outline"):
            main_data["building_outline"] = source_data["building_outline"]
            outline = source_data["building_outline"]
            print(f"  ✓ Building outline добавлен: {outline.get('id')}, {len(outline.get('vertices', []))} вершин")
        
        if source_data.get("foundation"):
            main_data["foundation"] = source_data["foundation"]
            foundation = source_data["foundation"]
            print(f"  ✓ Foundation добавлен: {foundation.get('id')}, {len(foundation.get('vertices', []))} вершин")
        
        if source_data.get("street"):
            main_data["street"] = source_data["street"]
            street = source_data["street"]
            print(f"  ✓ Street добавлен: {street.get('id')}, {len(street.get('vertices', []))} вершин")
        
        # Обновляем статистику
        if "statistics" not in main_data:
            main_data["statistics"] = {}
        
        main_data["statistics"]["has_building_outline"] = source_data.get("building_outline") is not None
        main_data["statistics"]["has_foundation"] = source_data.get("foundation") is not None
        main_data["statistics"]["has_street"] = source_data.get("street") is not None
        
        # Сохраняем обновленный основной файл
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(main_data, f, indent=2)
        
        print(f"  ✓ Основной JSON файл обновлен: {output_json_path}")
        print(f"  ✓ Копирование объектов завершено успешно")
        
    except Exception as e:
        print(f"  ✗ Ошибка при копировании объектов: {e}")

def main():
    """Основная функция с поддержкой аргументов командной строки"""
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        # Создаем имена выходных файлов в той же директории, что и входной файл
        input_dir = os.path.dirname(input_path) or '.'
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_json_path = os.path.join(input_dir, f'{base_name}_wall_coordinates.json')
        output_path = os.path.join(input_dir, f'{base_name}_wall_polygons.svg')
    else:
        # Используем файлы по умолчанию
        input_path = 'plan_floor1_objects.json'
        output_json_path = 'wall_coordinates.json'
        output_path = 'wall_polygons.svg'
    
    print("="*60)
    print("СОЗДАНИЕ ВЕКТОРНОЙ ВИЗУАЛИЗАЦИИ НА ОСНОВЕ ПРОЕМОВ С АНАЛИЗОМ ТИПОВ JUNCTIONS")
    print("="*60)
    
    # Параметры
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
    
    # Инициализируем JSON структуру для хранения данных
    json_data = initialize_json_data(input_path, wall_thickness)
    
    # Обрабатываем проемы с учетом junctions
    print(f"\n{'='*60}")
    print("ОБРАБОТКА ПРОЕМОВ С УЧЕТОМ JUNCTIONS")
    print(f"{'='*60}")
    
    wall_segments = process_openings_with_junctions(data, wall_thickness, json_data)
    
    # Обрабатываем колонны и создаем квадраты
    process_pillars_to_squares(data, wall_thickness, json_data)
    
    # Анализируем типы junctions с улучшенной логикой
    print(f"\n{'='*60}")
    print("АНАЛИЗ ТИПОВ JUNCTIONS С УЛУЧШЕННОЙ ЛОГИКОЙ")
    print(f"{'='*60}")
    
    junctions = parse_junctions(data)
    junctions_with_types = analyze_junction_types_with_thickness(junctions, data, wall_thickness)
    
    # Добавляем все junctions в JSON
    for junction in junctions_with_types:
        add_junction_to_json(json_data, junction)
    
    # Находим junctions не связанные с проемами и достраиваем недостающие сегменты
    print(f"\n{'='*60}")
    print("ПОИСК JUNCTIONS НЕ СВЯЗАННЫХ С ПРОЕМАМИ И ДОСТРАИВАНИЕ СТЕН")
    print(f"{'='*60}")
    
    junction_wall_segments = []  # Инициализируем пустой список для сегментов стен из junctions
    # Обрабатываем все подходящие junctions на каждой итерации, пока появляются новые сегменты
    max_iterations = max(20, len(junctions_with_types))
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Итерация {iteration} ---")

        # Находим junctions не связанные с проемами или уже построенными сегментами
        non_opening_junctions = find_junctions_not_related_to_openings(
            junctions_with_types, wall_segments, junction_wall_segments
        )

        if not non_opening_junctions:
            print("  ✓ Все junctions связаны с проемами или уже имеют достаточное количество сегментов стен")
            break

        added_this_iteration = 0
        for junction in non_opening_junctions:
            print(f"  Обработка junction {junction.id}...")

            new_segments = build_missing_wall_segments_for_junctions(
                [junction], wall_segments, junction_wall_segments, junctions_with_types, wall_thickness, json_data, data.get('wall_polygons', [])
            )

            if not new_segments:
                print(f"  ✗ Не удалось создать сегменты для junction {junction.id}")
                continue

            unique_segments = []
            for new_segment in new_segments:
                if not segment_exists(
                    new_segment.start_junction,
                    new_segment.end_junction,
                    wall_segments + junction_wall_segments + unique_segments,
                ):
                    unique_segments.append(new_segment)
                    print(
                        f"    ✓ Создан уникальный сегмент от J{new_segment.start_junction.id} до J{new_segment.end_junction.id}"
                    )
                else:
                    print(
                        f"    ✗ Сегмент от J{new_segment.start_junction.id} до J{new_segment.end_junction.id} уже существует, пропускаем"
                    )

            if unique_segments:
                junction_wall_segments.extend(unique_segments)
                added_this_iteration += len(unique_segments)
                print(
                    f"  ✓ Создано {len(unique_segments)} новых уникальных сегментов для junction {junction.id}"
                )

        if added_this_iteration == 0:
            print(f"  ✓ Новые сегменты стен не найдены на итерации {iteration}")
            break
        else:
            print(f"  ✓ Добавлено {added_this_iteration} сегментов на итерации {iteration}")
    
    # Перед обработкой L-junctions повторно скорректируем типы по фактическим связям
    # (на этом этапе уже есть junction_wall_segments после достраивания)
    junctions_with_types = override_junction_types_with_connectivity(
        junctions_with_types, wall_segments, junction_wall_segments, wall_thickness
    )

    # Process L-junctions and extend original segments
    print(f"\n{'='*60}")
    print("ОБРАБОТКА L-JUNCTIONS И РАСШИРЕНИЕ ОРИГИНАЛЬНЫХ СЕГМЕНТОВ")
    print(f"{'='*60}")
    
    extended_count = process_l_junction_extensions(
        junctions_with_types, wall_segments, junction_wall_segments,
        data.get('wall_polygons', []), wall_thickness
    )
    
    # Обновляем статистику расширенных сегментов
    json_data["statistics"]["extended_segments"] = extended_count
    
    # Обновляем JSON данные с расширенными сегментами
    print(f"\n{'='*60}")
    print("ОБНОВЛЕНИЕ JSON ДАННЫХ С РАСШИРЕННЫМИ СЕГМЕНТАМИ")
    print(f"{'='*60}")
    
    # Очищаем существующие списки сегментов в JSON
    json_data["wall_segments_from_openings"] = []
    json_data["wall_segments_from_junctions"] = []
    
    # Добавляем обновленные сегменты из проемов в JSON
    for segment in wall_segments:
        add_wall_segment_from_opening_to_json(json_data, segment)
        print(f"  ✓ Обновлен сегмент из проема: {segment.segment_id}")
    
    # Добавляем обновленные сегменты из junctions в JSON
    for segment in junction_wall_segments:
        add_wall_segment_from_junction_to_json(json_data, segment)
        print(f"  ✓ Обновлен сегмент из junction: {segment.segment_id}")
    
    print(f"  ✓ JSON данные обновлены с {extended_count} расширенными сегментами")
    
    # Добавляем привязки junctions к стенам
    print(f"\n{'='*60}")
    print("ДОБАВЛЕНИЕ ПРИВЯЗОК JUNCTIONS К СТЕНАМ")
    print(f"{'='*60}")
    
    update_existing_junctions_with_wall_bindings(json_data)
    
    # Выравнивание стен по проемам
    print(f"\n{'='*60}")
    print("ВЫРАВНИВАНИЕ СТЕН ПО ПРОЕМАМ")
    print(f"{'='*60}")
    
    # Проверяем на дубликаты перед выравниванием
    print(f"  Проверка на дубликаты сегментов:")
    print(f"    Сегменты из проемов: {len(wall_segments)}")
    print(f"    Сегменты из junctions: {len(junction_wall_segments)}")
    
    # Применяем выравнивание непосредственно к существующим сегментам
    apply_alignment_to_existing_segments(wall_segments, junction_wall_segments, data)
    
    print(f"  ✓ Выравнивание применено к {len(wall_segments)} сегментам из проемов и {len(junction_wall_segments)} сегментам из junctions")
    
    # Объединяем все сегменты стен в один список для визуализации
    all_wall_segments_for_visualization = []
    
    # Добавляем сегменты из проемов
    all_wall_segments_for_visualization.extend(wall_segments)
    
    # Добавляем сегменты из junctions
    all_wall_segments_for_visualization.extend(junction_wall_segments)
    
    # Сохраняем все данные в JSON
    save_json_data(json_data, output_json_path)
    
    # Создаем SVG на основе сохраненных JSON данных
    create_svg_from_json(output_json_path, output_path, data)
    
    # Добавляем объекты outline, foundation и street в основной JSON файл
    copy_to_main_json(input_path, output_json_path)
    
    print(f"\nГотово! Векторная визуализация с измененной логикой расширения сегментов создана: {output_path}")
    print(f"Данные координат сохранены в: {output_json_path}")

if __name__ == '__main__':
    # Сначала пробуем запустить основную функцию
    try:
        main()
    except Exception as e:
        print(f"✗ Ошибка при выполнении основной функции: {e}")
        print("\nЗапускаем замену колонн в существующем SVG файле...")
        replace_pillars_in_existing_svg()
