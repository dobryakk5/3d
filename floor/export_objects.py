#!/usr/bin/env python3
"""
Export floor plan objects to JSON for Blender import:
1. Walls with thickness and height
2. Windows and doors with wall references
3. Pillars (standalone, not intersecting with walls)
4. Rooms defined by wall boundaries
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import json
import datetime
from scipy import ndimage
from scipy.spatial import Delaunay
import sys
sys.path.append('floortrans')
from floortrans.models import get_model
import svgwrite

# Заглушка для отсутствующей функции
def enhance_lines_with_hatching(wall_mask, gray, min_line_length, min_line_overlap_ratio):
    """
    Заглушка для отсутствующей функции enhance_lines_with_hatching
    Возвращает исходную маску без изменений
    """
    print("   Внимание: используется заглушка для enhance_lines_with_hatching")
    return wall_mask, 0, 0
def detect_walls_from_raster(image_np, scale_factor=1.0):
    """
    Определяет прямые стены на архитектурном плане с помощью Hough Transform.
    Возвращает список стен в формате JSON.
    """
    # Настройки
    WALL_THICKNESS = 0.2
    WALL_HEIGHT = 3.0

    # === 1. Предобработка ===
    # Преобразуем RGB в градации серого, если необходимо
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Убираем шум и усиливаем контраст
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # === 2. Поиск прямых ===
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=80,
        maxLineGap=10
    )

    walls = []
    if lines is not None:
        for i, line in enumerate(lines, start=1):
            x1, y1, x2, y2 = line[0]
            walls.append({
                "id": f"wall_{i}",
                "start": {"x": float(x1 * scale_factor), "y": float(y1 * scale_factor)},
                "end": {"x": float(x2 * scale_factor), "y": float(y2 * scale_factor)},
                "thickness": WALL_THICKNESS,
                "height": WALL_HEIGHT,
                "source": "HoughTransform"
            })

    return walls


# Import detection functions from existing scripts
exec(open('cubicasa_vectorize.py').read().split('if __name__')[0])

# =============================================================================
# ФУНКЦИИ ИЗ ENHANCED_HATCHING_FIXED.PY ДЛЯ УЛУЧШЕННОГО ОБНАРУЖЕНИЯ СТЕН И СТОЛБОВ
# =============================================================================

# Параметры для настройки жесткости маски штриховки
KERNEL_SIZES = [35]  # Размеры ядер для разной толщины линий
DENSITY_THRESHOLDS = [0.32]  # Пороги плотности (чем выше, тем жестче)
ANGLES = [45, 135]  # Углы обнаружения линий
MIN_AREA = 1200  # Минимальная площадь области (None = отключено)
MAX_AREA = None  # Максимальная площадь области (None = отключено)
MIN_RECTANGULARITY = None  # Минимальное соотношение сторон для прямоугольности (None = отключено)
MIN_NEIGHBORS = None  # Минимальное количество соседей для связности (None = отключено)
ADD_LINES = True  # Добавлять горизонтальные и вертикальные линии (True/False)
MIN_LINE_LENGTH = 10  # Минимальная длина линии для добавления
MIN_LINE_OVERLAP_RATIO = 0.3  # Минимальное отношение пересечения линии с штриховкой

def enhance_local_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance local contrast using CLAHE"""
    try:
        # Конвертируем в LAB цветовое пространство
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except:
        # Если CLAHE не работает, используем простое улучшение контраста
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        return np.array(enhancer.enhance(1.2))

def adaptive_binary_threshold(gray, method='gaussian', block_size=11, C=2):
    """Адаптивная бинаризация"""
    if method == 'gaussian':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C)
    elif method == 'mean':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C)
    else:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block_size, C)

def create_line_kernel(length, angle_deg):
    """Создание ядра для обнаружения линий под заданным углом"""
    angle_rad = np.deg2rad(angle_deg)
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    for i in range(length):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    return kernel

def detect_with_params(binary, kernel_size, density_threshold, angles):
    """Обнаружение штриховки с заданными параметрами"""
    # Создаем ядра для заданных углов
    kernels = [create_line_kernel(kernel_size, angle) for angle in angles]
    
    # Обнаружение линий
    all_lines = np.zeros_like(binary, dtype=np.float32)
    for kernel in kernels:
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        all_lines += detected.astype(np.float32)
    
    # Вычисляем плотность
    kernel_avg = np.ones((20, 20), dtype=np.float32) / 400
    density = cv2.filter2D(all_lines, -1, kernel_avg)
    density_normalized = density / density.max() if density.max() > 0 else density
    
    # Применяем порог
    return (density_normalized > density_threshold).astype(np.uint8) * 255

def is_rectangular_shape(contour, min_ratio=0.6):
    """Проверка, является ли контур приблизительно прямоугольным"""
    if len(contour) < 4:
        return False
    
    # Получаем ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(contour)
    
    # Проверяем соотношение сторон
    if w == 0 or h == 0:
        return False
    
    aspect_ratio = min(w, h) / max(w, h)
    return aspect_ratio >= min_ratio

def filter_noise_regions_optimized(mask, min_area=100, max_area=50000, min_rectangularity=0.6, min_neighbors=0):
    """Оптимизированная фильтрация шумовых регионов с более мягкими условиями"""
    
    # Находим все компоненты
    labeled, num = ndimage.label(mask)
    
    # Собираем информацию о компонентах
    component_info = []
    for i in range(1, num + 1):
        component_mask = (labeled == i).astype(np.uint8) * 255
        
        # Проверяем площадь
        area = np.sum(component_mask > 0)
        if area < min_area or (max_area is not None and area > max_area):
            continue
        
        # Если фильтр по прямоугольности отключен (0), пропускаем все компоненты
        if min_rectangularity > 0:
            # Находим контуры
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Проверяем прямоугольность (с более мягким условием)
            is_rectangular = any(is_rectangular_shape(contour, min_rectangularity) for contour in contours)
            if not is_rectangular:
                continue
        
        component_info.append((i, component_mask))
    
    # Проверяем связность с соседями (только если явно указано)
    if min_neighbors > 0 and len(component_info) > 1:
        final_components = []
        
        for i, (comp_id, component_mask) in enumerate(component_info):
            # Упрощенная проверка связности
            is_connected = False
            for j, (other_id, other_mask) in enumerate(component_info):
                if i == j:
                    continue
                
                # Проверяем расстояние между компонентами
                if check_components_proximity(component_mask, other_mask, max_distance=30):
                    is_connected = True
                    break
            
            # Если компонент связан или требование связности не строгое
            if is_connected or min_neighbors <= 1:
                final_components.append((comp_id, component_mask))
        
        component_info = final_components
    
    # Создаем финальную маску
    filtered_mask = np.zeros_like(mask)
    for _, component_mask in component_info:
        filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
    
    return filtered_mask

def check_components_proximity(mask1, mask2, max_distance=30):
    """Проверка близости двух компонентов"""
    # Находим границы компонентов
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours1 or not contours2:
        return False
    
    # Проверяем расстояние между контурами
    for cnt1 in contours1:
        # Получаем центр первого контура
        M1 = cv2.moments(cnt1)
        if M1['m00'] == 0:
            continue
        cx1 = int(M1['m10'] / M1['m00'])
        cy1 = int(M1['m01'] / M1['m00'])
        
        for cnt2 in contours2:
            # Проверяем, находится ли центр первого контура внутри второго или рядом с ним
            dist = cv2.pointPolygonTest(cnt2, (cx1, cy1), True)
            if dist >= 0 and dist <= max_distance:
                return True
    
    return False

def detect_hatching_enhanced_fixed(image,
                           kernel_sizes=None,
                           density_thresholds=None,
                           angles=None,
                           min_area=None,
                           max_area=None,
                           min_rectangularity=None,
                           min_neighbors=None,  # По умолчанию отключено для стабильности
                           enhance_contrast=True,
                           add_lines=None,
                           min_line_length=None,
                           min_line_overlap_ratio=None):
    """
    Исправленная функция обнаружения штриховки с оптимизацией
    
    Args:
        image: Входное RGB изображение
        kernel_sizes: Список размеров ядер для разной толщины линий
        density_thresholds: Список порогов плотности
        angles: Список углов (только 45° и 135°)
        min_area: Минимальная площадь региона
        max_area: Максимальная площадь региона
        min_rectangularity: Минимальное соотношение сторон для прямоугольности
        min_neighbors: Минимальное количество соседей для связности (0 = отключено)
        enhance_contrast: Улучшать ли локальный контраст
        add_lines: Добавлять горизонтальные и вертикальные линии (True/False)
        min_line_length: Минимальная длина линии для добавления
        min_line_overlap_ratio: Минимальное отношение пересечения линии с штриховкой
    
    Returns:
        wall_mask: Бинарная маска обнаруженных стен
    """
    
    # Используем глобальные переменные, если параметры не указаны
    if kernel_sizes is None:
        kernel_sizes = KERNEL_SIZES
    if density_thresholds is None:
        density_thresholds = DENSITY_THRESHOLDS
    if angles is None:
        angles = ANGLES
    if min_area is None:
        min_area = MIN_AREA
    if max_area is None:
        max_area = MAX_AREA
    if min_rectangularity is None:
        min_rectangularity = MIN_RECTANGULARITY
    if min_neighbors is None:
        min_neighbors = MIN_NEIGHBORS
    if add_lines is None:
        add_lines = ADD_LINES
    if min_line_length is None:
        min_line_length = MIN_LINE_LENGTH
    if min_line_overlap_ratio is None:
        min_line_overlap_ratio = MIN_LINE_OVERLAP_RATIO
    
    # Улучшение контраста
    if enhance_contrast:
        image = enhance_local_contrast(image)
    
    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Адаптивная бинаризация
    binary = adaptive_binary_threshold(gray, method='gaussian', block_size=11, C=2)
    
    # Многократное обнаружение с разными параметрами
    all_masks = []
    
    for kernel_size in kernel_sizes:
        for density_threshold in density_thresholds:
            # Обнаружение с текущими параметрами
            mask = detect_with_params(binary, kernel_size, density_threshold, angles)
            all_masks.append(mask)
    
    # Объединение масок
    combined_mask = np.zeros_like(binary)
    for mask in all_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Морфологическая обработка для улучшения связности
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Фильтрация шумовых регионов (применяем только если включен хотя бы один фильтр)
    if min_area is not None or max_area is not None or min_rectangularity is not None or min_neighbors is not None:
        # Устанавливаем значения по умолчанию для фильтров, которые не включены
        filter_min_area = min_area if min_area is not None else 0
        filter_max_area = max_area if max_area is not None else None
        filter_min_rectangularity = min_rectangularity if min_rectangularity is not None else 0
        filter_min_neighbors = min_neighbors if min_neighbors is not None else 0
        
        wall_mask = filter_noise_regions_optimized(
            combined_mask,
            min_area=filter_min_area,
            max_area=filter_max_area,
            min_rectangularity=filter_min_rectangularity,
            min_neighbors=filter_min_neighbors
        )
    else:
        wall_mask = combined_mask
    
    # Финальная морфологическая обработка
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    
    # Добавление горизонтальных и вертикальных линий, если включено
    if add_lines:
        wall_mask, horiz_count, vert_count = enhance_lines_with_hatching(
            wall_mask, gray, min_line_length, min_line_overlap_ratio)
        print(f"Добавлено горизонтальных линий: {horiz_count}, вертикальных линий: {vert_count}")
    
    return wall_mask

def extract_wall_polygons(wall_mask, min_vertices=4, epsilon_factor=2.0):
    """
    Extract wall polygons from binary mask

    Args:
        wall_mask: Binary mask of detected walls
        min_vertices: Minimum number of vertices for a valid polygon
        epsilon_factor: Approximation tolerance for polygon simplification

    Returns:
        List of polygon dictionaries with vertices and metadata
    """
    # Find contours in the mask (use RETR_LIST to get all wall segments, not just outer boundary)
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for i, contour in enumerate(contours, 1):
        # Calculate contour area
        area = cv2.contourArea(contour)

        if area < 100:  # Skip very small contours (noise)
            continue

        # Simplify contour to polygon
        epsilon = epsilon_factor  # Approximation tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= min_vertices:
            # Convert to list of points
            vertices = [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]

            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)

            polygon_data = {
                "id": f"wall_polygon_{i}",
                "vertices": vertices,
                "area": float(area),
                "perimeter": float(perimeter),
                "num_vertices": len(vertices),
                "source": "hatching"
            }

            polygons.append(polygon_data)

    return polygons

def point_to_line_segment_distance(point, line_p1, line_p2):
    """Расстояние от точки до отрезка"""
    # Длина отрезка
    line_len = np.linalg.norm(line_p2 - line_p1)
    if line_len == 0:
        return np.linalg.norm(point - line_p1)
    
    # Проекция точки на линию
    t = max(0, min(1, np.dot(point - line_p1, line_p2 - line_p1) / (line_len ** 2)))
    projection = line_p1 + t * (line_p2 - line_p1)
    
    return np.linalg.norm(point - projection)

def calculate_polygon_distance(poly1_vertices, poly2_vertices):
    """
    Вычисление минимального расстояния между границами двух полигонов
    
    Args:
        poly1_vertices: Вершины первого полигона
        poly2_vertices: Вершины второго полигона
        
    Returns:
        min_distance: Минимальное расстояние между границами полигонов
    """
    # Конвертируем вершины в numpy arrays
    points1 = np.array([[v['x'], v['y']] for v in poly1_vertices], dtype=np.float32)
    points2 = np.array([[v['x'], v['y']] for v in poly2_vertices], dtype=np.float32)
    
    # Для каждой точки первого полигона находим минимальное расстояние до второго
    min_distance = float('inf')
    
    for point in points1:
        # Вычисляем расстояние от точки до каждого отрезка второго полигона
        for i in range(len(points2)):
            p1 = points2[i]
            p2 = points2[(i + 1) % len(points2)]  # Замыкаем полигон
            
            # Расстояние от точки до отрезка
            dist = point_to_line_segment_distance(point, p1, p2)
            min_distance = min(min_distance, dist)
    
    return min_distance

def is_polygon_near_openings(polygon_vertices, doors, windows, proximity_threshold=20):
    """
    Проверка, находится ли полигон рядом с дверями или окнами
    
    Args:
        polygon_vertices: Вершины полигона
        doors: Список обнаруженных дверей
        windows: Список обнаруженных окон
        proximity_threshold: Максимальное расстояние для считывания "рядом"
    
    Returns:
        bool: True, если полигон находится рядом с дверью или окном
    """
    # Конвертируем вершины в numpy array
    points = np.array([[v['x'], v['y']] for v in polygon_vertices], dtype=np.float32)
    
    # Объединяем двери и окна
    all_openings = doors + windows
    
    for opening in all_openings:
        x, y, w, h = opening['bbox']
        
        # Проверяем расстояние от каждой вершины полигона до границ проема
        for point in points:
            px, py = point
            
            # Расстояние до каждой из четырех сторон прямоугольника проема
            # Левая сторона
            if abs(px - x) <= proximity_threshold and y <= py <= y + h:
                return True
            
            # Правая сторона
            if abs(px - (x + w)) <= proximity_threshold and y <= py <= y + h:
                return True
            
            # Верхняя сторона
            if abs(py - y) <= proximity_threshold and x <= px <= x + w:
                return True
            
            # Нижняя сторона
            if abs(py - (y + h)) <= proximity_threshold and x <= px <= x + w:
                return True
            
            # Расстояние до углов (для полигонов, которые могут быть рядом с углами проемов)
            corners = [
                (x, y),           # верхний левый
                (x + w, y),       # верхний правый
                (x, y + h),       # нижний левый
                (x + w, y + h)    # нижний правый
            ]
            
            for cx, cy in corners:
                if np.sqrt((px - cx)**2 + (py - cy)**2) <= proximity_threshold:
                    return True
    
    return False

def analyze_polygon_connectivity(polygons, proximity_threshold=15):
    """
    Анализ связности полигонов на основе близости их границ
    
    Args:
        polygons: Список полигонов с вершинами
        proximity_threshold: Максимальное расстояние для считывания связанными
    
    Returns:
        adjacency_matrix: Матрица смежности полигонов
        components: Списки индексов полигонов в каждом компоненте связности
    """
    n = len(polygons)
    adjacency_matrix = np.zeros((n, n), dtype=bool)
    
    # Строим матрицу смежности
    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_polygon_distance(
                polygons[i]['vertices'],
                polygons[j]['vertices']
            )
            
            if distance <= proximity_threshold:
                adjacency_matrix[i][j] = True
                adjacency_matrix[j][i] = True
    
    # Находим компоненты связности с помощью DFS
    visited = [False] * n
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        
        for neighbor in range(n):
            if adjacency_matrix[node][neighbor] and not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    return adjacency_matrix, components

def classify_polygons(polygons, components, doors=None, windows=None):
    """
    Классификация полигонов на стены и колонны на основе близости к дверям/окнам
    
    Args:
        polygons: Список полигонов
        components: Компоненты связности (используется как запасной вариант)
        doors: Список обнаруженных дверей
        windows: Список обнаруженных окон
    
    Returns:
        wall_polygons: Список полигонов стен
        pillar_polygons: Список полигонов колонн
    """
    if not components:
        return [], []
    
    # Если двери или окна не предоставлены, используем старый метод
    if doors is None or windows is None:
        print("   Двери или окна не предоставлены, используем классификацию по связности")
        return classify_polygons_by_connectivity(polygons, components)
    
    wall_polygons = []
    pillar_polygons = []
    
    # Проверяем каждый полигон на близость к дверям/окнам
    for i, polygon in enumerate(polygons):
        polygon_with_type = polygon.copy()
        
        # Если полигон рядом с дверью или окном, это стена
        if is_polygon_near_openings(polygon['vertices'], doors, windows, proximity_threshold=20):
            polygon_with_type['type'] = 'wall'
            polygon_with_type['classification_reason'] = 'near_opening'
            wall_polygons.append(polygon_with_type)
        else:
            # В качестве запасного варианта используем классификацию по связности
            # Находим компонент с наибольшей общей площадью (основной контур стен)
            component_areas = []
            for component in components:
                total_area = sum(polygons[idx]['area'] for idx in component)
                component_areas.append(total_area)
            
            main_component_idx = max(range(len(components)), key=lambda idx: component_areas[idx])
            main_component = set(components[main_component_idx])
            
            if i in main_component:
                polygon_with_type['type'] = 'wall'
                polygon_with_type['classification_reason'] = 'main_component'
                wall_polygons.append(polygon_with_type)
            else:
                polygon_with_type['type'] = 'pillar'
                polygon_with_type['classification_reason'] = 'isolated'
                pillar_polygons.append(polygon_with_type)
    
    # Выводим информацию о классификации для отладки
    print(f"   Классификация по близости к проемам:")
    print(f"   Найдено {len(wall_polygons)} полигонов стен, {len(pillar_polygons)} полигонов колонн")
    
    return wall_polygons, pillar_polygons

def classify_polygons_by_connectivity(polygons, components):
    """
    Запасная функция классификации полигонов на основе связности (оригинальный метод)
    """
    if not components:
        return [], []
    
    # Находим компонент с наибольшей общей площадью (основной контур стен)
    component_areas = []
    for component in components:
        total_area = sum(polygons[i]['area'] for i in component)
        component_areas.append(total_area)
    
    main_component_idx = max(range(len(components)), key=lambda i: component_areas[i])
    main_component = set(components[main_component_idx])
    
    wall_polygons = []
    pillar_polygons = []
    
    for i, polygon in enumerate(polygons):
        # Добавляем тип полигона
        polygon_with_type = polygon.copy()
        
        if i in main_component:
            polygon_with_type['type'] = 'wall'
            polygon_with_type['classification_reason'] = 'main_component'
            wall_polygons.append(polygon_with_type)
        else:
            polygon_with_type['type'] = 'pillar'
            polygon_with_type['classification_reason'] = 'isolated'
            pillar_polygons.append(polygon_with_type)
    
    # Выводим информацию о компонентах для отладки
    print(f"   Компоненты связности (по площади): {component_areas}")
    print(f"   Основной контур стен: компонент {main_component_idx} (площадь={component_areas[main_component_idx]:.1f})")
    print(f"   Найдено {len(wall_polygons)} полигонов стен, {len(pillar_polygons)} полигонов колонн")
    
    return wall_polygons, pillar_polygons

def extract_wall_polygons_with_classification(wall_mask, min_vertices=4, epsilon_factor=2.0, proximity_threshold=15, doors=None, windows=None):
    """
    Извлечение полигонов стен с автоматической классификацией на стены и колонны
    
    Args:
        wall_mask: Бинарная маска обнаруженных стен
        min_vertices: Минимальное количество вершин для валидного полигона
        epsilon_factor: Допуск аппроксимации для упрощения полигона
        proximity_threshold: Порог близости для определения связности
        doors: Список обнаруженных дверей
        windows: Список обнаруженных окон
    
    Returns:
        wall_polygons: Список полигонов стен
        pillar_polygons: Список полигонов колонн
    """
    # Находим контуры (используем RETR_LIST для получения всех сегментов)
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Извлекаем все полигоны
    all_polygons = []
    for i, contour in enumerate(contours, 1):
        area = cv2.contourArea(contour)
        
        if area < 100:  # Пропускаем очень маленькие контуры (шум)
            continue
        
        # Упрощаем контур до полигона
        epsilon = epsilon_factor
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= min_vertices:
            vertices = [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]
            perimeter = cv2.arcLength(contour, True)
            
            polygon_data = {
                "id": f"wall_polygon_{i}",
                "vertices": vertices,
                "area": float(area),
                "perimeter": float(perimeter),
                "num_vertices": len(vertices),
                "source": "hatching"
            }
            
            all_polygons.append(polygon_data)
    
    # Анализируем связность и классифицируем полигоны
    if len(all_polygons) > 1:
        _, components = analyze_polygon_connectivity(all_polygons, proximity_threshold)
        wall_polygons, pillar_polygons = classify_polygons(all_polygons, components, doors, windows)
    else:
        # Если только один полигон, это стена
        wall_polygons = all_polygons
        if wall_polygons:
            wall_polygons[0]['type'] = 'wall'
            wall_polygons[0]['classification_reason'] = 'single_polygon'
        pillar_polygons = []
    
    return wall_polygons, pillar_polygons

def categorize_junctions(junctions_dict):
    """Collect all junctions without changing existing indices"""
    all_junctions = []
    
    for jtype, points in junctions_dict.items():
        for p in points:
            junction_data = {**p, 'type': jtype}
            # Если у junction уже есть id, сохраняем его
            # Если нет, присваиваем новый индекс
            if 'id' not in junction_data:
                junction_data['id'] = len(all_junctions) + 1
            all_junctions.append(junction_data)
    
    return all_junctions

def analyze_detection_method(detection, wall_segments, all_junctions):
    """Determine how a detection was validated"""
    methods = ['DL']  # All start with DL

    x, y, w, h = detection['bbox']
    center_x, center_y = x + w/2, y + h/2

    # Check if on wall
    on_wall = False
    for seg in wall_segments:
        x1, y1 = seg['start']['x'], seg['start']['y']
        x2, y2 = seg['end']['x'], seg['end']['y']

        # Check distance to wall line
        dx = x2 - x1
        dy = y2 - y1
        line_len_sq = dx*dx + dy*dy

        if line_len_sq == 0:
            continue

        t = max(0, min(1, ((center_x - x1) * dx + (center_y - y1) * dy) / line_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.sqrt((center_x - proj_x)**2 + (center_y - proj_y)**2)

        if dist < 30:
            on_wall = True
            break

    if on_wall:
        methods.append('Wall-aligned')

    # Check for nearby junctions
    has_junctions = False
    for j in all_junctions:
        jx, jy = j['x'], j['y']
        if abs(jx - center_x) < w + 50 and abs(jy - center_y) < h + 50:
            has_junctions = True
            break

    if has_junctions:
        methods.append('Junctions')

    return methods

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def find_wall_for_opening(opening, wall_segments):
    """Find which wall segment an opening belongs to"""
    x, y, w, h = opening['bbox']
    center_x, center_y = x + w/2, y + h/2
    
    best_wall = None
    min_distance = float('inf')
    
    for i, seg in enumerate(wall_segments):
        x1, y1 = seg['start']['x'], seg['start']['y']
        x2, y2 = seg['end']['x'], seg['end']['y']
        
        # Calculate distance from opening center to wall line
        dist = point_to_line_distance(center_x, center_y, x1, y1, x2, y2)
        
        # Check if opening is within wall bounds (with some tolerance)
        wall_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if wall_length > 0:
            # Project opening center onto wall line
            t = max(0, min(1, ((center_x - x1) * (x2 - x1) + (center_y - y1) * (y2 - y1)) / (wall_length**2)))
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            
            # Check if projection is within opening bounds
            if (abs(proj_x - center_x) <= w/2 + 10 and 
                abs(proj_y - center_y) <= h/2 + 10 and
                dist < min_distance):
                min_distance = dist
                best_wall = i
    
    return best_wall

def find_junctions_for_opening(opening, all_junctions, max_distance=50):
    """
    Находит ближайшие junctions для проема (двери/окна)
    
    Args:
        opening: Проем с bbox {'x', 'y', 'width', 'height'}
        all_junctions: Список всех junctions
        max_distance: Максимальное расстояние для поиска junctions
        
    Returns:
        dict: Словарь с junctions для каждой стороны проема
    """
    # Поддерживаем оба формата bbox: {'x', 'y', 'width', 'height'} и кортеж (x, y, w, h)
    if isinstance(opening['bbox'], dict):
        x, y, w, h = opening['bbox']['x'], opening['bbox']['y'], opening['bbox']['width'], opening['bbox']['height']
    else:
        # Если bbox это кортеж или список
        x, y, w, h = opening['bbox']
    
    # Вычисляем центры сторон проема
    left_center = (x, y + h/2)
    right_center = (x + w, y + h/2)
    top_center = (x + w/2, y)
    bottom_center = (x + w/2, y + h)
    
    # Функция для поиска ближайшего junction к точке
    def find_nearest_junction(point, junctions, max_dist):
        min_dist = float('inf')
        nearest_junction = None
        
        for junction in junctions:
            jx, jy = junction['x'], junction['y']
            dist = np.sqrt((jx - point[0])**2 + (jy - point[1])**2)
            
            if dist < min_dist and dist <= max_dist:
                min_dist = dist
                nearest_junction = junction
        
        return nearest_junction
    
    # Находим junctions для каждой стороны
    junctions_for_opening = {}
    
    # Левая сторона
    left_junction = find_nearest_junction(left_center, all_junctions, max_distance)
    if left_junction:
        junctions_for_opening['left'] = {
            'junction_id': left_junction.get('id'),
            'junction_name': f"J{left_junction.get('id')}" if left_junction.get('id') else None,
            'junction_type': left_junction.get('type'),
            'x': left_junction['x'],
            'y': left_junction['y'],
            'distance': np.sqrt((left_junction['x'] - left_center[0])**2 + (left_junction['y'] - left_center[1])**2)
        }
    
    # Правая сторона
    right_junction = find_nearest_junction(right_center, all_junctions, max_distance)
    if right_junction:
        junctions_for_opening['right'] = {
            'junction_id': right_junction.get('id'),
            'junction_name': f"J{right_junction.get('id')}" if right_junction.get('id') else None,
            'junction_type': right_junction.get('type'),
            'x': right_junction['x'],
            'y': right_junction['y'],
            'distance': np.sqrt((right_junction['x'] - right_center[0])**2 + (right_junction['y'] - right_center[1])**2)
        }
    
    # Верхняя сторона
    top_junction = find_nearest_junction(top_center, all_junctions, max_distance)
    if top_junction:
        junctions_for_opening['up'] = {
            'junction_id': top_junction.get('id'),
            'junction_name': f"J{top_junction.get('id')}" if top_junction.get('id') else None,
            'junction_type': top_junction.get('type'),
            'x': top_junction['x'],
            'y': top_junction['y'],
            'distance': np.sqrt((top_junction['x'] - top_center[0])**2 + (top_junction['y'] - top_center[1])**2)
        }
    
    # Нижняя сторона
    bottom_junction = find_nearest_junction(bottom_center, all_junctions, max_distance)
    if bottom_junction:
        junctions_for_opening['down'] = {
            'junction_id': bottom_junction.get('id'),
            'junction_name': f"J{bottom_junction.get('id')}" if bottom_junction.get('id') else None,
            'junction_type': bottom_junction.get('type'),
            'x': bottom_junction['x'],
            'y': bottom_junction['y'],
            'distance': np.sqrt((bottom_junction['x'] - bottom_center[0])**2 + (bottom_junction['y'] - bottom_center[1])**2)
        }
    
    return junctions_for_opening

def detect_pillars_for_export(image, wall_mask, wall_segments, min_area=100, max_area=60000):
    """
    Detect pillars with wall hatching texture (standalone only)
    МОДИФИЦИРОВАННАЯ ВЕРСИЯ: теперь использует только изолированные полигоны
    """
    # В новой версии эта функция просто возвращает пустой список,
    # так как колонны определяются через extract_wall_polygons_with_classification
    print("   Примечание: колонны теперь определяются как изолированные полигоны")
    return []

def extract_rooms_from_walls(wall_segments):
    """Extract room polygons from wall segments"""
    if not wall_segments:
        return []
    
    # Collect all wall endpoints
    all_points = []
    for seg in wall_segments:
        all_points.append([seg['start']['x'], seg['start']['y']])
        all_points.append([seg['end']['x'], seg['end']['y']])
    
    all_points = np.array(all_points)
    
    if len(all_points) < 3:
        return []
    
    # Create Delaunay triangulation
    try:
        tri = Delaunay(all_points)
    except:
        return []
    
    # Build edge set with length threshold
    alpha = 150  # Threshold distance
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx = simplex[i]
            p2_idx = simplex[(i+1)%3]
            p1 = all_points[p1_idx]
            p2 = all_points[p2_idx]
            length = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

            if length < alpha:
                edge = tuple(sorted([p1_idx, p2_idx]))
                edges.add(edge)

    # Find boundary edges (appear only once in triangle set)
    from collections import defaultdict
    edge_count = defaultdict(int)
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx = simplex[i]
            p2_idx = simplex[(i+1)%3]
            edge = tuple(sorted([p1_idx, p2_idx]))
            if edge in edges:
                edge_count[edge] += 1

    # Boundary edges appear in only one triangle
    boundary_edges = [e for e, count in edge_count.items() if count == 1]

    # Build adjacency graph
    graph = defaultdict(list)
    for e in boundary_edges:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])

    # Trace boundary
    if boundary_edges:
        start_idx = min(graph.keys(), key=lambda i: (all_points[i][0], all_points[i][1]))
        perimeter_indices = [start_idx]
        current = start_idx
        visited = {start_idx}

        while len(visited) < len(graph):
            neighbors = [n for n in graph[current] if n not in visited]
            if not neighbors:
                break
            next_idx = neighbors[0]
            perimeter_indices.append(next_idx)
            visited.add(next_idx)
            current = next_idx

        perimeter_points = all_points[perimeter_indices]
        
        # Create room from perimeter
        room = {
            'id': 'room_1',
            'vertices': [{'x': float(p[0]), 'y': float(p[1])} for p in perimeter_points],
            'wall_ids': [f'wall_{i+1}' for i in range(len(wall_segments))]
        }
        
        return [room]
    
    return []

def find_building_extremes(walls, pillar_polygons=None):
    """
    Находит минимальные и максимальные координаты X и Y по всем стенам и колоннам
    
    Args:
        walls: Список стен с координатами start и end
        pillar_polygons: Список полигонов колонн (опционально)
        
    Returns:
        dict: Словарь с min_x, max_x, min_y, max_y
    """
    if not walls and not pillar_polygons:
        return None
    
    # Инициализируем экстремальные значения
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    # Обрабатываем стены
    if walls:
        for wall in walls:
            x1, y1 = wall['start']['x'], wall['start']['y']
            x2, y2 = wall['end']['x'], wall['end']['y']
            
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)
    
    # Обрабатываем полигоны колонн
    if pillar_polygons:
        for polygon in pillar_polygons:
            for vertex in polygon['vertices']:
                x, y = vertex['x'], vertex['y']
                
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    return {
        'min_x': float(min_x),
        'max_x': float(max_x),
        'min_y': float(min_y),
        'max_y': float(max_y)
    }

def create_foundation_polygon(extremes):
    """
    Создает полигон фундамента на основе экстремальных координат
    
    Args:
        extremes: Словарь с min_x, max_x, min_y, max_y
        
    Returns:
        dict: Полигон фундамента с вершинами
    """
    if not extremes:
        return None
    
    # Создаем вершины прямоугольника фундамента
    vertices = [
        {"x": extremes['min_x'], "y": extremes['min_y']},  # нижний левый
        {"x": extremes['max_x'], "y": extremes['min_y']},  # нижний правый
        {"x": extremes['max_x'], "y": extremes['max_y']},  # верхний правый
        {"x": extremes['min_x'], "y": extremes['max_y']}   # верхний левый
    ]
    
    # Вычисляем площадь и периметр
    width = extremes['max_x'] - extremes['min_x']
    height = extremes['max_y'] - extremes['min_y']
    area = width * height
    perimeter = 2 * (width + height)
    
    return {
        "id": "foundation_1",
        "type": "foundation",
        "vertices": vertices,
        "area": float(area),
        "perimeter": float(perimeter),
        "num_vertices": 4,
        "source": "building_extremes"
    }

def filter_junctions_by_foundation(junctions, foundation):
    """
    Фильтрует junctions, оставляя только те, что находятся внутри фундамента
    
    Args:
        junctions: Список junctions с координатами x, y
        foundation: Полигон фундамента с вершинами
        
    Returns:
        list: Отфильтрованный список junctions
    """
    if not foundation or not junctions:
        return junctions
    
    # Получаем вершины фундамента
    vertices = foundation['vertices']
    
    # Создаем функцию для проверки точки внутри полигона
    def point_in_polygon(point, polygon):
        """Проверяет, находится ли точка внутри полигона"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]['x'], polygon[i]['y']
            xj, yj = polygon[j]['x'], polygon[j]['y']
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    # Фильтруем junctions
    filtered_junctions = []
    for junction in junctions:
        point = (junction['x'], junction['y'])
        if point_in_polygon(point, vertices):
            filtered_junctions.append(junction)
    
    return filtered_junctions

def find_building_outline_from_junctions(junctions, pillar_polygons=None, openings=None):
    """
    Создает точный контур здания на основе junctions, следуя за фактической формой дома
    с правильной логикой поворотов только под 90 градусов и приоритетом выбора:
    1) направо, 2) прямо, 3) налево.
    
    Args:
        junctions: Список junction points
        pillar_polygons: Список полигонов колонн для избежания прохода через них
        
    Returns:
        dict: Полигон контура здания с вершинами
    """
    if not junctions:
        return None
    
    # Если junctions слишком мало, используем прямоугольник по экстремальным точкам
    if len(junctions) < 3:
        min_x = min(j['x'] for j in junctions)
        max_x = max(j['x'] for j in junctions)
        min_y = min(j['y'] for j in junctions)
        max_y = max(j['y'] for j in junctions)
        
        vertices = [
            {"x": min_x, "y": min_y},  # нижний левый
            {"x": max_x, "y": min_y},  # нижний правый
            {"x": max_x, "y": max_y},  # верхний правый
            {"x": min_x, "y": max_y}   # верхний левый
        ]
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        perimeter = 2 * (width + height)
        
        return {
            "id": "building_outline_1",
            "type": "building_outline",
            "vertices": vertices,
            "area": float(area),
            "perimeter": float(perimeter),
            "num_vertices": len(vertices),
            "source": "junction_extremes"
        }
    
    # Функция для вычисления расстояния между двумя точками
    def distance(p1, p2):
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    
    # Функция для проверки, пересекает ли отрезок колонну
    def line_intersects_pillar(p1, p2, pillar):
        vertices = pillar['vertices']
        
        def point_in_polygon(point, polygon):
            x, y = point['x'], point['y']
            n = len(polygon)
            inside = False
            
            j = n - 1
            for i in range(n):
                xi, yi = polygon[i]['x'], polygon[i]['y']
                xj, yj = polygon[j]['x'], polygon[j]['y']
                
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            
            return inside
        
        if point_in_polygon(p1, vertices) or point_in_polygon(p2, vertices):
            return True
        
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            v1 = vertices[i]
            v2 = vertices[j]
            
            def segments_intersect(p1, p2, p3, p4):
                def ccw(A, B, C):
                    return (C['y'] - A['y']) * (B['x'] - A['x']) > (B['y'] - A['y']) * (C['x'] - A['x'])
                
                return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
            
            if segments_intersect(p1, p2, v1, v2):
                return True
        
        return False
    
    # Функция для поиска соседних junctions в указанном направлении с допуском
    def find_neighbors_in_direction(current_point, junctions, visited, direction, tolerance=20):
        neighbors = []
        
        for i, junction in enumerate(junctions):
            if i in visited:
                continue
                
            dx = junction['x'] - current_point['x']
            dy = junction['y'] - current_point['y']
            dist = distance(current_point, junction)
            
            # Проверяем направление движения с допуском tolerance
            # Фильтруем только по направлению, без ограничений по расстоянию
            if direction == 'right' and abs(dy) <= tolerance and dx > 0:
                neighbors.append((i, junction, dist))
            elif direction == 'left' and abs(dy) <= tolerance and dx < 0:
                neighbors.append((i, junction, dist))
            elif direction == 'up' and abs(dx) <= tolerance and dy < 0:  # dy < 0 так как Y растет вниз
                neighbors.append((i, junction, dist))
            elif direction == 'down' and abs(dx) <= tolerance and dy > 0:
                neighbors.append((i, junction, dist))
        
        # Сортируем по расстоянию и возвращаем всех подходящих соседей
        neighbors.sort(key=lambda x: x[2])
        return neighbors
    
    # Функция для определения возможных направлений движения из текущей точки
    # Приоритет: 1) направо (поворот по часовой стрелке), 2) прямо (продолжение), 3) налево (поворот против часовой)
    def get_possible_directions(current_direction):
        if current_direction == 'right':  # Движемся вправо (восток)
            # 1) направо = поворот вниз (юг) - по часовой стрелке
            # 2) прямо = продолжение вправо (восток)
            # 3) налево = поворот вверх (север) - против часовой стрелки
            return ['down', 'right', 'up']
        elif current_direction == 'up':  # Движемся вверх (север)
            # 1) направо = поворот вправо (восток) - по часовой стрелке
            # 2) прямо = продолжение вверх (север)
            # 3) налево = поворот влево (запад) - против часовой стрелки
            return ['right', 'up', 'left']
        elif current_direction == 'left':  # Движемся влево (запад)
            # 1) направо = поворот вверх (север) - по часовой стрелке
            # 2) прямо = продолжение влево (запад)
            # 3) налево = поворот вниз (юг) - против часовой стрелки
            return ['up', 'left', 'down']
        elif current_direction == 'down':  # Движемся вниз (юг)
            # 1) направо = поворот влево (запад) - по часовой стрелке
            # 2) прямо = продолжение вниз (юг)
            # 3) налево = поворот вправо (восток) - против часовой стрелки
            return ['left', 'down', 'right']
        return ['right', 'up', 'left', 'down']
    
    # Находим самую нижнюю-левую junction как начальную
    start_idx = min(range(len(junctions)), key=lambda i: (junctions[i]['y'], junctions[i]['x']))
    
    # Определяем начальное направление на основе ближайшего окна
    start_point = junctions[start_idx]
    current_direction = 'right'  # По умолчанию
    
    # Если есть информация об окнах, используем её для определения направления
    if openings:
        try:
            # Находим ближайшее окно к начальной точке
            min_dist = float('inf')
            closest_window = None
            
            for opening in openings:
                if opening['type'] == 'window':
                    x, y, w, h = opening['bbox']['x'], opening['bbox']['y'], opening['bbox']['width'], opening['bbox']['height']
                    center_x, center_y = x + w/2, y + h/2
                    dist = np.sqrt((center_x - start_point['x'])**2 + (center_y - start_point['y'])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_window = opening
            
            if closest_window:
                x, y, w, h = closest_window['bbox']['x'], closest_window['bbox']['y'], closest_window['bbox']['width'], closest_window['bbox']['height']
                
                # Определяем ориентацию окна
                if w > h * 1.5:  # Горизонтальное окно
                    # Определяем, в какую сторону от окна находится начальная точка
                    if start_point['x'] < x:  # Точка слева от окна
                        current_direction = 'right'
                    else:  # Точка справа от окна
                        current_direction = 'left'
                elif h > w * 1.5:  # Вертикальное окно
                    # Определяем, в какую сторону от окна находится начальная точка
                    if start_point['y'] < y:  # Точка выше окна
                        current_direction = 'down'
                    else:  # Точка ниже окна
                        current_direction = 'up'
                
                print(f"   Начальное направление определено как '{current_direction}' на основе ближайшего окна")
        except Exception as e:
            # Если произошла ошибка, используем направление по умолчанию
            print(f"   Ошибка при определении начального направления: {e}")
            pass
    
    # Начинаем построение контура
    outline_points = [junctions[start_idx]]
    visited = {start_idx}
    current_idx = start_idx
    
    max_iterations = len(junctions) * 2  # Предотвращение бесконечного цикла
    iteration = 0
    
    while iteration < max_iterations:
        current_point = junctions[current_idx]
        next_point = None
        next_idx = None
        
        # Получаем возможные направления в порядке приоритета
        possible_directions = get_possible_directions(current_direction)
        
        # Ищем следующую точку в каждом из возможных направлений
        for direction in possible_directions:
            neighbors = find_neighbors_in_direction(current_point, junctions, visited, direction)
            
            if neighbors:
                # Проверяем, не пересекает ли путь колонну
                for idx, junction, dist in neighbors:
                    passes_through_pillar = False
                    if pillar_polygons:
                        for pillar in pillar_polygons:
                            if line_intersects_pillar(current_point, junction, pillar):
                                passes_through_pillar = True
                                break
                    
                    if not passes_through_pillar:
                        next_point = junction
                        next_idx = idx
                        current_direction = direction
                        break
                
                if next_point is not None:
                    break
        
        # Если не нашли следующую точку в приоритетных направлениях, ищем любую ближайшую
        if next_point is None:
            # Сначала проверяем, можно ли вернуться к начальной точке для завершения контура
            start_point = junctions[start_idx]
            dist_to_start = distance(current_point, start_point)
            
            # Проверяем, достаточно ли точек в контуре для замыкания
            if len(outline_points) > 3 and dist_to_start < 500:  # Условное максимальное расстояние для замыкания
                # Проверяем, не пересекает ли путь к начальной точке колонну
                passes_through_pillar = False
                if pillar_polygons:
                    for pillar in pillar_polygons:
                        if line_intersects_pillar(current_point, start_point, pillar):
                            passes_through_pillar = True
                            break
                
                if not passes_through_pillar:
                    next_point = start_point
                    next_idx = start_idx
            
            # Если не удалось замкнуть контур, ищем ближайшую непосещенную точку
            if next_point is None:
                min_dist = float('inf')
                for i, junction in enumerate(junctions):
                    if i not in visited:
                        dist = distance(current_point, junction)
                        if dist < min_dist:
                            min_dist = dist
                            next_point = junction
                            next_idx = i
                
                if next_point is not None:
                    # Определяем новое направление на основе положения следующей точки
                    dx = next_point['x'] - current_point['x']
                    dy = next_point['y'] - current_point['y']
                    
                    if abs(dx) > abs(dy):
                        current_direction = 'right' if dx > 0 else 'left'
                    else:
                        current_direction = 'up' if dy < 0 else 'down'  # dy < 0 так как Y растет вниз
        
        # Если не нашли следующую точку, завершаем контур
        if next_point is None:
            break
        
        # Добавляем следующую точку в контур
        outline_points.append(next_point)
        visited.add(next_idx)
        current_idx = next_idx
        
        # Если мы вернулись к начальной точке, завершаем контур
        if current_idx == start_idx and len(outline_points) > 3:
            break
        
        iteration += 1
    
    # Если контур не замкнулся, пытаемся замкнуть его
    if len(outline_points) > 2 and outline_points[-1] != junctions[start_idx]:
        # Проверяем, можно ли замкнуть контур напрямую
        last_point = outline_points[-1]
        start_point = junctions[start_idx]
        
        # Проверяем, не пересекает ли замыкающий отрезок колонну
        can_close_directly = True
        if pillar_polygons:
            for pillar in pillar_polygons:
                if line_intersects_pillar(last_point, start_point, pillar):
                    can_close_directly = False
                    break
        
        if can_close_directly:
            # Замыкаем контур
            outline_points.append(start_point)
        else:
            # Ищем промежуточные точки для замыкания контура
            # Находим ближайшую непосещенную точку к начальной
            min_dist = float('inf')
            closest_to_start = None
            closest_idx = None
            
            for i, junction in enumerate(junctions):
                if i not in visited:
                    dist = distance(junction, start_point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_to_start = junction
                        closest_idx = i
            
            if closest_to_start is not None:
                # Добавляем промежуточную точку и затем начальную
                outline_points.append(closest_to_start)
                outline_points.append(start_point)
    
    # Если контур получился слишком маленьким, возвращаем пустой контур
    if len(outline_points) < 3:
        return {
            "id": "building_outline_1",
            "type": "building_outline",
            "vertices": [],
            "area": 0.0,
            "perimeter": 0.0,
            "num_vertices": 0,
            "source": "insufficient_outline_points"
        }
    
    # Создаем полигон с информацией о junctions для каждой вершины
    vertices = []
    for p in outline_points:
        # Находим соответствующий junction в исходном списке
        junction_id = None
        junction_type = None
        for j in junctions:
            if abs(j['x'] - p['x']) < 1 and abs(j['y'] - p['y']) < 1:
                junction_id = j.get('id')
                junction_type = j['type']
                break
        
        vertex_data = {
            "x": p["x"],
            "y": p["y"]
        }
        
        # Добавляем информацию о junction, если она найдена
        if junction_id is not None:
            vertex_data["junction_name"] = f"J{junction_id}"
            vertex_data["junction_id"] = junction_id
            vertex_data["junction_type"] = junction_type
        
        vertices.append(vertex_data)
    
    # Вычисляем площадь и периметр
    area = 0
    perimeter = 0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        area += vertices[i]["x"] * vertices[j]["y"] - vertices[j]["x"] * vertices[i]["y"]
        perimeter += np.sqrt((vertices[j]["x"] - vertices[i]["x"])**2 +
                            (vertices[j]["y"] - vertices[i]["y"])**2)
    
    area = abs(area) / 2
    
    return {
        "id": "building_outline_1",
        "type": "building_outline",
        "vertices": vertices,
        "area": float(area),
        "perimeter": float(perimeter),
        "num_vertices": len(vertices),
        "source": "junction_contour_fixed"
    }

def estimate_wall_thickness(wall_segments):
    """Estimate wall thickness from parallel segments"""
    if len(wall_segments) < 2:
        return 10  # Default thickness in pixels
    
    thickness_samples = []
    
    for i, seg1 in enumerate(wall_segments[:20]):  # Sample first 20
        p1_start = np.array([seg1['start']['x'], seg1['start']['y']])
        p1_end = np.array([seg1['end']['x'], seg1['end']['y']])
        vec1 = p1_end - p1_start
        len1 = np.linalg.norm(vec1)

        if len1 < 10:
            continue

        vec1_norm = vec1 / len1

        # Find parallel segments
        for seg2 in wall_segments[i+1:]:
            p2_start = np.array([seg2['start']['x'], seg2['start']['y']])
            p2_end = np.array([seg2['end']['x'], seg2['end']['y']])
            vec2 = p2_end - p2_start
            len2 = np.linalg.norm(vec2)

            if len2 < 10:
                continue

            vec2_norm = vec2 / len2

            # Check if parallel (dot product ~ ±1)
            dot = abs(np.dot(vec1_norm, vec2_norm))
            if dot > 0.95:  # Parallel
                # Measure perpendicular distance
                to_seg2 = p2_start - p1_start
                dist_to_seg2 = np.linalg.norm(to_seg2)

                if dist_to_seg2 > 0:
                    to_seg2_norm = to_seg2 / dist_to_seg2
                    # Cross product for 2D vectors
                    perp_dist = abs(vec1_norm[0] * to_seg2_norm[1] - vec1_norm[1] * to_seg2_norm[0]) * dist_to_seg2

                    if 5 < perp_dist < 50:  # Reasonable wall thickness
                        thickness_samples.append(perp_dist)

    if thickness_samples:
        median_thickness = np.median(thickness_samples)
        return min(median_thickness, 20)  # Cap at 20 pixels
    
    return 10  # Default thickness

def create_colored_svg(output_path, image_shape, walls, doors, windows, pillars, rooms, pillar_polygons, foundation, building_outline, scale_factor):
    """Create colored SVG file with all detected objects"""
    height, width = image_shape[:2]
    
    # Scale back to original resolution
    orig_width = int(width * scale_factor)
    orig_height = int(height * scale_factor)
    
    dwg = svgwrite.Drawing(output_path, size=(orig_width, orig_height), profile='tiny')
    
    # Define colors for different object types
    colors = {
        'wall_hatching': '#8B4513',  # Brown for hatching walls
        'wall_dl': '#4169E1',         # Royal blue for DL walls
        'window': '#00CED1',          # Dark turquoise for windows
        'door': '#228B22',            # Forest green for doors
        'pillar': '#FF6347',          # Tomato red for pillars
        'pillar_polygon': '#FFD700',  # Gold for pillar polygons
        'room': '#F0F8FF',            # Alice blue (light) for rooms
        'room_border': '#708090',     # Slate gray for room borders
        'foundation': '#FF0000',      # Red for foundation
        'building_outline': '#FFFF00'  # Yellow for building outline
    }
    
    # Draw rooms (background)
    if rooms:
        room_group = dwg.add(dwg.g(id='rooms'))
        for room in rooms:
            points = [(float(v['x'] * scale_factor), float(v['y'] * scale_factor)) for v in room['vertices']]
            if len(points) >= 3:
                room_group.add(dwg.polygon(
                    points,
                    fill=colors['room'],
                    stroke=colors['room_border'],
                    stroke_width=2,
                    opacity=0.3
                ))
    
    # Draw walls
    wall_group = dwg.add(dwg.g(id='walls'))
    for wall in walls:
        x1, y1 = wall['start']['x'], wall['start']['y']
        x2, y2 = wall['end']['x'], wall['end']['y']
        color = colors['wall_hatching'] if wall['source'] == 'hatching' else colors['wall_dl']
        wall_group.add(dwg.line(
            start=(float(x1 * scale_factor), float(y1 * scale_factor)),
            end=(float(x2 * scale_factor), float(y2 * scale_factor)),
            stroke=color,
            stroke_width=int(6 * scale_factor),
            stroke_linecap='round'
        ))
    
    # Draw pillars
    pillar_group = dwg.add(dwg.g(id='pillars'))
    for i, pillar in enumerate(pillars, 1):
        x, y, w, h = pillar['x'], pillar['y'], pillar['width'], pillar['height']
        pillar_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill=colors['pillar'],
            fill_opacity=0.7,
            stroke='darkred',
            stroke_width=int(2 * scale_factor)
        ))
        # Add label
        pillar_group.add(dwg.text(
            f'P{i}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='white',
            font_size=int(min(w, h) * 0.4 * scale_factor),
            font_weight='bold'
        ))
    
    # Draw windows
    window_group = dwg.add(dwg.g(id='windows'))
    for i, window in enumerate(windows, 1):
        x, y, w, h = window['bbox']
        window_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill=colors['window'],
            fill_opacity=0.7,
            stroke='darkcyan',
            stroke_width=int(3 * scale_factor)
        ))
        # Add label
        window_group.add(dwg.text(
            f'W{i}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='black',
            font_size=int(min(w, h) * 0.5 * scale_factor),
            font_weight='bold'
        ))
    
    # Draw doors
    door_group = dwg.add(dwg.g(id='doors'))
    for i, door in enumerate(doors, 1):
        x, y, w, h = door['bbox']
        door_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill=colors['door'],
            fill_opacity=0.7,
            stroke='darkgreen',
            stroke_width=int(3 * scale_factor)
        ))
        # Add label
        door_group.add(dwg.text(
            f'D{i}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='white',
            font_size=int(min(w, h) * 0.5 * scale_factor),
            font_weight='bold'
        ))
    
    # Draw pillar polygons
    if pillar_polygons:
        pillar_polygon_group = dwg.add(dwg.g(id='pillar_polygons'))
        for polygon in pillar_polygons:
            points = [(float(v['x'] * scale_factor), float(v['y'] * scale_factor)) for v in polygon['vertices']]
            if len(points) >= 3:
                pillar_polygon_group.add(dwg.polygon(
                    points,
                    fill=colors['pillar_polygon'],
                    stroke='darkred',
                    stroke_width=int(2 * scale_factor),
                    opacity=0.7
                ))
                # Add label
                if points:
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                    pillar_polygon_group.add(dwg.text(
                        'P',
                        insert=(center_x, center_y),
                        text_anchor='middle',
                        fill='black',
                        font_size=int(12 * scale_factor),
                        font_weight='bold'
                    ))
   
    # Draw foundation (red outline)
    if foundation:
        foundation_group = dwg.add(dwg.g(id='foundation'))
        points = [(float(v['x'] * scale_factor), float(v['y'] * scale_factor)) for v in foundation['vertices']]
        if len(points) >= 3:
            foundation_group.add(dwg.polygon(
                points,
                fill='none',
                stroke=colors['foundation'],
                stroke_width=int(3 * scale_factor),
                opacity=1.0
            ))
            # Add label
            if points:
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                foundation_group.add(dwg.text(
                    'FOUNDATION',
                    insert=(center_x, center_y),
                    text_anchor='middle',
                    fill=colors['foundation'],
                    font_size=int(14 * scale_factor),
                    font_weight='bold',
                    opacity=0.8
                ))
  
    # Draw building outline (yellow outline)
    if building_outline:
        building_outline_group = dwg.add(dwg.g(id='building_outline'))
        points = [(float(v['x'] * scale_factor), float(v['y'] * scale_factor)) for v in building_outline['vertices']]
        if len(points) >= 3:
            building_outline_group.add(dwg.polygon(
                points,
                fill='none',
                stroke=colors['building_outline'],
                stroke_width=int(3 * scale_factor),
                opacity=1.0
            ))
            # Add label
            if points:
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                building_outline_group.add(dwg.text(
                    'BUILDING OUTLINE',
                    insert=(center_x, center_y),
                    text_anchor='middle',
                    fill=colors['building_outline'],
                    font_size=int(14 * scale_factor),
                    font_weight='bold',
                    opacity=0.8
                ))
    
    # Add legend
    legend_group = dwg.add(dwg.g(id='legend', font_size=int(14 * scale_factor)))
    legend_x = int(20 * scale_factor)
    legend_y = int(30 * scale_factor)
    line_height = int(25 * scale_factor)
    
    legend_items = [
        ('Walls (Hatching)', colors['wall_hatching']),
        ('Walls (DL)', colors['wall_dl']),
        ('Windows', colors['window']),
        ('Doors', colors['door']),
        ('Pillars', colors['pillar']),
        ('Pillar Polygons', colors['pillar_polygon']),
        ('Rooms', colors['room']),
        ('Foundation', colors['foundation']),
        ('Building Outline', colors['building_outline'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + i * line_height
        
        # Draw color box
        legend_group.add(dwg.rect(
            insert=(legend_x, y_pos - int(10 * scale_factor)),
            size=(int(20 * scale_factor), int(15 * scale_factor)),
            fill=color,
            stroke='black',
            stroke_width=1
        ))
        
        # Add label
        legend_group.add(dwg.text(
            label,
            insert=(legend_x + int(30 * scale_factor), y_pos),
            fill='black',
            font_family='Arial'
        ))
    
    # Add title
    title_group = dwg.add(dwg.g(id='title'))
    title_group.add(dwg.text(
        'Floor Plan Detection Results',
        insert=(orig_width // 2, int(30 * scale_factor)),
        text_anchor='middle',
        fill='black',
        font_size=int(20 * scale_factor),
        font_weight='bold',
        font_family='Arial'
    ))
    
    # Add statistics
    stats_group = dwg.add(dwg.g(id='statistics', font_size=int(12 * scale_factor)))
    stats_text = [
        f"Walls: {len(walls)}",
        f"Windows: {len(windows)}",
        f"Doors: {len(doors)}",
        f"Pillars: {len(pillars)}",
        f"Pillar Polygons: {len(pillar_polygons)}",
        f"Rooms: {len(rooms)}",
        f"Foundations: {1 if foundation else 0}",
        f"Building Outlines: {1 if building_outline else 0}"
    ]
    
    for i, text in enumerate(stats_text):
        stats_group.add(dwg.text(
            text,
            insert=(orig_width - int(150 * scale_factor), legend_y + i * line_height),
            fill='black',
            font_family='Arial'
        ))
    
    dwg.save()
    return orig_width, orig_height

def main():
    """Main export function"""
    print("="*80)
    print("EXPORTING FLOOR PLAN OBJECTS TO JSON")
    print("="*80)

    image_path = 'plan_floor1.jpg'
    output_path = 'plan_floor1_objects.json'

    # Load model
    print("\n[1/6] Loading model...")
    try:
        model = get_model('hg_furukawa_original', 51)
        model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
        checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print("   Model loaded successfully")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return

    # Preprocess
    print("\n[2/6] Preprocessing...")
    try:
        img_orig = Image.open(image_path).convert('RGB')
        orig_width, orig_height = img_orig.size

        max_size = 2048
        w, h = orig_width, orig_height
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            img = img_orig
            scale = 1.0

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img_np = np.array(img)

        img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        print(f"   Image processed: {img_np.shape[1]}x{img_np.shape[0]} (scale: {scale:.2f})")
    except Exception as e:
        print(f"   Error preprocessing image: {e}")
        return

    # Inference
    print("\n[3/6] Running DL inference...")
    try:
        prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

        # Extract icons
        icons_logits = prediction[0, 33:44]
        icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

        door_prob = icons_pred[2]
        window_prob = icons_pred[1]

        threshold = 0.3
        door_mask = (door_prob > threshold).astype(np.uint8)
        window_mask = (window_prob > threshold).astype(np.uint8)

        _, doors = refine_dl_detections(door_mask, min_size=50, max_size=6000)
        _, windows = refine_dl_detections(window_mask, min_size=50, max_size=6000)
        
        print(f"   Raw detections: {len(doors)} doors, {len(windows)} windows")
    except Exception as e:
        print(f"   Error during inference: {e}")
        return

    # Detect walls
    print("\n[4/6] Detecting walls...")
    try:
        # Используем улучшенную функцию обнаружения штриховки
        # For wall segments, use MIN_AREA filter
        wall_mask_hatching = detect_hatching_enhanced_fixed(img_np)
        wall_segments_hatching = extract_wall_segments(wall_mask_hatching, min_length=50)

        # For wall polygons, load the reference strict mask
        # The issue is that MIN_AREA filter completely empties the mask when used with strict parameters
        # So we need to detect walls WITHOUT the area filter at all
        # This means we need to skip filter_noise_regions_optimized entirely
        try:
            import os
            ref_mask_path = 'enhanced_hatching_strict_mask.png'
            if os.path.exists(ref_mask_path):
                # Load reference mask at original resolution
                wall_mask_full_res = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
                # Downscale to match processing resolution
                wall_mask_polygons = cv2.resize(wall_mask_full_res, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_AREA)
                # Threshold to ensure binary
                _, wall_mask_polygons = cv2.threshold(wall_mask_polygons, 127, 255, cv2.THRESH_BINARY)
                print(f"   Loaded reference mask from {ref_mask_path}")
            else:
                # Fallback: generate without filtering
                print(f"   Reference mask not found, generating without area filter...")
                wall_mask_polygons = np.zeros_like(wall_mask_hatching)
        except Exception as e:
            print(f"   Error loading reference mask: {e}")
            wall_mask_polygons = np.zeros_like(wall_mask_hatching)

        wall_polygons_hatching, pillar_polygons_hatching = extract_wall_polygons_with_classification(
            wall_mask_polygons, doors=doors, windows=windows)
        print(f"   Extracted {len(wall_polygons_hatching)} wall polygons and {len(pillar_polygons_hatching)} pillar polygons from hatching")

        rooms_logits = prediction[0, 21:33]
        rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()
        wall_mask_dl = (rooms_pred == 2).astype(np.uint8) * 255
        wall_segments_dl = extract_wall_segments(wall_mask_dl, min_length=50)

        # Tag walls with source
        for seg in wall_segments_hatching:
            seg['source'] = 'hatching'
        for seg in wall_segments_dl:
            seg['source'] = 'DL'

        wall_segments = wall_segments_hatching + wall_segments_dl
        
        # Filter isolated wall segments
        def filter_connected_walls(wall_segments, min_component_size=5):
            """Keep only wall segments that belong to large connected components"""
            if not wall_segments:
                return []

            # Build adjacency graph
            from collections import defaultdict
            graph = defaultdict(set)

            def points_close(p1, p2, threshold=15):
                return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < threshold

            # Connect segments that share endpoints
            for i, seg1 in enumerate(wall_segments):
                for j, seg2 in enumerate(wall_segments):
                    if i >= j:
                        continue

                    if (points_close([seg1['start']['x'], seg1['start']['y']], [seg2['start']['x'], seg2['start']['y']]) or
                        points_close([seg1['start']['x'], seg1['start']['y']], [seg2['end']['x'], seg2['end']['y']]) or
                        points_close([seg1['end']['x'], seg1['end']['y']], [seg2['start']['x'], seg2['start']['y']]) or
                        points_close([seg1['end']['x'], seg1['end']['y']], [seg2['end']['x'], seg2['end']['y']])):
                        graph[i].add(j)
                        graph[j].add(i)

            # Find connected components using DFS
            visited = set()
            components = []

            def dfs(node, component):
                visited.add(node)
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, component)

            for i in range(len(wall_segments)):
                if i not in visited:
                    component = []
                    dfs(i, component)
                    components.append(component)

            # Keep components that meet size threshold
            valid_segments = []
            if components:
                for idx, comp in enumerate(components):
                    if len(comp) >= min_component_size:
                        valid_segments.extend([wall_segments[i] for i in comp])

            return valid_segments if valid_segments else wall_segments

        wall_segments = filter_connected_walls(wall_segments, min_component_size=5)
        
        print(f"   Detected: {len(wall_segments)} wall segments ({len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL)")
    except Exception as e:
        print(f"   Error detecting walls: {e}")
        return

    # Filter windows/doors
    print("\n[5/6] Processing openings...")
    try:
        # Remove nested boxes
        windows = remove_nested_boxes(windows, iou_threshold=0.15)
        doors = remove_nested_boxes(doors, iou_threshold=0.5)

        # Filter windows: edges must be on walls
        def filter_windows_strict(windows, wall_segments, max_distance=15):
            """Keep only windows whose edges are on walls"""
            valid = []
            for w in windows:
                x, y, width, height = w['bbox']

                # Check only edge midpoints
                edges = [
                    (x + width/2, y),              # top edge
                    (x + width/2, y + height),     # bottom edge
                    (x, y + height/2),             # left edge
                    (x + width, y + height/2),     # right edge
                ]

                # Check if at least one edge is on a wall
                on_wall = False
                for px, py in edges:
                    for seg in wall_segments:
                        x1, y1 = seg['start']['x'], seg['start']['y']
                        x2, y2 = seg['end']['x'], seg['end']['y']
                        dist = point_to_line_distance(px, py, x1, y1, x2, y2)
                        if dist < max_distance:
                            on_wall = True
                            break
                    if on_wall:
                        break

                if on_wall:
                    valid.append(w)

            return valid

        windows = filter_windows_strict(windows, wall_segments, max_distance=15)

        # Filter doors: reject if window_prob > door_prob at center
        def filter_door_vs_window(doors, door_prob, window_prob):
            """Reject doors where model predicts higher window probability"""
            valid_doors = []
            for door in doors:
                x, y, w, h = door['bbox']
                cx, cy = int(x + w/2), int(y + h/2)

                if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
                    d_conf = door_prob[cy, cx]
                    w_conf = window_prob[cy, cx]

                    # Keep only if door confidence is higher than window
                    if d_conf > w_conf:
                        valid_doors.append(door)
                else:
                    valid_doors.append(door)

            return valid_doors

        doors_before = len(doors)
        doors = filter_door_vs_window(doors, door_prob, window_prob)
        print(f"   After filtering: {len(windows)} windows, {len(doors)}/{doors_before} doors")

        # Extract junctions
        heatmaps = prediction[0, :21].cpu().data.numpy()
        junctions_dict = extract_junctions(prediction, threshold=threshold)
        all_junctions = categorize_junctions(junctions_dict)

        # Analyze detection methods
        for w in windows:
            w['methods'] = analyze_detection_method(w, wall_segments, all_junctions)
        for d in doors:
            d['methods'] = analyze_detection_method(d, wall_segments, all_junctions)

        # Detect pillars (standalone, not enclosed by walls) using enhanced hatching
        wall_mask = detect_hatching_enhanced_fixed(img_np, min_area=50, max_area=60000)
        pillars = detect_pillars_for_export(img_np, wall_mask, wall_segments, min_area=50, max_area=60000)
        
        print(f"   Detected: {len(pillars)} traditional pillars, {len(pillar_polygons_hatching)} polygon pillars, {len(all_junctions)} junctions")
    except Exception as e:
        print(f"   Error processing openings: {e}")
        return

    # Create JSON structure
    print("\n[6/6] Creating JSON structure...")
    try:
        # Estimate wall thickness
        wall_thickness_pixels = estimate_wall_thickness(wall_segments)
        wall_thickness_meters = 0.2  # Standard wall thickness in meters
        
        # Create JSON structure
        json_data = {
            "metadata": {
                "source_image": image_path,
                "scale_factor": float(scale),
                "units": "pixels",
                "wall_height": 3.0,
                "wall_thickness": wall_thickness_meters,
                "door_height": 2.1,
                "window_height": 1.5,
                "window_sill_height": 1.0,
                "timestamp": datetime.datetime.now().isoformat(),
                "model_info": {
                    "name": "hg_furukawa_original",
                    "weights": "model_best_val_loss_var.pkl"
                }
            },
            "walls": [],
            "wall_polygons": [],
            "pillar_polygons": [],  # Новое поле для полигонов колонн
            "openings": [],
            "pillars": [],
            "rooms": [],
            "junctions": [],
            "foundation": None,  # Поле для фундамента здания
            "building_outline": None,  # Поле для контура здания
            "statistics": {
                "walls": 0,
                "windows": 0,
                "doors": 0,
                "pillars": 0,
                "pillar_polygons": 0,  # Новое поле в статистике
                "rooms": 0,
                "junctions": 0,
                "foundations": 0,  # Новое поле в статистике для фундамента
                "building_outlines": 0  # Новое поле в статистике для контура здания
            }
        }

        # Export walls
        for i, seg in enumerate(wall_segments):
            wall_data = {
                "id": f"wall_{i+1}",
                "start": {"x": float(seg['start']['x']), "y": float(seg['start']['y'])},
                "end": {"x": float(seg['end']['x']), "y": float(seg['end']['y'])},
                "thickness": wall_thickness_meters,
                "height": 3.0,
                "source": seg['source']
            }
            json_data["walls"].append(wall_data)

        # Export wall polygons
        for polygon in wall_polygons_hatching:
            json_data["wall_polygons"].append(polygon)

        # Export pillar polygons
        for polygon in pillar_polygons_hatching:
            json_data["pillar_polygons"].append(polygon)

        # Export windows
        for i, window in enumerate(windows):
            x, y, w, h = window['bbox']
            wall_id = find_wall_for_opening(window, wall_segments)
            
            # Находим junctions для окна
            window_junctions = find_junctions_for_opening(window, all_junctions, max_distance=50)
            
            window_data = {
                "id": f"window_{i+1}",
                "type": "window",
                "bbox": {"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                "wall_id": f"wall_{wall_id+1}" if wall_id is not None else None,
                "height": 1.5,
                "sill_height": 1.0,
                "methods": window['methods'],
                "junctions": window_junctions  # Добавляем junctions
            }
            
            # Add confidence values
            cx, cy = int(x + w/2), int(y + h/2)
            if 0 <= cy < window_prob.shape[0] and 0 <= cx < window_prob.shape[1]:
                window_data["confidence"] = {
                    "window_prob": float(window_prob[cy, cx]),
                    "door_prob": float(door_prob[cy, cx])
                }
            
            json_data["openings"].append(window_data)

        # Export doors
        for i, door in enumerate(doors):
            x, y, w, h = door['bbox']
            wall_id = find_wall_for_opening(door, wall_segments)
            
            # Находим junctions для двери
            door_junctions = find_junctions_for_opening(door, all_junctions, max_distance=50)
            
            door_data = {
                "id": f"door_{i+1}",
                "type": "door",
                "bbox": {"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                "wall_id": f"wall_{wall_id+1}" if wall_id is not None else None,
                "height": 2.1,
                "methods": door['methods'],
                "junctions": door_junctions  # Добавляем junctions
            }
            
            # Add confidence values
            cx, cy = int(x + w/2), int(y + h/2)
            if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
                door_data["confidence"] = {
                    "door_prob": float(door_prob[cy, cx]),
                    "window_prob": float(window_prob[cy, cx])
                }
            
            json_data["openings"].append(door_data)

        # Export pillars from polygons (если они есть)
        for i, polygon in enumerate(pillar_polygons_hatching):
            # Вычисляем bounding box для полигона
            vertices = polygon['vertices']
            x_coords = [v['x'] for v in vertices]
            y_coords = [v['y'] for v in vertices]
            
            pillar_data = {
                "id": f"pillar_{i+1}",
                "type": "pillar_polygon",
                "bbox": {
                    "x": float(min(x_coords)),
                    "y": float(min(y_coords)),
                    "width": float(max(x_coords) - min(x_coords)),
                    "height": float(max(y_coords) - min(y_coords))
                },
                "vertices": vertices,  # Сохраняем и вершины
                "height": 3.0,
                "area": float(polygon['area']),
                "perimeter": float(polygon['perimeter']),
                "num_vertices": polygon['num_vertices'],
                "source": polygon['source']
            }
            json_data["pillars"].append(pillar_data)

        # Export traditional pillars (если они есть)
        for i, pillar in enumerate(pillars):
            pillar_data = {
                "id": f"pillar_{i+len(pillar_polygons_hatching)+1}",
                "type": "traditional_pillar",
                "bbox": {
                    "x": float(pillar['x']),
                    "y": float(pillar['y']),
                    "width": float(pillar['width']),
                    "height": float(pillar['height'])
                },
                "height": 3.0,
                "area": float(pillar['area']),
                "aspect_ratio": float(pillar['aspect_ratio'])
            }
            json_data["pillars"].append(pillar_data)

        # Export junctions
        for junction in all_junctions:
            junction_data = {
                "x": float(junction['x']),
                "y": float(junction['y']),
                "type": junction['type']
            }
            # Добавляем ID, если он существует
            if 'id' in junction:
                junction_data["id"] = junction['id']
            json_data["junctions"].append(junction_data)

        # Extract rooms
        rooms = extract_rooms_from_walls(wall_segments)
        for room in rooms:
            json_data["rooms"].append(room)

        # Create foundation from building extremes (including pillars)
        print("\n   Creating building foundation...")
        try:
            # Преобразуем wall_segments в правильный формат для find_building_extremes
            formatted_walls = []
            for seg in wall_segments:
                formatted_walls.append({
                    "start": {"x": seg['start']['x'], "y": seg['start']['y']},
                    "end": {"x": seg['end']['x'], "y": seg['end']['y']}
                })
            
            building_extremes = find_building_extremes(formatted_walls, pillar_polygons_hatching)
            if building_extremes:
                foundation = create_foundation_polygon(building_extremes)
                if foundation:
                    json_data["foundation"] = foundation
                    print(f"   Foundation created: {foundation['area']:.1f} sq.px, perimeter: {foundation['perimeter']:.1f} px")
                    
                    # Фильтруем junctions, оставляя только те внутри фундамента
                    original_junctions = json_data["junctions"].copy()
                    json_data["junctions"] = filter_junctions_by_foundation(
                        original_junctions,
                        json_data["foundation"]
                    )
                    
                    print(f"   Junctions filtered: {len(original_junctions)} -> {len(json_data['junctions'])}")
                else:
                    print("   Failed to create foundation polygon")
            else:
                print("   No walls or pillars found for foundation creation")
        except Exception as e:
            print(f"   Error creating foundation: {e}")

        # Create building outline from filtered junctions (only those inside foundation)
        print("\n   Creating building outline from filtered junctions...")
        try:
            # Используем отфильтрованные junctions для создания контура здания
            # Передаем также информацию о колоннах и окнах, чтобы избегать их и использовать ориентацию
            building_outline = find_building_outline_from_junctions(
                json_data["junctions"],
                json_data["pillar_polygons"],
                json_data["openings"]
            )
            if building_outline:
                json_data["building_outline"] = building_outline
                print(f"   Building outline created: {building_outline['area']:.1f} sq.px, perimeter: {building_outline['perimeter']:.1f} px")
            else:
                print("   No filtered junctions found for building outline creation")
        except Exception as e:
            print(f"   Error creating building outline: {e}")

        # Update statistics
        json_data["statistics"]["walls"] = len(json_data["walls"])
        json_data["statistics"]["windows"] = len([o for o in json_data["openings"] if o["type"] == "window"])
        json_data["statistics"]["doors"] = len([o for o in json_data["openings"] if o["type"] == "door"])
        json_data["statistics"]["pillars"] = len(json_data["pillars"])
        json_data["statistics"]["pillar_polygons"] = len(json_data["pillar_polygons"])
        json_data["statistics"]["rooms"] = len(json_data["rooms"])
        json_data["statistics"]["junctions"] = len(json_data["junctions"])
        json_data["statistics"]["foundations"] = 1 if json_data["foundation"] else 0
        json_data["statistics"]["building_outlines"] = 1 if json_data["building_outline"] else 0

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"   JSON saved to: {output_path}")
        
        # Create colored SVG visualization
        print("\n   Creating SVG visualization...")
        svg_output_path = output_path.replace('.json', '_colored.svg')
        try:
            svg_w, svg_h = create_colored_svg(
                svg_output_path,
                img_np.shape,
                wall_segments,
                doors,
                windows,
                pillars,
                rooms,
                pillar_polygons_hatching,  # Новый параметр
                json_data["foundation"],   # Добавляем фундамент
                json_data["building_outline"],  # Добавляем контур здания
                scale
            )
            print(f"   SVG saved to: {svg_output_path} ({svg_w}x{svg_h})")
        except Exception as e:
            print(f"   Error creating SVG: {e}")

    except Exception as e:
        print(f"   Error creating JSON: {e}")
        return

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Walls:   {len(json_data['walls'])}")
    print(f"  Windows: {len([o for o in json_data['openings'] if o['type'] == 'window'])}")
    print(f"  Doors:   {len([o for o in json_data['openings'] if o['type'] == 'door'])}")
    print(f"  Pillars: {len(json_data['pillars'])}")
    print(f"  Pillar polygons: {len(json_data['pillar_polygons'])}")
    print(f"  Rooms:   {len(json_data['rooms'])}")
    print(f"\nOutput:")
    print(f"  JSON: {output_path}")
    print(f"  SVG:  {output_path.replace('.json', '_colored.svg')}")

def test_foundation_creation():
    """
    Тестовая функция для проверки создания фундамента
    """
    print("Тестирование создания фундамента...")
    
    # Создаем тестовые данные стен
    test_walls = [
        {"start": {"x": 100, "y": 100}, "end": {"x": 200, "y": 100}},
        {"start": {"x": 200, "y": 100}, "end": {"x": 200, "y": 200}},
        {"start": {"x": 200, "y": 200}, "end": {"x": 100, "y": 200}},
        {"start": {"x": 100, "y": 200}, "end": {"x": 100, "y": 100}}
    ]
    
    # Создаем тестовые данные колонн
    test_pillars = [
        {
            "vertices": [
                {"x": 50, "y": 50},
                {"x": 60, "y": 50},
                {"x": 60, "y": 60},
                {"x": 50, "y": 60}
            ]
        }
    ]
    
    # Тестируем функцию поиска экстремальных координат
    extremes = find_building_extremes(test_walls, test_pillars)
    print(f"Экстремальные координаты: {extremes}")
    
    # Тестируем создание фундамента
    foundation = create_foundation_polygon(extremes)
    print(f"Фундамент: {foundation}")
    
    # Ожидаемый результат:
    # min_x: 50, max_x: 200, min_y: 50, max_y: 200
    
    return foundation

if __name__ == '__main__':
    main()