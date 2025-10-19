#!/usr/bin/env python3
"""
Исправленная версия enhanced_hatching.py с оптимизацией и исправленными ошибками
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion
from PIL import Image, ImageEnhance
import time
# from line_detection import enhance_lines_with_hatching  # Закомментировано, так как модуль отсутствует

# Заглушка для отсутствующей функции
def enhance_lines_with_hatching(wall_mask, gray, min_line_length, min_line_overlap_ratio):
    """
    Заглушка для отсутствующей функции enhance_lines_with_hatching
    Возвращает исходную маску без изменений
    """
    print("   Внимание: используется заглушка для enhance_lines_with_hatching")
    return wall_mask, 0, 0

# =============================================================================
# ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ ЖЕСТКОСТИ МАСКИ ШТРИХОВКИ
# =============================================================================
# По умолчанию все фильтры отключены (None), чтобы включить фильтр, установите значение.
# Можно включать фильтры по отдельности или в комбинации:
#
# - DENSITY_THRESHOLDS: Пороги плотности (чем выше, тем жестче)
#   * Для жесткой маски: [0.20, 0.30, 0.40]
#   * Для мягкой маски: [0.10, 0.15, 0.20]
#
# - MIN_AREA: Минимальная площадь области (None = отключено)
#   * Для жесткой маски: 200-300
#   * Для мягкой маски: 50-100
#
# - MAX_AREA: Максимальная площадь области (None = отключено)
#   * Для жесткой маски: 30000-40000
#   * Для мягкой маски: 60000-80000
#
# - MIN_RECTANGULARITY: Минимальное соотношение сторон для прямоугольности (None = отключено)
#   * Для жесткой маски: 0.8
#   * Для мягкой маски: 0.5
#
# - MIN_NEIGHBORS: Минимальное количество соседей для связности (None = отключено)
#   * 0 = отключено, 1 = требуется хотя бы один сосед
#
# - ADD_LINES: Добавлять горизонтальные и вертикальные линии (True/False)
#   * True - добавлять линии, которые пересекаются с областями штриховки
#   * False - только обнаружение штриховки
#
# - MIN_LINE_LENGTH: Минимальная длина линии для добавления
#   * Рекомендуемые значения: 30-100
#
# - MIN_LINE_OVERLAP_RATIO: Минимальное отношение пересечения линии с штриховкой
#   * Рекомендуемые значения: 0.2-0.5
# =============================================================================

KERNEL_SIZES = [35]  # Размеры ядер для разной толщины линий
DENSITY_THRESHOLDS = [0.32]  # Пороги плотности (чем выше, тем жестче)
ANGLES = [45, 135]  # Углы обнаружения линий
MIN_AREA = 1200  # Минимальная площадь области (None = отключено)
MAX_AREA = None  # Максимальная площадь области (None = отключено)
MIN_RECTANGULARITY = None  # Минимальное соотношение сторон для прямоугольности (None = отключено)
MIN_NEIGHBORS = None  # Минимальное количество соседей для связности (None = отключено)
ADD_LINES = False  # Добавлять горизонтальные и вертикальные линии (True/False)
MIN_LINE_LENGTH = 10  # Минимальная длина линии для добавления
MIN_LINE_OVERLAP_RATIO = 0.3  # Минимальное отношение пересечения линии с штриховкой
OUTPUT_FILENAME = 'enhanced_hatching_strict_mask.png'  # Имя выходного файла

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

    # Собираем информацию о компонентах с их bounding boxes
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

        # Сохраняем компонент с его центром масс для проверки изолированности
        y_coords, x_coords = np.where(component_mask > 0)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        component_info.append((i, component_mask, centroid_x, centroid_y, area))

    # Проверяем связность с соседями (только если явно указано)
    if min_neighbors > 0 and len(component_info) > 1:
        final_components = []

        for i, info in enumerate(component_info):
            comp_id, component_mask = info[0], info[1]
            # Упрощенная проверка связности
            is_connected = False
            for j, other_info in enumerate(component_info):
                if i == j:
                    continue

                other_id, other_mask = other_info[0], other_info[1]
                # Проверяем расстояние между компонентами
                if check_components_proximity(component_mask, other_mask, max_distance=30):
                    is_connected = True
                    break

            # Если компонент связан или требование связности не строгое
            if is_connected or min_neighbors <= 1:
                final_components.append((comp_id, component_mask))

        component_info = [(info[0], info[1]) for info in final_components]
    else:
        # Преобразуем обратно к формату (id, mask)
        component_info = [(info[0], info[1]) for info in component_info]
    
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

# Тестовая функция для демонстрации
def test_enhanced_hatching():
    """Тестирование исправленной функции обнаружения штриховки"""
    try:
        # Загрузка изображения
        image_path = 'plan_floor1.jpg'
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)
        
        print("Тестирование исправленной функции обнаружения штриховки...")
        start_time = time.time()
        
        # Применение функции с параметрами из глобальных переменных
        wall_mask = detect_hatching_enhanced_fixed(img_np)
        
        end_time = time.time()
        
        # Сохранение результата
        cv2.imwrite(OUTPUT_FILENAME, wall_mask)
        
        # Статистика
        hatching_pixels = np.sum(wall_mask > 0)
        print(f"Обнаружено пикселей штриховки: {hatching_pixels}")
        print(f"Время выполнения: {end_time - start_time:.2f} секунд")
        print(f"Результат сохранен в: {OUTPUT_FILENAME}")
        
        return wall_mask
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return None

# Примеры использования с разными фильтрами
def test_with_min_area_filter():
    """Тестирование с фильтром по минимальной площади"""
    print("\n=== Тест с фильтром по минимальной площади ===")
    image_path = 'plan_floor1.jpg'
    image = Image.open(image_path).convert('RGB')
    img_np = np.array(image)
    
    # Включаем только фильтр по минимальной площади
    wall_mask = detect_hatching_enhanced_fixed(
        img_np,
        min_area=1500  # Только этот фильтр включен
    )
    
    hatching_pixels = np.sum(wall_mask > 0)
    print(f"Обнаружено пикселей штриховки: {hatching_pixels}")
    cv2.imwrite('hatching_min_area_filter.png', wall_mask)
    print("Результат сохранен в: hatching_min_area_filter.png")
    return wall_mask

def test_with_rectangularity_filter():
    """Тестирование с фильтром по прямоугольности"""
    print("\n=== Тест с фильтром по прямоугольности ===")
    image_path = 'plan_floor1.jpg'
    image = Image.open(image_path).convert('RGB')
    img_np = np.array(image)
    
    # Включаем только фильтр по прямоугольности
    wall_mask = detect_hatching_enhanced_fixed(
        img_np,
        min_rectangularity=0.6  # Только этот фильтр включен
    )
    
    hatching_pixels = np.sum(wall_mask > 0)
    print(f"Обнаружено пикселей штриховки: {hatching_pixels}")
    cv2.imwrite('hatching_rectangularity_filter.png', wall_mask)
    print("Результат сохранен в: hatching_rectangularity_filter.png")
    return wall_mask

def test_with_multiple_filters():
    """Тестирование с несколькими фильтрами"""
    print("\n=== Тест с несколькими фильтрами ===")
    image_path = 'plan_floor1.jpg'
    image = Image.open(image_path).convert('RGB')
    img_np = np.array(image)
    
    # Включаем несколько фильтров
    wall_mask = detect_hatching_enhanced_fixed(
        img_np,
        min_area=1500,
        max_area=50000,
        min_rectangularity=0.6
        # min_neighbors не указан, поэтому отключен
    )
    
    hatching_pixels = np.sum(wall_mask > 0)
    print(f"Обнаружено пикселей штриховки: {hatching_pixels}")
    cv2.imwrite('hatching_multiple_filters.png', wall_mask)
    print("Результат сохранен в: hatching_multiple_filters.png")
    return wall_mask

if __name__ == '__main__':
    # Запускаем основной тест без фильтров
    test_enhanced_hatching()
    
    # Раскомментируйте нужные тесты:
    # test_with_min_area_filter()        # Только фильтр по минимальной площади
    # test_with_rectangularity_filter()  # Только фильтр по прямоугольности
    # test_with_multiple_filters()       # Несколько фильтров