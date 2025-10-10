#!/usr/bin/env python3
"""
Модуль для обнаружения горизонтальных и вертикальных линий в планах помещений
"""

import cv2
import numpy as np
from scipy import ndimage

def detect_horizontal_lines(gray, min_length=50, max_gap=10, threshold=50):
    """
    Обнаружение горизонтальных линий с использованием преобразования Хафа
    
    Args:
        gray: Изображение в градациях серого
        min_length: Минимальная длина линии
        max_gap: Максимальный разрыв в линии
        threshold: Порог для преобразования Хафа
    
    Returns:
        Бинарная маска с обнаруженными горизонтальными линиями
    """
    # Бинаризация изображения
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Обнаружение горизонтальных линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=threshold,
                           minLineLength=min_length, maxLineGap=max_gap)
    
    # Создаем маску для горизонтальных линий
    mask = np.zeros_like(gray)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Проверяем, что линия достаточно горизонтальная
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Порог горизонтальности
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    
    return mask

def detect_vertical_lines(gray, min_length=50, max_gap=10, threshold=50):
    """
    Обнаружение вертикальных линий с использованием преобразования Хафа
    
    Args:
        gray: Изображение в градациях серого
        min_length: Минимальная длина линии
        max_gap: Максимальный разрыв в линии
        threshold: Порог для преобразования Хафа
    
    Returns:
        Бинарная маска с обнаруженными вертикальными линиями
    """
    # Бинаризация изображения
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Обнаружение вертикальных линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=threshold,
                           minLineLength=min_length, maxLineGap=max_gap)
    
    # Создаем маску для вертикальных линий
    mask = np.zeros_like(gray)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Проверяем, что линия достаточно вертикальная
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 75 < angle < 105:  # Порог вертикальности
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    
    return mask

def detect_lines_morphology(gray, line_type='horizontal', kernel_length=50, kernel_thickness=1):
    """
    Обнаружение линий с использованием морфологических операций
    
    Args:
        gray: Изображение в градациях серого
        line_type: 'horizontal' или 'vertical'
        kernel_length: Длина морфологического ядра
        kernel_thickness: Толщина морфологического ядра
    
    Returns:
        Бинарная маска с обнаруженными линиями
    """
    # Бинаризация изображения
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Создаем морфологическое ядро
    if line_type == 'horizontal':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, kernel_thickness))
    else:  # vertical
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_thickness, kernel_length))
    
    # Применяем морфологические операции
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return lines

def check_hatching_line_intersection(hatching_mask, line_mask, min_overlap_ratio=0.3):
    """
    Проверка пересечения маски штриховки с линиями
    
    Args:
        hatching_mask: Маска штриховки
        line_mask: Маска линий
        min_overlap_ratio: Минимальное отношение пересечения к площади линии
    
    Returns:
        Список компонентов линий, которые пересекаются с штриховкой
    """
    # Находим компоненты линий
    labeled_lines, num_lines = ndimage.label(line_mask)
    
    intersecting_lines = []
    
    for i in range(1, num_lines + 1):
        line_component = (labeled_lines == i)
        line_area = np.sum(line_component)
        
        if line_area == 0:
            continue
        
        # Проверяем пересечение с штриховкой
        intersection = np.logical_and(line_component, hatching_mask > 0)
        intersection_area = np.sum(intersection)
        
        # Вычисляем отношение пересечения к площади линии
        overlap_ratio = intersection_area / line_area
        
        if overlap_ratio >= min_overlap_ratio:
            intersecting_lines.append((i, line_component, overlap_ratio))
    
    return intersecting_lines

def enhance_lines_with_hatching(hatching_mask, gray, 
                               min_line_length=50, 
                               min_overlap_ratio=0.3,
                               line_thickness=2):
    """
    Улучшение маски штриховки путем добавления горизонтальных и вертикальных линий,
    которые пересекаются с областями штриховки
    
    Args:
        hatching_mask: Исходная маска штриховки
        gray: Изображение в градациях серого
        min_line_length: Минимальная длина линии для добавления
        min_overlap_ratio: Минимальное отношение пересечения
        line_thickness: Толщина добавляемых линий
    
    Returns:
        Улучшенная маска штриховки с добавленными линиями
    """
    # Обнаружение горизонтальных линий
    horizontal_lines = detect_lines_morphology(gray, 'horizontal', 
                                             kernel_length=min_line_length, 
                                             kernel_thickness=line_thickness)
    
    # Обнаружение вертикальных линий
    vertical_lines = detect_lines_morphology(gray, 'vertical', 
                                           kernel_length=min_line_length, 
                                           kernel_thickness=line_thickness)
    
    # Проверка пересечений с горизонтальными линиями
    horizontal_intersections = check_hatching_line_intersection(
        hatching_mask, horizontal_lines, min_overlap_ratio)
    
    # Проверка пересечений с вертикальными линиями
    vertical_intersections = check_hatching_line_intersection(
        hatching_mask, vertical_lines, min_overlap_ratio)
    
    # Создаем улучшенную маску
    enhanced_mask = hatching_mask.copy()
    
    # Добавляем пересекающиеся горизонтальные линии
    for _, line_component, _ in horizontal_intersections:
        enhanced_mask[line_component] = 255
    
    # Добавляем пересекающиеся вертикальные линии
    for _, line_component, _ in vertical_intersections:
        enhanced_mask[line_component] = 255
    
    return enhanced_mask, len(horizontal_intersections), len(vertical_intersections)