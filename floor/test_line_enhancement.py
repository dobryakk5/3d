#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности добавления линий к маске штриховки
"""

import numpy as np
from PIL import Image
import cv2
from enhanced_hatching_fixed import detect_hatching_enhanced_fixed

def test_line_enhancement():
    """Тестирование добавления горизонтальных и вертикальных линий"""
    
    # Загрузка изображения
    image_path = 'plan_floor1.jpg'
    try:
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)
        print(f"Изображение загружено: {image_path}")
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return
    
    # Тест 1: Только обнаружение штриховки (без линий)
    print("\n=== Тест 1: Только обнаружение штриховки ===")
    wall_mask_no_lines = detect_hatching_enhanced_fixed(
        img_np,
        add_lines=False,
        min_area=2
    )
    
    hatching_pixels_no_lines = np.sum(wall_mask_no_lines > 0)
    print(f"Обнаружено пикселей штриховки (без линий): {hatching_pixels_no_lines}")
    cv2.imwrite('test_1_no_lines.png', wall_mask_no_lines)
    print("Результат сохранен в: test_1_no_lines.png")
    
    # Тест 2: Обнаружение штриховки + добавление линий
    print("\n=== Тест 2: Обнаружение штриховки + добавление линий ===")
    wall_mask_with_lines = detect_hatching_enhanced_fixed(
        img_np,
        add_lines=True,
        min_line_length=30,
        min_line_overlap_ratio=0.2,
        min_area=2
    )
    
    hatching_pixels_with_lines = np.sum(wall_mask_with_lines > 0)
    print(f"Обнаружено пикселей штриховки (с линиями): {hatching_pixels_with_lines}")
    cv2.imwrite('test_2_with_lines.png', wall_mask_with_lines)
    print("Результат сохранен в: test_2_with_lines.png")
    
    # Тест 3: Более строгие параметры для линий
    print("\n=== Тест 3: Более строгие параметры для линий ===")
    wall_mask_strict = detect_hatching_enhanced_fixed(
        img_np,
        add_lines=True,
        min_line_length=50,
        min_line_overlap_ratio=0.4,
        min_area=2
    )
    
    hatching_pixels_strict = np.sum(wall_mask_strict > 0)
    print(f"Обнаружено пикселей штриховки (строгие параметры): {hatching_pixels_strict}")
    cv2.imwrite('test_3_strict_lines.png', wall_mask_strict)
    print("Результат сохранен в: test_3_strict_lines.png")
    
    # Сравнение результатов
    print("\n=== Сравнение результатов ===")
    print(f"Только штриховка: {hatching_pixels_no_lines} пикселей")
    print(f"Штриховка + линии: {hatching_pixels_with_lines} пикселей")
    print(f"Строгие параметры: {hatching_pixels_strict} пикселей")
    
    # Вычисляем разницу
    diff_pixels = hatching_pixels_with_lines - hatching_pixels_no_lines
    diff_strict = hatching_pixels_strict - hatching_pixels_no_lines
    
    print(f"Добавлено пикселей (базовые параметры): {diff_pixels}")
    print(f"Добавлено пикселей (строгие параметры): {diff_strict}")
    
    # Визуализация разницы
    if diff_pixels > 0:
        diff_mask = cv2.bitwise_xor(wall_mask_with_lines, wall_mask_no_lines)
        cv2.imwrite('test_diff_basic.png', diff_mask)
        print("Разница (базовые параметры) сохранена в: test_diff_basic.png")
    
    if diff_strict > 0:
        diff_mask_strict = cv2.bitwise_xor(wall_mask_strict, wall_mask_no_lines)
        cv2.imwrite('test_diff_strict.png', diff_mask_strict)
        print("Разница (строгие параметры) сохранена в: test_diff_strict.png")

def test_different_line_lengths():
    """Тестирование разных длин линий"""
    
    # Загрузка изображения
    image_path = 'plan_floor1.jpg'
    try:
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)
        print(f"Изображение загружено: {image_path}")
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return
    
    print("\n=== Тестирование разных длин линий ===")
    
    for line_length in [20, 30, 50, 80]:
        print(f"\n--- Длина линии: {line_length} ---")
        wall_mask = detect_hatching_enhanced_fixed(
            img_np,
            add_lines=True,
            min_line_length=line_length,
            min_line_overlap_ratio=0.3,
            min_area=2
        )
        
        hatching_pixels = np.sum(wall_mask > 0)
        print(f"Обнаружено пикселей штриховки: {hatching_pixels}")
        cv2.imwrite(f'test_line_length_{line_length}.png', wall_mask)
        print(f"Результат сохранен в: test_line_length_{line_length}.png")

if __name__ == '__main__':
    test_line_enhancement()
    test_different_line_lengths()