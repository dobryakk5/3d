#!/usr/bin/env python3
"""
Тестирование фильтрации шумовых регионов
"""

import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import time

# Импортируем функции из debug_enhanced_hatching.py
from debug_enhanced_hatching import (
    detect_hatching_enhanced_optimized,
    filter_noise_regions_optimized
)

def test_with_filtering():
    """Тестирование с включенной фильтрацией"""
    try:
        print("Начало тестирования с фильтрацией...")
        
        # Загрузка изображения
        image_path = 'plan_floor1.jpg'
        print(f"Загрузка изображения: {image_path}")
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)
        print(f"Изображение загружено, размер: {img_np.shape}")
        
        # Сначала получаем маску без фильтрации
        print("\n=== Получение маски без фильтрации ===")
        wall_mask_no_filter = detect_hatching_enhanced_optimized(
            img_np,
            kernel_sizes=[25],
            density_thresholds=[0.2],
            angles=[45],
            min_neighbors=0  # Без фильтрации
        )
        
        # Теперь тестируем фильтрацию отдельно
        print("\n=== Тестирование фильтрации шумовых регионов ===")
        start_time = time.time()
        
        filtered_mask = filter_noise_regions_optimized(
            wall_mask_no_filter,
            min_area=100,
            max_area=50000,
            min_rectangularity=0.6,
            min_neighbors=1
        )
        
        end_time = time.time()
        print(f"Фильтрация завершена за {end_time - start_time:.2f} секунд")
        
        # Сохранение результатов
        cv2.imwrite('mask_no_filter.png', wall_mask_no_filter)
        cv2.imwrite('mask_with_filter.png', filtered_mask)
        
        # Статистика
        pixels_no_filter = np.sum(wall_mask_no_filter > 0)
        pixels_with_filter = np.sum(filtered_mask > 0)
        
        print(f"\nСтатистика:")
        print(f"Пикселей без фильтрации: {pixels_no_filter}")
        print(f"Пикселей с фильтрацией: {pixels_with_filter}")
        print(f"Разница: {pixels_no_filter - pixels_with_filter}")
        
        return filtered_mask
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_pipeline_with_filtering():
    """Тестирование полного конвейера с фильтрацией"""
    try:
        print("\n=== Тестирование полного конвейера с фильтрацией ===")
        
        # Загрузка изображения
        image_path = 'plan_floor1.jpg'
        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)
        
        # Полный конвейер с фильтрацией
        wall_mask = detect_hatching_enhanced_optimized(
            img_np,
            kernel_sizes=[25],
            density_thresholds=[0.2],
            angles=[45],
            min_neighbors=1  # С фильтрацией
        )
        
        # Сохранение результата
        cv2.imwrite('full_pipeline_with_filter.png', wall_mask)
        
        # Статистика
        hatching_pixels = np.sum(wall_mask > 0)
        print(f"Обнаружено пикселей штриховки: {hatching_pixels}")
        
        return wall_mask
        
    except Exception as e:
        print(f"Ошибка при тестировании полного конвейера: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Тестируем фильтрацию отдельно
    test_with_filtering()
    
    # Тестируем полный конвейер с фильтрацией
    test_full_pipeline_with_filtering()