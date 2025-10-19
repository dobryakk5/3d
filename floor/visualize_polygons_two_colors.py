#!/usr/bin/env python3
"""
Визуализация полигонов стен, окон и дверей с разными цветами:
- Красный контур для стен (определенный по растру изображения)
- Коричневый для колонн с полной заливкой
- Синий для окон
- Зеленый для дверей
"""

import json
import cv2
import numpy as np
from PIL import Image

def detect_contours_from_image(image):
    """Обнаруживает контуры на изображении с помощью Canny"""
    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детектор границ Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Морфологические операции для улучшения контуров
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

def visualize_polygons_two_colors():
    """Создает визуализацию полигонов стен, окон и дверей с разными цветами"""

    # Загружаем JSON с полигонами
    with open('plan_floor1_objects.json', 'r') as f:
        data = json.load(f)

    # Загружаем растровое изображение
    img = cv2.imread('plan_floor1.jpg')
    if img is None:
        print("Ошибка: не удалось загрузить plan_floor1.jpg")
        return

    orig_height, orig_width = img.shape[:2]
    print(f"Размер растра: {orig_width}x{orig_height}")

    # Получаем масштаб из JSON
    scale_factor = data['metadata']['scale_factor']
    inverse_scale = 1.0 / scale_factor
    print(f"Scale factor: {scale_factor}")
    print(f"Inverse scale: {inverse_scale}")

    # Создаем копию изображения для отрисовки
    overlay = img.copy()

    # Получаем полигоны
    wall_polygons = data.get('wall_polygons', [])
    pillar_polygons = data.get('pillar_polygons', [])  # Новое поле!
    openings = data.get('openings', [])
    pillars = data.get('pillars', [])
    
    print(f"\nНайдено полигонов стен: {len(wall_polygons)}")
    print(f"Найдено полигонов колонн: {len(pillar_polygons)}")  # Новый вывод
    print(f"Найдено окон и дверей: {len(openings)}")
    print(f"Найдено колонн (bbox): {len(pillars)}")

    # Определяем цвета (BGR формат для OpenCV)
    main_wall_color = (0, 0, 255)      # Красный для стен
    pillar_color = (42, 42, 165)       # Коричневый для колонн
    window_color = (255, 0, 0)         # Синий для окон
    door_color = (0, 255, 0)           # Зеленый для дверей

    # Обнаруживаем контуры на исходном изображении
    print("\nОбнаружение контуров на изображении...")
    contours_edges = detect_contours_from_image(img)
    
    # Накладываем контуры на изображение
    contour_overlay = overlay.copy()
    # Преобразуем контуры в 3-канальное изображение для отображения красным цветом
    contours_colored = np.zeros_like(contour_overlay)
    contours_colored[contours_edges > 0] = main_wall_color
    
    # Комбинируем контуры с основным изображением
    mask = contours_edges > 0
    contour_overlay[mask] = main_wall_color

    # Отрисовываем каждый полигон
    for idx, polygon in enumerate(wall_polygons):
        vertices = polygon['vertices']
        polygon_id = polygon['id']
        area = polygon['area']
        num_vertices = polygon['num_vertices']

        print(f"\n{polygon_id}:")
        print(f"  Вершин: {num_vertices}")
        print(f"  Площадь: {area:.1f} px²")

        # Масштабируем координаты обратно к размеру растра
        scaled_vertices = []
        for v in vertices:
            x_scaled = int(v['x'] * inverse_scale)
            y_scaled = int(v['y'] * inverse_scale)
            scaled_vertices.append([x_scaled, y_scaled])

        # Конвертируем в numpy array
        pts = np.array(scaled_vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Все полигоны здесь - стены контура
        color = main_wall_color
        polygon_type = "стена контура"

        print(f"  Тип: {polygon_type}")

        # Все стены отображаем только контуром, без заливки (убрали дифференцированный подход)
        cv2.polylines(overlay, [pts], True, color, 3, cv2.LINE_AA)

        # Добавляем номер полигона в центре полигона
        M = cv2.moments(pts)
        if M['m00'] != 0:
            label_x = int(M['m10'] / M['m00'])
            label_y = int(M['m01'] / M['m00'])

            cv2.putText(overlay, f"P{idx+1}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(overlay, f"P{idx+1}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    # НОВАЯ СЕКЦИЯ: Отрисовка полигонов колонн
    for idx, polygon in enumerate(pillar_polygons):
        vertices = polygon['vertices']
        polygon_id = polygon['id']
        area = polygon['area']
        num_vertices = polygon['num_vertices']

        print(f"\n{polygon_id}:")
        print(f"  Вершин: {num_vertices}")
        print(f"  Площадь: {area:.1f} px²")

        # Масштабируем координаты обратно к размеру растра
        scaled_vertices = []
        for v in vertices:
            x_scaled = int(v['x'] * inverse_scale)
            y_scaled = int(v['y'] * inverse_scale)
            scaled_vertices.append([x_scaled, y_scaled])

        # Конвертируем в numpy array
        pts = np.array(scaled_vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Полигоны колонн
        color = pillar_color
        polygon_type = "колонна"

        print(f"  Тип: {polygon_type}")

        # Отрисовываем заполненный полигон
        cv2.fillPoly(overlay, [pts], color)
        # Отрисовываем контур полигона
        cv2.polylines(overlay, [pts], True, color, 3, cv2.LINE_AA)

        # Добавляем метку колонны
        M = cv2.moments(pts)
        if M['m00'] != 0:
            label_x = int(M['m10'] / M['m00'])
            label_y = int(M['m01'] / M['m00'])

            cv2.putText(overlay, f"P{idx+1}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            cv2.putText(overlay, f"P{idx+1}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Отрисовываем окна и двери
    print(f"\nОтрисовка окон и дверей:")
    for opening in openings:
        opening_id = opening['id']
        opening_type = opening['type']
        bbox = opening['bbox']
        
        # Масштабируем координаты bbox
        x = int(bbox['x'] * inverse_scale)
        y = int(bbox['y'] * inverse_scale)
        width = int(bbox['width'] * inverse_scale)
        height = int(bbox['height'] * inverse_scale)
        
        # Выбираем цвет в зависимости от типа
        if opening_type == 'window':
            color = window_color
            print(f"  {opening_id}: окно ({x}, {y}, {width}x{height})")
        elif opening_type == 'door':
            color = door_color
            print(f"  {opening_id}: дверь ({x}, {y}, {width}x{height})")
        else:
            continue
        
        # Рисуем прямоугольник (только контур)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 3)
        
        # Добавляем метку
        label = opening_type.upper()[0] + opening_id.split('_')[1]
        cv2.putText(overlay, label, (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, label, (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Сохраняем результат
    output_path = 'wall_polygons_two_colors.png'
    cv2.imwrite(output_path, overlay)
    print(f"\nСохранено: {output_path}")
    print(f"Размер: {overlay.shape[1]}x{overlay.shape[0]}")

    # Создаем уменьшенную версию для предварительного просмотра
    scale_preview = 0.3
    preview_width = int(overlay.shape[1] * scale_preview)
    preview_height = int(overlay.shape[0] * scale_preview)
    preview = cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

    output_path_preview = 'wall_polygons_two_colors_preview.png'
    cv2.imwrite(output_path_preview, preview)
    print(f"Сохранено (preview): {output_path_preview} ({preview_width}x{preview_height})")

    # Обновляем вывод статистики
    pillar_polygon_count = len(pillar_polygons)  # Новая переменная
    main_wall_count = len(wall_polygons)
    window_count = sum(1 for o in openings if o['type'] == 'window')
    door_count = sum(1 for o in openings if o['type'] == 'door')
    
    print(f"\nСтатистика:")
    print(f"  Стены (контур по растру): {main_wall_count} (красный контур)")
    print(f"  Полигоны колонн: {pillar_polygon_count} (коричневый цвет)")
    print(f"  Колонны (bbox): {len(pillars)} (из старого формата)")
    print(f"  Окна: {window_count} (синий контур)")
    print(f"  Двери: {door_count} (зеленый контур)")

if __name__ == '__main__':
    visualize_polygons_two_colors()