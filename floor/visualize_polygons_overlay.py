#!/usr/bin/env python3
"""
Визуализация полигонов стен с наложением на растровое изображение
"""

import json
import cv2
import numpy as np
from PIL import Image

def visualize_polygons_overlay():
    """Создает визуализацию полигонов стен поверх растрового изображения"""

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
    print(f"\nНайдено полигонов: {len(wall_polygons)}")

    # Цвета для разных полигонов (BGR формат для OpenCV)
    colors = [
        (0, 0, 255),      # Красный
        (0, 255, 255),    # Желтый
        (255, 0, 255),    # Пурпурный
        (0, 255, 0),      # Зеленый
        (255, 0, 0),      # Синий
    ]

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

        # Выбираем цвет
        color = colors[idx % len(colors)]

        # Для большого полигона (P10) - только контур, без заливки
        # Для маленьких - заливка с прозрачностью
        if area > 50000:  # Большой полигон (P10)
            # Только контур, без заливки
            cv2.polylines(overlay, [pts], True, color, 5, cv2.LINE_AA)
        else:
            # Отрисовываем заполненный полигон с прозрачностью
            temp_overlay = overlay.copy()
            cv2.fillPoly(temp_overlay, [pts], color)
            cv2.addWeighted(temp_overlay, 0.3, overlay, 0.7, 0, overlay)

            # Отрисовываем контур полигона
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

    # Сохраняем результат
    output_path = 'wall_polygons_overlay.png'
    cv2.imwrite(output_path, overlay)
    print(f"\nСохранено: {output_path}")
    print(f"Размер: {overlay.shape[1]}x{overlay.shape[0]}")

    # Также создаем версию только с полигонами (без растра)
    polygons_only = np.ones_like(img) * 255  # Белый фон

    for idx, polygon in enumerate(wall_polygons):
        vertices = polygon['vertices']
        area = polygon['area']

        # Масштабируем координаты
        scaled_vertices = []
        for v in vertices:
            x_scaled = int(v['x'] * inverse_scale)
            y_scaled = int(v['y'] * inverse_scale)
            scaled_vertices.append([x_scaled, y_scaled])

        pts = np.array(scaled_vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))

        color = colors[idx % len(colors)]

        # Для большого полигона (P10) - только контур, без заливки
        if area > 50000:
            # Только контур
            cv2.polylines(polygons_only, [pts], True, color, 5, cv2.LINE_AA)
        else:
            # Заливка
            cv2.fillPoly(polygons_only, [pts], color)

            # Контур
            cv2.polylines(polygons_only, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)

        # Номер в центре полигона
        M = cv2.moments(pts)
        if M['m00'] != 0:
            label_x = int(M['m10'] / M['m00'])
            label_y = int(M['m01'] / M['m00'])

            cv2.putText(polygons_only, f"P{idx+1}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(polygons_only, f"P{idx+1}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    output_path_clean = 'wall_polygons_clean.png'
    cv2.imwrite(output_path_clean, polygons_only)
    print(f"Сохранено: {output_path_clean}")

    # Создаем уменьшенную версию для предварительного просмотра
    scale_preview = 0.3
    preview_width = int(overlay.shape[1] * scale_preview)
    preview_height = int(overlay.shape[0] * scale_preview)
    preview = cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

    output_path_preview = 'wall_polygons_overlay_preview.png'
    cv2.imwrite(output_path_preview, preview)
    print(f"Сохранено (preview): {output_path_preview} ({preview_width}x{preview_height})")

if __name__ == '__main__':
    visualize_polygons_overlay()
