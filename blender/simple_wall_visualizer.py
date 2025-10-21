#!/usr/bin/env python3
"""
Простой скрипт для визуализации всех сегментов стен из wall_coordinates.json
"""

import json
from PIL import Image, ImageDraw, ImageFont


def visualize_all_segments():
    """Визуализирует все сегменты стен"""
    
    # Загружаем данные
    with open("wall_coordinates.json", 'r') as f:
        data = json.load(f)
    
    # Собираем все сегменты
    all_segments = []
    
    # Сегменты от проемов
    for segment in data["wall_segments_from_openings"]:
        all_segments.append(segment)
    
    # Сегменты от соединений
    for segment in data["wall_segments_from_junctions"]:
        all_segments.append(segment)
    
    print(f"Всего сегментов: {len(all_segments)}")
    
    # Вычисляем размеры изображения
    max_x, max_y = 0, 0
    min_x, min_y = float('inf'), float('inf')
    
    for segment in all_segments:
        bbox = segment["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
        min_x = min(min_x, bbox["x"])
        min_y = min(min_y, bbox["y"])
    
    # Добавляем отступы
    padding = 50
    scale = 1.0
    width = int((max_x - min_x) * scale) + padding * 2
    height = int((max_y - min_y) * scale) + padding * 2
    
    print(f"Размеры изображения: {width}x{height}")
    print(f"Границы: X({min_x}, {max_x}), Y({min_y}, {max_y})")
    
    # Создаем изображение
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Рисуем сегменты
    for i, segment in enumerate(all_segments):
        bbox = segment["bbox"]
        x = int((bbox["x"] - min_x) * scale) + padding
        y = int((bbox["y"] - min_y) * scale) + padding
        w = int(bbox["width"] * scale)
        h = int(bbox["height"] * scale)
        
        # Определяем цвет по типу сегмента
        if segment["segment_id"].startswith("wall_window_"):
            color = 'blue'
        elif segment["segment_id"].startswith("wall_door_"):
            color = 'green'
        elif segment["segment_id"].startswith("wall_junction_"):
            color = 'red'
        else:
            color = 'black'
        
        # Рисуем сегмент
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        
        # Добавляем номер сегмента
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        draw.text((x + w/2 - 10, y + h/2 - 5), str(i+1), fill=color, font=font)
    
    # Добавляем заголовок
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Все сегменты стен", fill='black', font=font)
    
    # Добавляем легенду
    legend_y = 30
    legend_items = [
        ("Синий - сегменты от окон", 'blue'),
        ("Зеленый - сегменты от дверей", 'green'),
        ("Красный - сегменты от соединений", 'red')
    ]
    
    for text, color in legend_items:
        draw.text((10, legend_y), text, fill=color, font=font)
        legend_y += 20
    
    # Сохраняем изображение
    img.save("all_segments.jpg")
    print("Изображение сохранено: all_segments.jpg")
    
    # Выводим информацию о сегментах
    print("\nИнформация о сегментах:")
    for i, segment in enumerate(all_segments):
        bbox = segment["bbox"]
        print(f"{i+1}. {segment['segment_id']}: ({bbox['x']}, {bbox['y']}) {bbox['width']}x{bbox['height']}")


if __name__ == "__main__":
    visualize_all_segments()