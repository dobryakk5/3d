"""
Упрощенный детектор внешних стен на основе анализа границ здания
"""

import json
from typing import List, Dict, Set

def detect_external_walls(segments: List[Dict], pillars: List[Dict] = None) -> Set[str]:
    """
    Определяет внешние стены на основе анализа границ здания
    
    Args:
        segments: список сегментов стен
        pillars: список колонн (опционально)
    
    Returns:
        Set[str]: множество ID внешних сегментов
    """
    # Собираем все координаты для определения границ здания
    all_x_coords = []
    all_y_coords = []
    
    # Добавляем координаты сегментов стен
    for segment in segments:
        bbox = segment["bbox"]
        all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
        all_y_coords.extend([bbox["y"], bbox["y"] + bbox["height"]])
    
    # Добавляем координаты колонн
    if pillars:
        for pillar in pillars:
            bbox = pillar["bbox"]
            all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
            all_y_coords.extend([bbox["y"], bbox["y"] + bbox["height"]])
    
    if not all_x_coords or not all_y_coords:
        return set()
    
    # Определяем границы здания
    min_x, max_x = min(all_x_coords), max(all_x_coords)
    min_y, max_y = min(all_y_coords), max(all_y_coords)
    
    # Добавляем небольшой допуск
    tolerance = 5
    
    # Определяем внешние сегменты
    external_segments = set()
    
    for segment in segments:
        bbox = segment["bbox"]
        wall_left = bbox["x"]
        wall_right = bbox["x"] + bbox["width"]
        wall_top = bbox["y"]
        wall_bottom = bbox["y"] + bbox["height"]
        
        # Проверяем, находится ли стена на границе
        is_boundary = (
            abs(wall_left - min_x) <= tolerance or
            abs(wall_right - max_x) <= tolerance or
            abs(wall_top - min_y) <= tolerance or
            abs(wall_bottom - max_y) <= tolerance
        )
        
        if is_boundary:
            external_segments.add(segment["segment_id"])
    
    return external_segments

def detect_external_walls_with_openings(wall_segments: List[Dict], openings: List[Dict]) -> Set[str]:
    """
    Определяет внешние стены с учетом наличия окон
    
    Args:
        wall_segments: список сегментов стен
        openings: список проемов
    
    Returns:
        Set[str]: множество ID внешних сегментов
    """
    external_segments = detect_external_walls(wall_segments)
    
    # Дополнительная проверка: сегменты с окнами всегда внешние
    opening_ids = {opening["id"] for opening in openings if opening["type"] == "window"}
    
    for segment in wall_segments:
        if "opening_id" in segment and segment["opening_id"] in opening_ids:
            external_segments.add(segment["segment_id"])
    
    return external_segments

def test_external_wall_detection():
    """Тестирует определение внешних стен"""
    print("Тестирование упрощенного детектора внешних стен")
    print("=" * 60)
    
    # Загружаем данные
    with open("wall_coordinates.json", 'r') as f:
        data = json.load(f)
    
    # Объединяем все сегменты
    all_segments = []
    
    # Сегменты от проемов
    for segment in data["wall_segments_from_openings"]:
        all_segments.append(segment)
    
    # Сегменты от соединений
    for segment in data["wall_segments_from_junctions"]:
        all_segments.append(segment)
    
    pillars = data.get("pillar_squares", [])
    
    print(f"Всего сегментов: {len(all_segments)}")
    print(f"  Сегментов от проемов: {len(data['wall_segments_from_openings'])}")
    print(f"  Сегментов от соединений: {len(data['wall_segments_from_junctions'])}")
    print(f"  Колонн: {len(pillars)}")
    
    # Определяем внешние стены
    external_segments = detect_external_walls(all_segments, pillars)
    
    # Дополнительная проверка с окнами
    external_segments_with_openings = detect_external_walls_with_openings(
        all_segments, data["openings"]
    )
    
    print(f"\nНайдено внешних сегментов (границы): {len(external_segments)}")
    print(f"Найдено внешних сегментов (с окнами): {len(external_segments_with_openings)}")
    
    # Сравниваем результаты
    only_boundaries = external_segments - external_segments_with_openings
    only_openings = external_segments_with_openings - external_segments
    
    if only_boundaries:
        print(f"\nСегменты только по границам: {len(only_boundaries)}")
        for seg_id in sorted(only_boundaries):
            print(f"  {seg_id}")
    
    if only_openings:
        print(f"\nСегменты только с окнами: {len(only_openings)}")
        for seg_id in sorted(only_openings):
            print(f"  {seg_id}")
    
    # Анализируем сегменты рядом с колоннами
    print("\nАнализ сегментов рядом с колоннами:")
    print("-" * 50)
    
    for pillar in pillars:
        pillar_bbox = pillar["bbox"]
        pillar_top = pillar_bbox["y"] + pillar_bbox["height"]
        
        print(f"\nКолонна {pillar['id']}:")
        print(f"  Границы: x={pillar_bbox['x']}-{pillar_bbox['x'] + pillar_bbox['width']}, "
              f"y={pillar_bbox['y']}-{pillar_top}")
        print(f"  Верхняя граница: {pillar_top}")
        
        # Ищем сегменты рядом с колонной
        nearby_segments = []
        for segment in all_segments:
            seg_bbox = segment["bbox"]
            seg_top = seg_bbox["y"] + seg_bbox["height"]
            
            # Проверяем, находится ли сегмент рядом с колонной
            if (abs(seg_top - pillar_top) < 50 or  # Близко по Y
                abs(seg_bbox["y"] - pillar_bbox["y"]) < 50 or  # Близко по Y снизу
                abs(seg_bbox["x"] - pillar_bbox["x"]) < 50 or  # Близко по X
                abs(seg_bbox["x"] + seg_bbox["width"] - (pillar_bbox["x"] + pillar_bbox["width"])) < 50):  # Близко по X справа
                
                is_external = segment["segment_id"] in external_segments_with_openings
                is_external_boundary = segment["segment_id"] in external_segments
                has_window = "opening_id" in segment and any(
                    o["id"] == segment["opening_id"] and o["type"] == "window" 
                    for o in data["openings"]
                )
                
                nearby_segments.append({
                    "segment_id": segment["segment_id"],
                    "is_external": is_external,
                    "is_external_boundary": is_external_boundary,
                    "has_window": has_window,
                    "bbox": seg_bbox
                })
        
        if nearby_segments:
            print(f"  Найдено {len(nearby_segments)} сегментов рядом с колонной:")
            for seg in nearby_segments:
                status = "Внешняя" if seg["is_external"] else "Внутренняя"
                reason = []
                if seg["has_window"]:
                    reason.append("окно")
                if seg["is_external_boundary"]:
                    reason.append("граница")
                
                reason_str = f" ({', '.join(reason)})" if reason else ""
                print(f"    {seg['segment_id']}: {status}{reason_str}")
        else:
            print(f"  Сегменты рядом с колонной не найдены")
    
    # Статистика
    print("\nСтатистика:")
    print(f"  Всего сегментов: {len(all_segments)}")
    print(f"  Внешних сегментов: {len(external_segments_with_openings)}")
    print(f"  Внутренних сегментов: {len(all_segments) - len(external_segments_with_openings)}")
    print(f"  Процент внешних: {len(external_segments_with_openings) / len(all_segments) * 100:.1f}%")

if __name__ == "__main__":
    test_external_wall_detection()