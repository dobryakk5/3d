import json
import os

def invert_json_coordinates(json_path):
    """
    Инвертирует X-координаты в JSON файле, чтобы избежать инверсии в коде
    
    Args:
        json_path: путь к JSON файлу
    
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    try:
        # Загружаем JSON файл
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Находим максимальную X-координату для определения центра инверсии
        all_x_coords = []
        
        # Собираем X-координаты из junctions
        for junction in data["junctions"]:
            all_x_coords.append(junction["x"])
        
        # Собираем X-координаты из сегментов стен
        for segment in data["wall_segments_from_openings"]:
            bbox = segment["bbox"]
            all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
        
        for segment in data["wall_segments_from_junctions"]:
            bbox = segment["bbox"]
            all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
        
        # Собираем X-координаты из проемов
        for opening in data["openings"]:
            bbox = opening["bbox"]
            all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
        
        # Собираем X-координаты из колонн
        if "pillar_squares" in data:
            for pillar in data["pillar_squares"]:
                bbox = pillar["bbox"]
                all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
        
        # Собираем X-координаты из building_outline
        if "building_outline" in data and "vertices" in data["building_outline"]:
            for vertex in data["building_outline"]["vertices"]:
                all_x_coords.append(vertex["x"])
        
        # Собираем X-координаты из foundation
        if "foundation" in data and "vertices" in data["foundation"]:
            for vertex in data["foundation"]["vertices"]:
                all_x_coords.append(vertex["x"])
        
        if all_x_coords:
            max_x = max(all_x_coords)
            center_x = max_x / 2
            print(f"Центр инверсии X: {center_x}")
            
            # Инвертируем X-координаты в junctions
            for junction in data["junctions"]:
                old_x = junction["x"]
                junction["x"] = 2 * center_x - old_x
            
            # Инвертируем X-координаты в сегментах стен от проемов
            for segment in data["wall_segments_from_openings"]:
                bbox = segment["bbox"]
                old_x = bbox["x"]
                bbox["x"] = 2 * center_x - old_x - bbox["width"]
            
            # Инвертируем X-координаты в сегментах стен от соединений
            for segment in data["wall_segments_from_junctions"]:
                bbox = segment["bbox"]
                old_x = bbox["x"]
                bbox["x"] = 2 * center_x - old_x - bbox["width"]
            
            # Инвертируем X-координаты в проемах
            for opening in data["openings"]:
                bbox = opening["bbox"]
                old_x = bbox["x"]
                bbox["x"] = 2 * center_x - old_x - bbox["width"]
            
            # Инвертируем X-координаты в колоннах
            if "pillar_squares" in data:
                for pillar in data["pillar_squares"]:
                    bbox = pillar["bbox"]
                    old_x = bbox["x"]
                    bbox["x"] = 2 * center_x - old_x - bbox["width"]
            
            # Инвертируем X-координаты в building_outline
            if "building_outline" in data and "vertices" in data["building_outline"]:
                for vertex in data["building_outline"]["vertices"]:
                    old_x = vertex["x"]
                    vertex["x"] = 2 * center_x - old_x
            
            # Инвертируем X-координаты в foundation
            if "foundation" in data and "vertices" in data["foundation"]:
                for vertex in data["foundation"]["vertices"]:
                    old_x = vertex["x"]
                    vertex["x"] = 2 * center_x - old_x
            
            print(f"X-координаты инвертированы относительно центра {center_x}")
        
        # Создаем резервную копию исходного файла
        backup_path = json_path + ".backup_before_invert_fixed"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Создана резервная копия: {backup_path}")
        
        # Сохраняем обновленный JSON файл
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON файл с инвертированными координатами сохранен: {json_path}")
        return True
        
    except Exception as e:
        print(f"Ошибка при инвертировании координат в JSON файле: {e}")
        return False

if __name__ == "__main__":
    # Путь к JSON файлу
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "wall_coordinates.json")
    
    # Инвертируем координаты в JSON файле
    success = invert_json_coordinates(json_path)
    
    if success:
        print("Инвертирование координат в JSON файле завершено успешно")
    else:
        print("Ошибка при инвертировании координат в JSON файле")