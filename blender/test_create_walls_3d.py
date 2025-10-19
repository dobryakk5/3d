"""
Тестовый скрипт для проверки функциональности create_walls_3d.py
"""

import bpy
import os
import sys

# Добавляем путь к текущей директории для импорта модуля
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from create_walls_3d import create_3d_walls_from_coordinates
    print("Модуль create_walls_3d успешно импортирован")
except ImportError as e:
    print(f"Ошибка импорта модуля: {e}")
    sys.exit(1)

def test_wall_creation():
    """
    Тестовая функция для создания 3D стен
    """
    print("=" * 50)
    print("ТЕСТ СОЗДАНИЯ 3D СТЕН")
    print("=" * 50)
    
    # Путь к файлу с координатами
    json_path = os.path.join(current_dir, "wall_coordinates.json")
    
    # Проверяем существование файла
    if not os.path.exists(json_path):
        print(f"ОШИБКА: Файл {json_path} не найден!")
        return False
    
    print(f"Используется файл: {json_path}")
    
    # Создаем 3D стены с высотой 2.5 метра
    wall_height = 2.5
    print(f"Высота стен: {wall_height} метра")
    
    try:
        # Вызываем основную функцию
        result = create_3d_walls_from_coordinates(
            json_path=json_path,
            wall_height=wall_height,
            export_obj=True,
            clear_scene=True
        )
        
        if result:
            print("\n✅ Тест успешно пройден!")
            
            # Проверяем наличие объекта в сцене
            if "Walls_3D" in bpy.data.objects:
                wall_obj = bpy.data.objects["Walls_3D"]
                print(f"Создан объект: {wall_obj.name}")
                print(f"Количество вершин: {len(wall_obj.data.vertices)}")
                print(f"Количество полигонов: {len(wall_obj.data.polygons)}")
            
            # Проверяем наличие экспортированного файла
            obj_path = os.path.join(current_dir, "wall_coordinates_3d.obj")
            if os.path.exists(obj_path):
                print(f"Экспортированный файл: {obj_path}")
            
            return True
        else:
            print("\n❌ Тест не пройден!")
            return False
            
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении теста: {e}")
        return False

def test_different_heights():
    """
    Тест с разной высотой стен
    """
    print("\n" + "=" * 50)
    print("ТЕСТ С РАЗНОЙ ВЫСОТОЙ СТЕН")
    print("=" * 50)
    
    json_path = os.path.join(current_dir, "wall_coordinates.json")
    
    heights = [2.5, 3.0, 3.5]
    
    for height in heights:
        print(f"\nТест с высотой стен: {height} метра")
        try:
            result = create_3d_walls_from_coordinates(
                json_path=json_path,
                wall_height=height,
                export_obj=False,  # Не экспортируем для экономии времени
                clear_scene=True
            )
            
            if result:
                print(f"✅ Успешно созданы стены высотой {height} метра")
            else:
                print(f"❌ Ошибка при создании стен высотой {height} метра")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    # Запускаем основной тест
    success = test_wall_creation()
    
    # Запускаем дополнительные тесты
    test_different_heights()
    
    print("\n" + "=" * 50)
    if success:
        print("ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО")
    else:
        print("ТЕСТЫ ЗАВЕРШЕНЫ С ОШИБКАМИ")
    print("=" * 50)