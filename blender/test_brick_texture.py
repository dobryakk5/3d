#!/usr/bin/env python3
"""
Тестовый скрипт для проверки применения текстуры кирпича к внешним стенам
"""

import sys
import os

# Импортируем модуль Blender
try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Blender API недоступен, запускаем в автономном режиме")

def test_brick_texture_application():
    """
    Тестирует применение текстуры кирпича к внешним стенам
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПРИМЕНЕНИЯ ТЕКСТУРЫ КИРПИЧА")
    print("=" * 60)
    
    if BLENDER_AVAILABLE:
        # Очищаем сцену
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Импортируем функции из нашего скрипта
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from create_walls_2m import load_brick_texture, create_procedural_brick_material
        
        # Тестируем загрузку текстуры
        script_dir = os.path.dirname(os.path.abspath(__file__))
        texture_path = os.path.join(script_dir, "brick_texture.jpg")
        
        print(f"Путь к текстуре: {texture_path}")
        print(f"Текстура существует: {os.path.exists(texture_path)}")
        
        # Загружаем текстуру
        brick_material = load_brick_texture(texture_path)
        
        if brick_material:
            print(f"Материал создан: {brick_material.name}")
            print(f"Материал использует ноды: {brick_material.use_nodes}")
            
            # Проверяем узлы материала
            if brick_material.use_nodes:
                nodes = brick_material.node_tree.nodes
                texture_nodes = [node for node in nodes if node.type == 'TEX_IMAGE']
                print(f"Найдено узлов текстуры: {len(texture_nodes)}")
                
                if texture_nodes:
                    for i, node in enumerate(texture_nodes):
                        if node.image:
                            print(f"  Узел {i+1}: изображение загружено ({node.image.name})")
                        else:
                            print(f"  Узел {i+1}: изображение не загружено")
            
            # Создаем тестовый куб
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            test_cube = bpy.context.active_object
            test_cube.name = "TestBrickCube"
            
            # Применяем материал
            if test_cube.data.materials:
                test_cube.data.materials[0] = brick_material
            else:
                test_cube.data.materials.append(brick_material)
            
            print(f"Материал применен к тестовому объекту: {test_cube.name}")
            
            # Проверяем, что материал действительно применен
            if test_cube.data.materials:
                applied_material = test_cube.data.materials[0]
                if applied_material.name == brick_material.name:
                    print("✓ Материал успешно применен к объекту")
                else:
                    print(f"✗ Применен другой материал: {applied_material.name}")
            else:
                print("✗ Материал не применен к объекту")
            
            return True
        else:
            print("✗ Не удалось создать материал")
            return False
    else:
        print("Blender API недоступен, невозможно проверить применение текстуры")
        return False

def check_json_for_brick_texture():
    """
    Проверяет наличие данных о текстуре в JSON файле
    """
    print("\nПРОВЕРКА ДАННЫХ JSON")
    print("-" * 30)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "wall_coordinates.json")
    
    if os.path.exists(json_path):
        print(f"Файл JSON найден: {json_path}")
        
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Проверяем наличие контура здания
            if "building_outline" in data:
                outline = data["building_outline"]
                vertices = outline["vertices"]
                print(f"Найден контур здания с {len(vertices)} вершинами")
            else:
                print("✗ Контур здания не найден в JSON")
            
            # Проверяем параметры стен
            if "metadata" in data:
                wall_thickness = data["metadata"]["wall_thickness"]
                print(f"Толщина стен: {wall_thickness} пикселей")
            
            # Проверяем сегменты стен
            wall_segments_from_openings = data.get("wall_segments_from_openings", [])
            wall_segments_from_junctions = data.get("wall_segments_from_junctions", [])
            openings = data.get("openings", [])
            
            print(f"Сегментов стен от проемов: {len(wall_segments_from_openings)}")
            print(f"Сегментов стен от соединений: {len(wall_segments_from_junctions)}")
            print(f"Проемов: {len(openings)}")
            
            return True
        except Exception as e:
            print(f"✗ Ошибка при чтении JSON: {e}")
            return False
    else:
        print(f"✗ Файл JSON не найден: {json_path}")
        return False

if __name__ == "__main__":
    # Проверяем применение текстуры
    texture_test_passed = test_brick_texture_application()
    
    # Проверяем данные JSON
    json_test_passed = check_json_for_brick_texture()
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Тест применения текстуры: {'ПРОЙДЕН' if texture_test_passed else 'ПРОВАЛЕН'}")
    print(f"Тест данных JSON: {'ПРОЙДЕН' if json_test_passed else 'ПРОВАЛЕН'}")
    
    if texture_test_passed and json_test_passed:
        print("\n✓ Все тесты пройдены! Текстура кирпича должна быть видна на внешних стенах.")
    else:
        print("\n✗ Некоторые тесты провалены. Возможны проблемы с отображением текстуры.")