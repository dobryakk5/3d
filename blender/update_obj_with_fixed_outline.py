#!/usr/bin/env python3
"""
Обновление OBJ файла с исправленными координатами контура здания

Скрипт читает существующий OBJ файл и JSON с исправленными координатами,
затем обновляет вершины контура здания в OBJ файле.
"""

import json
import re
from typing import Dict, List, Tuple

def load_json_data(file_path: str) -> Dict:
    """Загружает данные из JSON файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_obj_file(file_path: str) -> List[str]:
    """Загружает OBJ файл как список строк"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def save_obj_file(lines: List[str], file_path: str) -> None:
    """Сохраняет OBJ файл"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def find_outline_wall_vertices(obj_lines: List[str]) -> List[int]:
    """
    Находит индексы вершин контура здания в OBJ файле
    
    Returns:
        Список индексов строк с вершинами контура здания
    """
    # Ищем секцию Outline_Wall
    outline_section_start = None
    outline_section_end = None
    
    for i, line in enumerate(obj_lines):
        if line.startswith('o Outline_Wall'):
            outline_section_start = i
        elif outline_section_start is not None and line.startswith('o ') and not line.startswith('o Outline_Wall'):
            outline_section_end = i
            break
    
    if outline_section_start is None:
        print("Предупреждение: секция Outline_Wall не найдена в OBJ файле")
        return []
    
    # Ищем вершины в секции Outline_Wall
    vertex_indices = []
    for i in range(outline_section_start, outline_section_end if outline_section_end else len(obj_lines)):
        if obj_lines[i].startswith('v '):
            vertex_indices.append(i)
    
    return vertex_indices

def update_outline_vertices(obj_lines: List[str], vertex_indices: List[int],
                          outline_vertices: List[Dict], scale_factor: float = 0.01) -> List[str]:
    """
    Обновляет вершины контура здания в OBJ файле
    
    Args:
        obj_lines: Строки OBJ файла
        vertex_indices: Индексы строк с вершинами контура
        outline_vertices: Вершины контура из JSON
        scale_factor: Масштабный коэффициент для преобразования координат
    
    Returns:
        Обновленные строки OBJ файла
    """
    # Проверяем, что количество вершин в OBJ в 2 раза больше, чем в JSON
    # (каждая вершина дублируется для нижней и верхней части стены)
    if len(vertex_indices) != len(outline_vertices) * 2:
        print(f"Предупреждение: количество вершин не совпадает")
        print(f"  Вершин в OBJ: {len(vertex_indices)}")
        print(f"  Вершин в JSON: {len(outline_vertices)}")
        print(f"  Ожидается: {len(outline_vertices) * 2} вершин в OBJ")
        return obj_lines
    
    print(f"Обновление {len(vertex_indices)} вершин из {len(outline_vertices)} вершин JSON")
    
    # Обновляем вершины
    updated_lines = obj_lines.copy()
    
    # OBJ использует систему координат (x, z, y) с инвертированием Y
    # Нижние вершины (первые половина)
    for i in range(len(outline_vertices)):
        # Нижняя вершина
        line_idx = vertex_indices[i]
        vertex = outline_vertices[i]
        
        x = vertex['x'] * scale_factor
        y = vertex['y'] * scale_factor
        
        # В Blender используется (x, y, z), но в OBJ это становится (x, z, y) с инвертированием Y
        updated_lines[line_idx] = f"v {x:.6f} 0.000000 {-y:.6f}\n"
        
        # Верхняя вершина (с тем же индексом + смещение)
        line_idx = vertex_indices[i + len(outline_vertices)]
        
        # В Blender используется (x, y, z), но в OBJ это становится (x, z, y) с инвертированием Y
        wall_height = 3.0  # Высота стены из create_walls_2m.py
        updated_lines[line_idx] = f"v {x:.6f} {wall_height:.6f} {-y:.6f}\n"
    
    return updated_lines

def update_obj_with_fixed_outline():
    """Основная функция обновления OBJ файла"""
    print("=" * 60)
    print("ОБНОВЛЕНИЕ OBJ ФАЙЛА С ИСПРАВЛЕННЫМИ КООРДИНАТАМИ КОНТУРА")
    print("=" * 60)
    
    # Пути к файлам
    json_path = 'wall_coordinates.json'
    input_obj_path = 'wall_coordinates_3d.obj'
    output_obj_path = 'wall_coordinates_3d_fixed_outline.obj'
    
    # Загружаем данные
    print("Загрузка данных...")
    json_data = load_json_data(json_path)
    obj_lines = load_obj_file(input_obj_path)
    
    # Получаем исправленные вершины контура
    outline_vertices = json_data['building_outline']['vertices']
    print(f"Загружено {len(outline_vertices)} вершин контура здания")
    
    # Проверяем информацию о преобразовании
    if 'coordinate_transformation' in json_data['building_outline']:
        trans = json_data['building_outline']['coordinate_transformation']
        print(f"Применено преобразование:")
        print(f"  dx={trans['offset_x']}, dy={trans['offset_y']}")
    
    # Находим вершины контура в OBJ файле
    print("\nПоиск вершин контура в OBJ файле...")
    vertex_indices = find_outline_wall_vertices(obj_lines)
    print(f"Найдено {len(vertex_indices)} вершин контура в OBJ файле")
    
    if not vertex_indices:
        print("Ошибка: вершины контура не найдены в OBJ файле")
        return
    
    # Обновляем вершины
    print("\nОбновление вершин контура...")
    updated_lines = update_outline_vertices(obj_lines, vertex_indices, outline_vertices)
    
    # Сохраняем обновленный OBJ файл
    print(f"\nСохранение обновленного OBJ файла: {output_obj_path}")
    save_obj_file(updated_lines, output_obj_path)
    
    print("\n" + "=" * 60)
    print("ОБНОВЛЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Создан файл: {output_obj_path}")
    print(f"Обновлено вершин: {len(vertex_indices)}")

if __name__ == "__main__":
    update_obj_with_fixed_outline()