#!/usr/bin/env python3
"""
Итоговый скрипт для решения проблемы J18-J25
Выравнивание стен
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import math

def normalize_opening_thickness(data):
    """
    Приводит все проемы к одной толщине (минимальная толщина двери)
    """
    print("\n1.1. Нормализация толщины проемов...")
    
    # Находим минимальную толщину двери
    door_thicknesses = []
    for opening in data.get('openings', []):
        if opening.get('type') == 'door':
            bbox = opening.get('bbox', {})
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            thickness = min(width, height)
            if thickness > 0:
                door_thicknesses.append(thickness)
    
    if not door_thicknesses:
        # Если дверей нет, используем стандартную толщину
        wall_thickness = 20.0
        print(f"  ✓ Двери не найдены, используем стандартную толщину: {wall_thickness} px")
    else:
        wall_thickness = min(door_thicknesses)
        print(f"  ✓ Минимальная толщина двери определена: {wall_thickness} px")
    
    # Нормализуем толщину всех проемов
    normalized_count = 0
    for opening in data.get('openings', []):
        bbox = opening.get('bbox', {})
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        # Определяем ориентацию проема
        if width > height:  # Горизонтальный проем
            # Высота становится толщиной стены
            if height != wall_thickness:
                # Центрируем по вертикали
                y_center = bbox['y'] + height / 2
                new_y = y_center - wall_thickness / 2
                bbox['y'] = new_y
                bbox['height'] = wall_thickness
                normalized_count += 1
                print(f"    ✓ {opening['id']}: высота {height} -> {wall_thickness}")
        else:  # Вертикальный проем
            # Ширина становится толщиной стены
            if width != wall_thickness:
                # Центрируем по горизонтали
                x_center = bbox['x'] + width / 2
                new_x = x_center - wall_thickness / 2
                bbox['x'] = new_x
                bbox['width'] = wall_thickness
                normalized_count += 1
                print(f"    ✓ {opening['id']}: ширина {width} -> {wall_thickness}")
    
    print(f"  ✓ Нормализовано {normalized_count} проемов к толщине {wall_thickness} px")
    return data

def apply_alignment(data, tolerance=50.0, debug=False):
    """Apply wall alignment to the entire dataset"""
    if debug:
        print("  [ALIGNMENT] Starting wall alignment process")
    
    openings = data.get('openings', [])
    
    if debug:
        print(f"  [ALIGNMENT] Processing {len(openings)} openings")
    
    # Group openings by wall
    wall_groups = group_openings_by_wall(openings, tolerance, debug)
    
    # Create aligned dataset
    aligned_data = data.copy()
    aligned_openings = []
    processed_opening_ids = set()
    
    for group in wall_groups:
        if len(group['openings']) > 1:  # Only align walls with multiple openings
            if debug:
                print(f"  [ALIGNMENT] Aligning wall group at {group['position']} ({group['orientation']})")
            aligned_group = align_wall_group(group, debug)
            aligned_openings.extend(aligned_group)
            
            # Mark as processed
            for opening in group['openings']:
                processed_opening_ids.add(opening.get('id', ''))
        else:
            # Single opening wall - add as is
            aligned_openings.extend(group['openings'])
    
    # Add unprocessed openings (single-opening walls)
    unprocessed_count = 0
    for opening in openings:
        if opening.get('id', '') not in processed_opening_ids:
            aligned_openings.append(opening)
            unprocessed_count += 1
    
    if debug:
        print(f"  [ALIGNMENT] Processed {len(processed_opening_ids)} openings in {len(wall_groups)} wall groups")
        print(f"  [ALIGNMENT] Left {unprocessed_count} single-opening walls unmodified")
        print("  [ALIGNMENT] Wall alignment completed")
    
    aligned_data['openings'] = aligned_openings
    return aligned_data

def group_openings_by_wall(openings, tolerance=50.0, debug=False):
    """Group openings that belong to the same wall"""
    if debug:
        print(f"  [ALIGNMENT] Grouping {len(openings)} openings by wall")
    groups = []
    
    # Group by orientation first
    horizontal_openings = []
    vertical_openings = []
    
    for opening in openings:
        bbox = opening.get('bbox', {})
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        if width > height:
            horizontal_openings.append(opening)
        else:
            vertical_openings.append(opening)
    
    if debug:
        print(f"  [ALIGNMENT] Found {len(horizontal_openings)} horizontal and {len(vertical_openings)} vertical openings")
    
    # Group horizontal openings by Y-coordinate
    h_groups = group_by_position(horizontal_openings, 'y', 'horizontal', tolerance, debug)
    # Group vertical openings by X-coordinate
    v_groups = group_by_position(vertical_openings, 'x', 'vertical', tolerance, debug)
    
    groups.extend(h_groups)
    groups.extend(v_groups)
    
    if debug:
        print(f"  [ALIGNMENT] Created {len(groups)} wall groups")
    return groups

def group_by_position(openings, position_key, orientation, tolerance=50.0, debug=False):
    """Group openings by their position coordinate"""
    groups = []
    ungrouped = openings.copy()
    
    while ungrouped:
        # Start a new group
        reference = ungrouped.pop(0)
        ref_bbox = reference.get('bbox', {})
        ref_pos = ref_bbox.get(position_key, 0)
        
        group_openings = [reference]
        
        # Find all openings within tolerance
        i = 0
        while i < len(ungrouped):
            opening = ungrouped[i]
            bbox = opening.get('bbox', {})
            pos = bbox.get(position_key, 0)
            
            if abs(pos - ref_pos) <= tolerance:
                group_openings.append(ungrouped.pop(i))
            else:
                i += 1
        
        # Only create group if multiple openings
        if len(group_openings) > 1:
            group = {
                'orientation': orientation,
                'position': ref_pos,
                'openings': group_openings
            }
            groups.append(group)
            if debug:
                print(f"  [ALIGNMENT] Created {orientation} group at {position_key}={ref_pos:.1f} with {len(group_openings)} openings")
    
    return groups

def align_wall_group(wall_group, debug=False):
    """Align all openings in a wall group to the furthest-out opening"""
    orientation = wall_group['orientation']
    openings = wall_group['openings']
    
    if orientation == 'horizontal':
        # Find opening with minimum Y (closest to top/exterior)
        furthest_out = min(openings, key=lambda o: o['bbox']['y'])
        target_y = furthest_out['bbox']['y']
        
        if debug:
            print(f"  [ALIGNMENT] Target Y position: {target_y:.1f} (opening {furthest_out.get('id', 'unknown')})")
        
        aligned_openings = []
        for opening in openings:
            aligned = create_aligned_opening(opening, target_y, 'y')
            aligned_openings.append(aligned)
            if debug:
                print(f"  [ALIGNMENT] Aligned opening {opening.get('id', 'unknown')}: Y {opening['bbox']['y']:.1f} -> {target_y:.1f}")
        
        return aligned_openings
    
    else:  # vertical
        # Find opening with minimum X (closest to left/exterior)
        furthest_out = min(openings, key=lambda o: o['bbox']['x'])
        target_x = furthest_out['bbox']['x']
        
        if debug:
            print(f"  [ALIGNMENT] Target X position: {target_x:.1f} (opening {furthest_out.get('id', 'unknown')})")
        
        aligned_openings = []
        for opening in openings:
            aligned = create_aligned_opening(opening, target_x, 'x')
            aligned_openings.append(aligned)
            if debug:
                print(f"  [ALIGNMENT] Aligned opening {opening.get('id', 'unknown')}: X {opening['bbox']['x']:.1f} -> {target_x:.1f}")
        
        return aligned_openings

def create_aligned_opening(opening, target_position, position_key):
    """Create an aligned copy of an opening"""
    aligned = opening.copy()
    aligned_bbox = opening['bbox'].copy()
    
    # Update position
    original_pos = aligned_bbox[position_key]
    aligned_bbox[position_key] = target_position
    
    # Store alignment info
    aligned['original_bbox'] = opening['bbox']
    aligned['aligned_bbox'] = aligned_bbox
    aligned['alignment_offset'] = target_position - original_pos
    
    aligned['bbox'] = aligned_bbox
    return aligned

# =============================================================================
# ФУНКЦИИ ВЫРАВНИВАНИЯ СТЕН ПО ПРОЕМАМ
# =============================================================================

def get_min_door_thickness(data):
    """
    Определяет минимальную толщину дверей для использования как допуск
    
    Args:
        data: Словарь с данными плана
    
    Returns:
        Минимальная толщина двери в пикселях
    """
    door_thicknesses = []
    for opening in data.get('openings', []):
        if opening.get('type') == 'door':
            bbox = opening.get('bbox', {})
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            thickness = min(width, height)
            if thickness > 0:
                door_thicknesses.append(thickness)
    
    if door_thicknesses:
        return min(door_thicknesses)
    else:
        # Если дверей нет, используем стандартную толщину
        return 20.0

def get_wall_orientation(wall):
    """
    Определяет ориентацию стены (горизонтальная/вертикальная)
    
    Args:
        wall: Словарь с информацией о стене
    
    Returns:
        'horizontal', 'vertical' или 'unknown'
    """
    # Проверяем стены с start/end координатами
    if 'start' in wall and 'end' in wall:
        start = wall['start']
        end = wall['end']
        
        dx = abs(end['x'] - start['x'])
        dy = abs(end['y'] - start['y'])
        
        if dx > dy * 1.5:
            return 'horizontal'
        elif dy > dx * 1.5:
            return 'vertical'
        else:
            return 'unknown'
    
    # Проверяем стены в формате bbox
    elif 'width' in wall and 'height' in wall and 'x' in wall and 'y' in wall:
        width = wall['width']
        height = wall['height']
        
        if width > height * 1.5:
            return 'horizontal'
        elif height > width * 1.5:
            return 'vertical'
        else:
            return 'unknown'
    
    # Проверяем стены с явным указанием ориентации
    elif 'orientation' in wall:
        return wall['orientation']
    
    return 'unknown'

def group_walls_by_orientation_and_position(walls, tolerance):
    """
    Группирует стены по ориентации и позиции с учетом допуска
    
    Args:
        walls: Список стен
        tolerance: Допуск для группировки
    
    Returns:
        Словарь с группами стен
        {
            'horizontal': {position1: [walls], position2: [walls], ...},
            'vertical': {position1: [walls], position2: [walls], ...}
        }
    """
    groups = {
        'horizontal': {},
        'vertical': {}
    }
    
    # Разделяем стены по ориентации
    horizontal_walls = []
    vertical_walls = []
    
    for wall in walls:
        orientation = get_wall_orientation(wall)
        if orientation == 'horizontal':
            horizontal_walls.append(wall)
        elif orientation == 'vertical':
            vertical_walls.append(wall)
    
    # Группируем горизонтальные стены по Y-координате
    h_groups = {}
    for wall in horizontal_walls:
        # Определяем Y-координату в зависимости от формата стены
        if 'start' in wall:
            y_pos = wall['start']['y']  # Формат start/end
        else:
            y_pos = wall['y'] + wall['height'] / 2  # Формат bbox, используем центр
        
        # Ищем существующую группу в пределах допуска
        found_group = None
        for group_y in h_groups:
            if abs(group_y - y_pos) <= tolerance:
                found_group = group_y
                break
        
        if found_group is not None:
            h_groups[found_group].append(wall)
        else:
            h_groups[y_pos] = [wall]
    
    groups['horizontal'] = h_groups
    
    # Группируем вертикальные стены по X-координате
    v_groups = {}
    for wall in vertical_walls:
        # Определяем X-координату в зависимости от формата стены
        if 'start' in wall:
            x_pos = wall['start']['x']  # Формат start/end
        else:
            x_pos = wall['x'] + wall['width'] / 2  # Формат bbox, используем центр
        
        # Ищем существующую группу в пределах допуска
        found_group = None
        for group_x in v_groups:
            if abs(group_x - x_pos) <= tolerance:
                found_group = group_x
                break
        
        if found_group is not None:
            v_groups[found_group].append(wall)
        else:
            v_groups[x_pos] = [wall]
    
    groups['vertical'] = v_groups
    
    return groups

def find_openings_for_wall_group(wall_group, openings, tolerance):
    """
    Находит проемы на той же горизонтали/вертикали для группы стен
    
    Args:
        wall_group: Словарь с информацией о группе стен
        openings: Список всех проемов
        tolerance: Допуск для поиска
    
    Returns:
        Список проемов на той же горизонтали/вертикали
    """
    orientation = wall_group['orientation']
    position = wall_group['position']
    related_openings = []
    
    for opening in openings:
        bbox = opening.get('bbox', {})
        if not bbox:
            continue
        
        # Определяем ориентацию проема
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        opening_orientation = 'horizontal' if width > height else 'vertical'
        
        # Проверяем, находится ли проем на той же горизонтали/вертикали
        if orientation == 'horizontal' and opening_orientation == 'horizontal':
            # Для горизонтальных стен проверяем Y-координату проема
            opening_y = bbox['y']
            if abs(opening_y - position) <= tolerance:
                related_openings.append(opening)
        
        elif orientation == 'vertical' and opening_orientation == 'vertical':
            # Для вертикальных стен проверяем X-координату проема
            opening_x = bbox['x']
            if abs(opening_x - position) <= tolerance:
                related_openings.append(opening)
    
    return related_openings

def select_reference_opening(wall_group, related_openings):
    """
    Выбирает проем для выравнивания (ближайший к центру группы стен)
    
    Args:
        wall_group: Группа стен
        related_openings: Список проемов на той же горизонтали/вертикали
    
    Returns:
        Выбранный проем
    """
    if not related_openings:
        return None
    
    # Вычисляем центр группы стен
    walls = wall_group['walls']
    if not walls:
        return related_openings[0]
    
    center_x = 0
    center_y = 0
    for wall in walls:
        if 'start' in wall and 'end' in wall:
            # Формат start/end
            center_x += (wall['start']['x'] + wall['end']['x']) / 2
            center_y += (wall['start']['y'] + wall['end']['y']) / 2
        else:
            # Формат bbox
            center_x += wall['x'] + wall['width'] / 2
            center_y += wall['y'] + wall['height'] / 2
    
    center_x /= len(walls)
    center_y /= len(walls)
    
    # Находим ближайший проем к центру группы
    min_distance = float('inf')
    reference_opening = None
    
    for opening in related_openings:
        bbox = opening.get('bbox', {})
        opening_center_x = bbox['x'] + bbox['width'] / 2
        opening_center_y = bbox['y'] + bbox['height'] / 2
        
        distance = math.sqrt((opening_center_x - center_x)**2 + (opening_center_y - center_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            reference_opening = opening
    
    return reference_opening

def align_wall_group_to_opening(wall_group, reference_opening, debug=False):
    """
    Выравнивает группу стен по уровню проема
    
    Args:
        wall_group: Группа стен
        reference_opening: Проем для выравнивания
        debug: Флаг отладочной информации
    
    Returns:
        Список выровненных стен
    """
    orientation = wall_group['orientation']
    walls = wall_group['walls']
    
    if not reference_opening:
        return walls
    
    bbox = reference_opening.get('bbox', {})
    if not bbox:
        return walls
    
    aligned_walls = []
    
    if orientation == 'horizontal':
        # Выравниваем горизонтальные стены по Y-координате проема
        target_y = bbox['y']
        
        for wall in walls:
            aligned_wall = wall.copy()
            
            # Проверяем формат стены
            if 'start' in wall and 'end' in wall:
                # Формат start/end
                start = aligned_wall['start'].copy()
                end = aligned_wall['end'].copy()
                
                # Сохраняем исходные координаты
                original_start_y = start['y']
                original_end_y = end['y']
                
                # Выравниваем Y-координаты
                start['y'] = target_y
                end['y'] = target_y
                
                aligned_wall['start'] = start
                aligned_wall['end'] = end
                
                # Сохраняем информацию о выравнивании
                aligned_wall['original_start'] = wall['start']
                aligned_wall['original_end'] = wall['end']
                aligned_wall['alignment_offset'] = target_y - original_start_y
                
                if debug:
                    print(f"    Стена {wall.get('id', 'unknown')}: Y {original_start_y} -> {target_y}")
            
            else:
                # Формат bbox
                original_y = aligned_wall['y']
                
                # Выравниваем Y-координату
                aligned_wall['y'] = target_y
                
                # Сохраняем информацию о выравнивании
                aligned_wall['original_y'] = original_y
                aligned_wall['alignment_offset'] = target_y - original_y
                
                if debug:
                    print(f"    Стена {wall.get('id', 'unknown')}: Y {original_y} -> {target_y}")
            
            aligned_walls.append(aligned_wall)
    
    elif orientation == 'vertical':
        # Выравниваем вертикальные стены по X-координате проема
        target_x = bbox['x']
        
        for wall in walls:
            aligned_wall = wall.copy()
            
            # Проверяем формат стены
            if 'start' in wall and 'end' in wall:
                # Формат start/end
                start = aligned_wall['start'].copy()
                end = aligned_wall['end'].copy()
                
                # Сохраняем исходные координаты
                original_start_x = start['x']
                original_end_x = end['x']
                
                # Выравниваем X-координаты
                start['x'] = target_x
                end['x'] = target_x
                
                aligned_wall['start'] = start
                aligned_wall['end'] = end
                
                # Сохраняем информацию о выравнивании
                aligned_wall['original_start'] = wall['start']
                aligned_wall['original_end'] = wall['end']
                aligned_wall['alignment_offset'] = target_x - original_start_x
                
                if debug:
                    print(f"    Стена {wall.get('id', 'unknown')}: X {original_start_x} -> {target_x}")
            
            else:
                # Формат bbox
                original_x = aligned_wall['x']
                
                # Выравниваем X-координату
                aligned_wall['x'] = target_x
                
                # Сохраняем информацию о выравнивании
                aligned_wall['original_x'] = original_x
                aligned_wall['alignment_offset'] = target_x - original_x
                
                if debug:
                    print(f"    Стена {wall.get('id', 'unknown')}: X {original_x} -> {target_x}")
            
            aligned_walls.append(aligned_wall)
    
    return aligned_walls

def align_walls_by_openings(data, tolerance=None, debug=False):
    """
    Выравнивает стены по уровню проемов на той же горизонтали или вертикали
    
    Args:
        data: Словарь с данными плана
        tolerance: Допуск для группировки (по умолчанию - минимальная толщина двери)
        debug: Флаг отладочной информации
    
    Returns:
        Обновленные данные с выровненными стенами
    """
    if debug:
        print("  [WALL ALIGNMENT] Starting wall alignment by openings")
    
    # 1. Определение толщины допуска
    if tolerance is None:
        tolerance = get_min_door_thickness(data)
    
    if debug:
        print(f"  [WALL ALIGNMENT] Using tolerance: {tolerance} px")
    
    # 2. Получение стен и проемов
    walls = data.get('walls', [])
    openings = data.get('openings', [])
    
    if debug:
        print(f"  [WALL ALIGNMENT] Processing {len(walls)} walls and {len(openings)} openings")
    
    # 3. Группировка стен по ориентации и позиции
    wall_groups = group_walls_by_orientation_and_position(walls, tolerance)
    
    if debug:
        h_groups_count = sum(len(group) for group in wall_groups['horizontal'].values())
        v_groups_count = sum(len(group) for group in wall_groups['vertical'].values())
        print(f"  [WALL ALIGNMENT] Found {len(wall_groups['horizontal'])} horizontal groups with {h_groups_count} walls")
        print(f"  [WALL ALIGNMENT] Found {len(wall_groups['vertical'])} vertical groups with {v_groups_count} walls")
    
    # 4. Выравнивание каждой группы
    aligned_walls = []
    aligned_groups_count = 0
    
    # Обработка горизонтальных групп
    for position, group_walls in wall_groups['horizontal'].items():
        wall_group = {
            'orientation': 'horizontal',
            'position': position,
            'walls': group_walls
        }
        
        # Поиск проемов для этой группы
        related_openings = find_openings_for_wall_group(wall_group, openings, tolerance)
        
        if related_openings:
            # Выбор проема для выравнивания
            reference_opening = select_reference_opening(wall_group, related_openings)
            
            if reference_opening:
                # Выравнивание стен по проему
                aligned_group = align_wall_group_to_opening(wall_group, reference_opening, debug)
                aligned_walls.extend(aligned_group)
                aligned_groups_count += 1
                
                if debug:
                    print(f"  [WALL ALIGNMENT] Aligned horizontal group at Y={position} by opening {reference_opening.get('id')}")
            else:
                # Проемы найдены, но не удалось выбрать опорный
                aligned_walls.extend(group_walls)
        else:
            # Проемы не найдены, оставляем стены как есть
            aligned_walls.extend(group_walls)
            
            if debug:
                print(f"  [WALL ALIGNMENT] No openings found for horizontal group at Y={position}")
    
    # Обработка вертикальных групп
    for position, group_walls in wall_groups['vertical'].items():
        wall_group = {
            'orientation': 'vertical',
            'position': position,
            'walls': group_walls
        }
        
        # Поиск проемов для этой группы
        related_openings = find_openings_for_wall_group(wall_group, openings, tolerance)
        
        if related_openings:
            # Выбор проема для выравнивания
            reference_opening = select_reference_opening(wall_group, related_openings)
            
            if reference_opening:
                # Выравнивание стен по проему
                aligned_group = align_wall_group_to_opening(wall_group, reference_opening, debug)
                aligned_walls.extend(aligned_group)
                aligned_groups_count += 1
                
                if debug:
                    print(f"  [WALL ALIGNMENT] Aligned vertical group at X={position} by opening {reference_opening.get('id')}")
            else:
                # Проемы найдены, но не удалось выбрать опорный
                aligned_walls.extend(group_walls)
        else:
            # Проемы не найдены, оставляем стены как есть
            aligned_walls.extend(group_walls)
            
            if debug:
                print(f"  [WALL ALIGNMENT] No openings found for vertical group at X={position}")
    
    if debug:
        print(f"  [WALL ALIGNMENT] Aligned {aligned_groups_count} wall groups")
        print("  [WALL ALIGNMENT] Wall alignment by openings completed")
    
    # 5. Обновление данных
    result_data = data.copy()
    result_data['walls'] = aligned_walls
    
    return result_data

def main():
    print("="*60)
    print("РЕШЕНИЕ ПРОБЛЕМЫ J18-J25 - ВЫРАВНИВАНИЕ ПРОЕМОВ")
    print("="*60)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        # Перезаписываем тот же файл (in-place)
        output_file = input_file
        print(f"Используем входной файл: {input_file}")
        print(f"Выходной файл (перезапись): {output_file}")
    else:
        # Используем файлы по умолчанию (in-place)
        input_file = 'plan_floor1_objects.json'
        output_file = 'plan_floor1_objects.json'
        print(f"Используем файл по умолчанию: {input_file}")
    
    # Шаг 1: Загрузка данных
    print("\n1. Загрузка данных...")
    try:
        # Используем io для чтения файла
        import io
        
        # Открываем файл в бинарном режиме
        with open(input_file, 'rb') as f:
            raw_data = f.read()
        
        # Пробуем декодировать как UTF-8
        try:
            text_data = raw_data.decode('utf-8')
            data = json.loads(text_data)
            print(f"  ✓ Файл загружен с кодировкой: UTF-8")
        except UnicodeDecodeError:
            # Пробуем другие кодировки
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    text_data = raw_data.decode(encoding)
                    data = json.loads(text_data)
                    print(f"  ✓ Файл загружен с кодировкой: {encoding}")
                    break
                except Exception:
                    continue
            else:
                print(f"  ✗ Не удалось загрузить файл с поддерживаемыми кодировками")
                raise
    except Exception as e:
        print(f"  ✗ Ошибка загрузки файла: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Шаг 2: Нормализация толщины проемов
    data = normalize_opening_thickness(data)
    
    # Шаг 3: Применение выравнивания проемов
    print("\n3. Применение выравнивания проемов...")
    aligned_data = apply_alignment(data, tolerance=50.0, debug=True)
    
    # Шаг 4: Сохранение выравненных данных
    print("\n4. Сохранение выравненных данных...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aligned_data, f, indent=2)
    
    # Шаг 5: Проверка результатов выравнивания проемов
    print("\n5. Проверка результатов выравнивания проемов...")
    openings = aligned_data.get('openings', [])
    
    # Проверяем горизонтальные стены с множественными проемами
    horizontal_groups = {}
    for opening in openings:
        bbox = opening['bbox']
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        if width > height:  # Horizontal opening
            y_pos = bbox['y']
            if y_pos not in horizontal_groups:
                horizontal_groups[y_pos] = []
            horizontal_groups[y_pos].append(opening)
    
    # Проверяем выравнивание проемов
    aligned_count = 0
    for y_pos, group in horizontal_groups.items():
        if len(group) > 1:
            positions = [o['bbox']['y'] for o in group]
            unique_positions = set(positions)
            if len(unique_positions) == 1:
                aligned_count += 1
                print(f"  ✓ Горизонтальная стена Y={y_pos} выровнена ({len(group)} openings)")
            else:
                print(f"  ✗ Горизонтальная стена Y={y_pos} не выровнена: {unique_positions}")
    
    # Шаг 6: Завершение
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"  ✓ Выровнено {aligned_count} горизонтальных стен по проемам")
    print(f"  ✓ Данные сохранены в {output_file}")
    print("="*60)
    
    print("\n✓ Проблема J18-J25 решена!")
    print("\n✓ Выравнивание проемов выполнено!")

if __name__ == '__main__':
    main()
