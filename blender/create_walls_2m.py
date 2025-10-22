import bpy
import bmesh
import json
import os
from mathutils import Vector, Matrix
import random

def point_to_line_distance(point, line_start, line_end):
    """
    Вычисляет расстояние от точки до отрезка
    
    Args:
        point: точка (x, y)
        line_start: начало отрезка (x, y)
        line_end: конец отрезка (x, y)
    
    Returns:
        float: расстояние от точки до отрезка
    """
    import math
    
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Длина отрезка в квадрате
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    
    if line_length_sq == 0:
        # Отрезок вырожден в точку
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Параметр t для проекции точки на отрезок
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
    
    # Ближайшая точка на отрезке
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    # Расстояние от точки до проекции
    return math.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)

def is_bbox_on_outline(bbox, outline_vertices, tolerance):
    """
    Проверяет, пересекается ли bounding box с контуром здания
    
    Args:
        bbox: bounding box элемента {"x":, "y":, "width":, "height":}
        outline_vertices: список вершин контура здания [{"x":, "y":}, ...]
        tolerance: допустимое расстояние до контура (толщина стены)
    
    Returns:
        bool: True если bbox пересекается с контуром
    """
    # Получаем углы bbox
    bbox_corners = [
        (bbox["x"], bbox["y"]),
        (bbox["x"] + bbox["width"], bbox["y"]),
        (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
        (bbox["x"], bbox["y"] + bbox["height"])
    ]
    
    # Проверяем каждый угол bbox на близость к контуру
    for corner in bbox_corners:
        for i in range(len(outline_vertices)):
            start_vertex = outline_vertices[i]
            end_vertex = outline_vertices[(i + 1) % len(outline_vertices)]
            
            distance = point_to_line_distance(
                corner,
                (start_vertex["x"], start_vertex["y"]),
                (end_vertex["x"], end_vertex["y"])
            )
            
            if distance <= tolerance:
                return True
    
    # Также проверяем центр bbox
    center_x = bbox["x"] + bbox["width"] / 2
    center_y = bbox["y"] + bbox["height"] / 2
    
    for i in range(len(outline_vertices)):
        start_vertex = outline_vertices[i]
        end_vertex = outline_vertices[(i + 1) % len(outline_vertices)]
        
        distance = point_to_line_distance(
            (center_x, center_y),
            (start_vertex["x"], start_vertex["y"]),
            (end_vertex["x"], end_vertex["y"])
        )
        
        if distance <= tolerance:
            return True
    
    return False

def mark_external_elements_by_outline(wall_segments_from_openings, wall_segments_from_junctions,
                                    openings, outline_vertices, wall_thickness, scale_factor=1.0):
    """
    Определяет внешние стены и проемы на основе контура здания
    
    Args:
        wall_segments_from_openings: список сегментов стен от проемов
        wall_segments_from_junctions: список сегментов стен от соединений
        openings: список проемов
        outline_vertices: вершины контура здания
        wall_thickness: толщина стены
        scale_factor: масштабный коэффициент
    
    Returns:
        tuple: (external_wall_ids, external_opening_ids)
    """
    # Применяем масштабирование к толщине стены для tolerance
    tolerance = wall_thickness * scale_factor
    
    external_wall_ids = set()
    external_opening_ids = set()
    
    # Проверяем сегменты стен от проемов
    for segment in wall_segments_from_openings:
        if is_bbox_on_outline(segment["bbox"], outline_vertices, tolerance):
            external_wall_ids.add(segment["segment_id"])
            # Также отмечаем связанный проем как внешний
            if "opening_id" in segment:
                external_opening_ids.add(segment["opening_id"])
    
    # Проверяем сегменты стен от соединений
    for segment in wall_segments_from_junctions:
        if is_bbox_on_outline(segment["bbox"], outline_vertices, tolerance):
            external_wall_ids.add(segment["segment_id"])
    
    # Дополнительно проверяем проемы, которые могут не быть связаны с внешними сегментами
    for opening in openings:
        if is_bbox_on_outline(opening["bbox"], outline_vertices, tolerance):
            external_opening_ids.add(opening["id"])
    
    return external_wall_ids, external_opening_ids

def mark_external_walls_by_junctions(wall_segments_from_openings, wall_segments_from_junctions,
                                   junctions, building_outline):
    """
    Определяет внешние стены на основе последовательности junctions из building_outline
    
    Args:
        wall_segments_from_openings: список сегментов стен от проемов
        wall_segments_from_junctions: список сегментов стен от соединений
        junctions: список junctions
        building_outline: контур здания с последовательностью junctions
    
    Returns:
        set: множество ID внешних стен
    """
    external_wall_ids = set()
    
    # Получаем последовательность junction_id из building_outline
    outline_junction_ids = []
    for vertex in building_outline["vertices"]:
        if "junction_id" in vertex:
            outline_junction_ids.append(vertex["junction_id"])
    
    print(f"Последовательность junctions в контуре: {outline_junction_ids}")
    
    # Создаем словарь для быстрого доступа к junction по ID
    junction_dict = {j["id"]: j for j in junctions}
    
    # Создаем множество всех пар junction_id, которые образуют внешний контур
    outline_pairs = set()
    for i in range(len(outline_junction_ids)):
        current_junction_id = outline_junction_ids[i]
        next_junction_id = outline_junction_ids[(i + 1) % len(outline_junction_ids)]
        
        # Добавляем обе направления пары
        outline_pairs.add((current_junction_id, next_junction_id))
        outline_pairs.add((next_junction_id, current_junction_id))
    
    print(f"Пары junctions внешнего контура: {outline_pairs}")
    
    # Проверяем сегменты стен от проемов
    for segment in wall_segments_from_openings:
        start_junction_id = segment.get("start_junction_id")
        end_junction_id = segment.get("end_junction_id")
        
        if start_junction_id is not None and end_junction_id is not None:
            # Проверяем, образует ли сегмент часть внешнего контура
            if (start_junction_id, end_junction_id) in outline_pairs:
                external_wall_ids.add(segment["segment_id"])
                print(f"  Внешняя стена (проем): {segment['segment_id']} -> junctions {start_junction_id}-{end_junction_id}")
    
    # Проверяем сегменты стен от соединений
    for segment in wall_segments_from_junctions:
        start_junction_id = segment.get("start_junction_id")
        end_junction_id = segment.get("end_junction_id")
        
        if start_junction_id is not None and end_junction_id is not None:
            # Проверяем, образует ли сегмент часть внешнего контура
            if (start_junction_id, end_junction_id) in outline_pairs:
                external_wall_ids.add(segment["segment_id"])
                print(f"  Внешняя стена (соединение): {segment['segment_id']} -> junctions {start_junction_id}-{end_junction_id}")
    
    print(f"Всего найдено внешних стен: {len(external_wall_ids)}")
    return external_wall_ids

def create_junction_labels(junctions, wall_segments_from_openings, wall_segments_from_junctions, scale_factor=1.0, wall_height=2.0):
    """
    Создает текстовые метки с номерами junctions в Blender относительно стен
    
    Args:
        junctions: список junctions
        wall_segments_from_openings: список сегментов стен от проемов
        wall_segments_from_junctions: список сегментов стен от junctions
        scale_factor: масштабный коэффициент
        wall_height: высота стен для позиционирования меток
    
    Returns:
        list: список объектов текстовых меток
    """
    label_objects = []
    
    # Проверяем доступность функции создания текста в Blender
    try:
        # Проверяем, доступен ли bpy
        import bpy
        
        # Создаем словарь для быстрого поиска сегментов стен по ID
        wall_segments = {}
        
        # Добавляем сегменты от проемов
        for segment in wall_segments_from_openings:
            wall_segments[segment["segment_id"]] = segment
        
        # Добавляем сегменты от junctions
        for segment in wall_segments_from_junctions:
            wall_segments[segment["segment_id"]] = segment
        
        for junction in junctions:
            junction_id = junction["id"]
            
            # Ищем связанные сегменты стен
            connected_segments = junction.get("connected_wall_segments", [])
            
            if not connected_segments:
                print(f"Предупреждение: Junction {junction_id} не имеет связанных стен")
                continue
            
            # Берем первый связанный сегмент для позиционирования
            first_segment_id = connected_segments[0]["segment_id"]
            
            if first_segment_id not in wall_segments:
                print(f"Предупреждение: Сегмент стены {first_segment_id} не найден для Junction {junction_id}")
                continue
            
            segment = wall_segments[first_segment_id]
            
            # Определяем позицию для метки на основе сегмента стены
            bbox = segment["bbox"]
            
            # Вычисляем центр сегмента стены
            center_x = (bbox["x"] + bbox["width"] / 2) * scale_factor
            center_y = (bbox["y"] + bbox["height"] / 2) * scale_factor
            
            # Если junction привязан к началу или концу сегмента, смещаем позицию
            connection_type = connected_segments[0].get("connection_type", "")
            if connection_type == "start":
                # Смещаем к началу сегмента
                offset_x = -bbox["width"] * scale_factor * 0.3
                offset_y = -bbox["height"] * scale_factor * 0.3
                center_x += offset_x
                center_y += offset_y
            elif connection_type == "end":
                # Смещаем к концу сегмента
                offset_x = bbox["width"] * scale_factor * 0.3
                offset_y = bbox["height"] * scale_factor * 0.3
                center_x += offset_x
                center_y += offset_y
            
            # Создаем текстовый объект
            bpy.ops.object.text_add(location=(center_x, center_y, wall_height + 0.5))
            text_obj = bpy.context.active_object
            text_obj.name = f"Junction_Label_{junction_id}"
            
            # Устанавливаем текст
            text_obj.data.body = str(junction_id)
            
            # Настраиваем размер шрифта
            text_obj.data.size = 0.3  # Размер шрифта
            
            # Центрируем текст
            text_obj.data.align_x = 'CENTER'
            text_obj.data.align_y = 'CENTER'
            
            # Создаем материал для текста
            text_material = bpy.data.materials.new(name=f"JunctionLabelMaterial_{junction_id}")
            text_material.diffuse_color = (1.0, 1.0, 1.0, 1.0)  # Белый цвет
            text_material.use_nodes = True
            
            # Настраиваем материал для лучшей видимости
            nodes = text_material.node_tree.nodes
            links = text_material.node_tree.links
            
            # Очищаем стандартные узлы
            nodes.clear()
            
            # Создаем основные узлы
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            
            # Настраиваем эмиссию для лучшей видимости
            principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Белый
            
            # Проверяем наличие входов эмиссии (зависит от версии Blender)
            if 'Emission' in principled_bsdf.inputs:
                principled_bsdf.inputs['Emission'].default_value = (1.0, 1.0, 1.0, 1.0)  # Белая эмиссия
                if 'Emission Strength' in principled_bsdf.inputs:
                    principled_bsdf.inputs['Emission Strength'].default_value = 0.3  # Умеренная сила эмиссии
            else:
                # Если входа эмиссии нет, используем только базовый цвет
                principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Белый
                # Увеличиваем шероховатость для лучшей видимости
                if 'Roughness' in principled_bsdf.inputs:
                    principled_bsdf.inputs['Roughness'].default_value = 0.8
            
            # Соединяем узлы
            links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
            
            # Применяем материал
            if text_obj.data.materials:
                text_obj.data.materials[0] = text_material
            else:
                text_obj.data.materials.append(text_material)
            
            label_objects.append(text_obj)
            
        print(f"Создано {len(label_objects)} текстовых меток для junctions относительно стен")
        return label_objects
        
    except ImportError:
        print("Предупреждение: модуль bpy недоступен, создание текстовых меток невозможно")
        return []
    except Exception as e:
        print(f"Ошибка при создании текстовых меток: {e}")
        return []

def build_outline_from_wall_connections(junctions, wall_segments_from_openings, wall_segments_from_junctions, building_outline, scale_factor=1.0):
    """
    Строит внешний контур здания на основе привязки junctions к стенам, используя порядок из building_outline
    
    Args:
        junctions: список junctions
        wall_segments_from_openings: список сегментов стен от проемов
        wall_segments_from_junctions: список сегментов стен от junctions
        building_outline: контур здания с порядком обхода junctions
        scale_factor: масштабный коэффициент
    
    Returns:
        list: список вершин внешнего контура в порядке обхода
    """
    try:
        # Создаем словарь для быстрого поиска junctions по ID
        junction_dict = {j["id"]: j for j in junctions}
        
        # Создаем словарь для быстрого поиска сегментов стен по ID
        wall_segments = {}
        
        # Добавляем сегменты от проемов
        for segment in wall_segments_from_openings:
            wall_segments[segment["segment_id"]] = segment
        
        # Добавляем сегменты от junctions
        for segment in wall_segments_from_junctions:
            wall_segments[segment["segment_id"]] = segment
        
        # Получаем порядок junctions из building_outline
        outline_junction_ids = []
        for vertex in building_outline["vertices"]:
            if "junction_id" in vertex:
                outline_junction_ids.append(vertex["junction_id"])
        
        print(f"Порядок junctions из building_outline: {outline_junction_ids}")
        
        # Строим контур, следуя порядку junctions
        outline_vertices = []
        
        for junction_id in outline_junction_ids:
            if junction_id not in junction_dict:
                print(f"Предупреждение: Junction {junction_id} не найден в junctions")
                continue
                
            junction = junction_dict[junction_id]
            
            # Ищем связанные сегменты стен
            connected_segments = junction.get("connected_wall_segments", [])
            
            if not connected_segments:
                print(f"Предупреждение: Junction {junction_id} не имеет связанных стен")
                # Используем координаты junction как запасной вариант
                x = junction["x"] * scale_factor
                y = junction["y"] * scale_factor
                outline_vertices.append({
                    "x": x,
                    "y": y,
                    "junction_id": junction_id,
                    "junction_type": junction.get("junction_type", "unknown"),
                    "position_source": "junction_coordinates"
                })
                continue
            
            # Берем первый связанный сегмент для позиционирования
            first_segment_id = connected_segments[0]["segment_id"]
            
            if first_segment_id not in wall_segments:
                print(f"Предупреждение: Сегмент стены {first_segment_id} не найден для Junction {junction_id}")
                # Используем координаты junction как запасной вариант
                x = junction["x"] * scale_factor
                y = junction["y"] * scale_factor
                outline_vertices.append({
                    "x": x,
                    "y": y,
                    "junction_id": junction_id,
                    "junction_type": junction.get("junction_type", "unknown"),
                    "position_source": "junction_coordinates"
                })
                continue
            
            segment = wall_segments[first_segment_id]
            
            # Определяем позицию для метки на основе сегмента стены
            bbox = segment["bbox"]
            
            # Вычисляем центр сегмента стены
            center_x = (bbox["x"] + bbox["width"] / 2) * scale_factor
            center_y = (bbox["y"] + bbox["height"] / 2) * scale_factor
            
            # Если junction привязан к началу или концу сегмента, смещаем позицию
            connection_type = connected_segments[0].get("connection_type", "")
            if connection_type == "start":
                # Смещаем к началу сегмента
                offset_x = -bbox["width"] * scale_factor * 0.3
                offset_y = -bbox["height"] * scale_factor * 0.3
                center_x += offset_x
                center_y += offset_y
            elif connection_type == "end":
                # Смещаем к концу сегмента
                offset_x = bbox["width"] * scale_factor * 0.3
                offset_y = bbox["height"] * scale_factor * 0.3
                center_x += offset_x
                center_y += offset_y
            
            outline_vertices.append({
                "x": center_x,
                "y": center_y,
                "junction_id": junction_id,
                "junction_type": junction.get("junction_type", "unknown"),
                "position_source": "wall_segment",
                "segment_id": first_segment_id
            })
        
        print(f"Построен внешний контур из {len(outline_vertices)} вершин на основе стен")
        return outline_vertices
        
    except Exception as e:
        print(f"Ошибка при построении внешнего контура: {e}")
        return []

def mark_external_walls_by_junctions_and_walls(junctions, wall_segments_from_openings, wall_segments_from_junctions, building_outline):
    """
    Определяет внешние стены на основе порядка junctions из building_outline и их привязок к стенам
    
    Args:
        junctions: список junctions
        wall_segments_from_openings: сегменты стен от проемов
        wall_segments_from_junctions: сегменты стен от соединений
        building_outline: контур здания с порядком junctions
    
    Returns:
        set: множество ID внешних стен
    """
    external_wall_ids = set()
    
    # Создаем словарь для быстрого поиска junctions по ID
    junction_dict = {j["id"]: j for j in junctions}
    
    # Создаем словарь для быстрого поиска сегментов стен по ID
    wall_segments = {}
    
    # Добавляем сегменты от проемов
    for segment in wall_segments_from_openings:
        wall_segments[segment["segment_id"]] = segment
    
    # Добавляем сегменты от junctions
    for segment in wall_segments_from_junctions:
        wall_segments[segment["segment_id"]] = segment
    
    # Получаем порядок junctions из building_outline
    outline_junction_ids = []
    for vertex in building_outline["vertices"]:
        if "junction_id" in vertex:
            outline_junction_ids.append(vertex["junction_id"])
    
    print(f"Порядок junctions из building_outline: {outline_junction_ids}")
    
    # Создаем множество всех пар junction_id, которые образуют внешний контур
    outline_pairs = set()
    for i in range(len(outline_junction_ids)):
        current_junction_id = outline_junction_ids[i]
        next_junction_id = outline_junction_ids[(i + 1) % len(outline_junction_ids)]
        
        # Добавляем обе направления пары
        outline_pairs.add((current_junction_id, next_junction_id))
        outline_pairs.add((next_junction_id, current_junction_id))
    
    print(f"Пары junctions внешнего контура: {outline_pairs}")
    
    # Проверяем сегменты стен от проемов
    for segment in wall_segments_from_openings:
        start_junction_id = segment.get("start_junction_id")
        end_junction_id = segment.get("end_junction_id")
        
        if start_junction_id is not None and end_junction_id is not None:
            # Проверяем, образует ли сегмент часть внешнего контура
            if (start_junction_id, end_junction_id) in outline_pairs:
                external_wall_ids.add(segment["segment_id"])
                print(f"  Внешняя стена (проем): {segment['segment_id']} -> junctions {start_junction_id}-{end_junction_id}")
    
    # Проверяем сегменты стен от соединений
    for segment in wall_segments_from_junctions:
        start_junction_id = segment.get("start_junction_id")
        end_junction_id = segment.get("end_junction_id")
        
        if start_junction_id is not None and end_junction_id is not None:
            # Проверяем, образует ли сегмент часть внешнего контура
            if (start_junction_id, end_junction_id) in outline_pairs:
                external_wall_ids.add(segment["segment_id"])
                print(f"  Внешняя стена (соединение): {segment['segment_id']} -> junctions {start_junction_id}-{end_junction_id}")
    
    print(f"Всего найдено внешних стен: {len(external_wall_ids)}")
    return external_wall_ids

def is_external_wall(segment_data, external_wall_ids):
    """
    Определяет, является ли стена внешней на основе предварительно вычисленных ID
    
    Args:
        segment_data: данные сегмента стены
        external_wall_ids: множество ID внешних стен
    
    Returns:
        bool: True если стена внешняя, False если внутренняя
    """
    return segment_data["segment_id"] in external_wall_ids

def is_external_opening(opening_data, external_opening_ids):
    """
    Определяет, является ли проем внешним на основе предварительно вычисленных ID
    
    Args:
        opening_data: данные проема
        external_opening_ids: множество ID внешних проемов
    
    Returns:
        bool: True если проем внешний, False если внутренний
    """
    return opening_data["id"] in external_opening_ids

def load_brick_texture(texture_path="brick_texture.jpg"):
    """
    Загружает текстуру кирпича и создает материал для внешних стен
    
    Args:
        texture_path: путь к файлу текстуры
        
    Returns:
        bpy.types.Material: материал с текстурой кирпича
    """
    # Проверяем, существует ли текстура
    if not os.path.exists(texture_path):
        print(f"Предупреждение: Файл текстуры не найден: {texture_path}")
        print("Используем стандартный материал для внешних стен")
        # Создаем материал с цветом кирпича
        brick_mat = bpy.data.materials.new(name="ExternalWallMaterial")
        brick_mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)  # Серый цвет кирпича
        return brick_mat
    
    # Создаем новый материал
    brick_mat = bpy.data.materials.new(name="BrickWallMaterial")
    brick_mat.use_nodes = True
    
    # Получаем узлы материала
    nodes = brick_mat.node_tree.nodes
    links = brick_mat.node_tree.links
    
    # Очищаем стандартные узлы
    nodes.clear()
    
    # Создаем основные узлы
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Создаем узел для текстуры
    texture_node = nodes.new(type='ShaderNodeTexImage')
    
    # Загружаем изображение
    try:
        image = bpy.data.images.load(texture_path)
        texture_node.image = image
        print(f"Текстура кирпича успешно загружена: {texture_path}")
    except Exception as e:
        print(f"Ошибка при загрузке текстуры: {e}")
        print("Используем стандартный материал для внешних стен")
        brick_mat.use_nodes = False  # Отключаем ноды, если не удалось загрузить текстуру
        brick_mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)  # Серый цвет кирпича
        return brick_mat
    
    # Создаем узел для координат текстуры
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    
    # Создаем узел для масштабирования текстуры
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (1.0, 1.0, 1.0)  # Масштаб кирпича (без увеличения)
    
    # Соединяем узлы
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], texture_node.inputs['Vector'])
    links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
    
    return brick_mat

def create_procedural_brick_material():
    """
    Создает процедурный материал кирпича без использования файла текстуры
    """
    # Создаем новый материал
    brick_mat = bpy.data.materials.new(name="ProceduralBrickMaterial")
    brick_mat.use_nodes = True
    
    # Получаем узлы материала
    nodes = brick_mat.node_tree.nodes
    links = brick_mat.node_tree.links
    
    # Очищаем стандартные узлы
    nodes.clear()
    
    # Создаем основные узлы
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Создаем узел для процедурной текстуры кирпича
    brick_texture = nodes.new(type='ShaderNodeTexBrick')
    brick_texture.inputs['Scale'].default_value = 5.0
    brick_texture.inputs['Mortar Size'].default_value = 0.02
    brick_texture.inputs['Mortar Smooth'].default_value = 1.0
    brick_texture.inputs['Bias'].default_value = 0.0
    brick_texture.inputs['Brick Width'].default_value = 0.5
    brick_texture.inputs['Row Height'].default_value = 0.25
    
    # Создаем узлы для цветов кирпича и раствора
    brick_color = nodes.new(type='ShaderNodeRGB')
    brick_color.outputs['Color'].default_value = (0.6, 0.4, 0.2, 1.0)  # Цвет кирпича
    
    mortar_color = nodes.new(type='ShaderNodeRGB')
    mortar_color.outputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # Цвет раствора
    
    # Создаем узел для смешивания цветов
    mix_color = nodes.new(type='ShaderNodeMixRGB')
    mix_color.inputs['Fac'].default_value = 1.0  # Полное смешивание
    
    # Соединяем узлы
    links.new(brick_texture.outputs['Color'], mix_color.inputs['Fac'])
    links.new(brick_color.outputs['Color'], mix_color.inputs['Color1'])
    links.new(mortar_color.outputs['Color'], mix_color.inputs['Color2'])
    links.new(mix_color.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Настраиваем параметры материала для лучшего отображения
    principled_bsdf.inputs['Roughness'].default_value = 0.8  # Добавляем шероховатость
    principled_bsdf.inputs['Specular'].default_value = 0.2  # Уменьшаем блики
    
    print("Создана процедурная текстура кирпича")
    
    return brick_mat

def create_brick_texture_if_needed(texture_path="brick_texture.jpg"):
    """
    Создает простую процедурную текстуру кирпича, если она не существует
    
    Args:
        texture_path: путь к файлу текстуры
        
    Returns:
        str: путь к созданному или существующему файлу текстуры
    """
    # Если текстура уже существует, не создаем ее заново
    if os.path.exists(texture_path):
        print(f"Текстура кирпича уже существует: {texture_path}")
        return texture_path
    
    try:
        print(f"Создание текстуры кирпича: {texture_path}")
        
        # Создаем новое изображение
        width, height = 512, 512
        image = bpy.data.images.new("BrickTexture", width=width, height=height)
        
        # Создаем процедурную текстуру кирпича
        pixels = []
        brick_width = 8  # Ширина кирпича в пикселях
        brick_height = 4  # Высота кирпича в пикселях
        mortar_width = 1  # Ширина шва в пикселях
        
        for y in range(height):
            for x in range(width):
                # Определяем, находится ли пиксель на шве
                in_mortar_x = (x % (brick_width + mortar_width)) >= brick_width
                in_mortar_y = (y % (brick_height + mortar_width)) >= brick_height
                
                if in_mortar_x or in_mortar_y:
                    # Цвет шва (светло-серый)
                    r, g, b = 0.7, 0.7, 0.7
                else:
                    # Цвет кирпича (серый)
                    r, g, b = 0.5, 0.5, 0.5
                
                # Добавляем небольшую вариацию цвета для реалистичности
                variation = random.uniform(-0.05, 0.05)
                r = max(0, min(1, r + variation))
                g = max(0, min(1, g + variation))
                b = max(0, min(1, b + variation))
                
                pixels.extend([r, g, b, 1.0])  # RGBA
        
        # Устанавливаем пиксели
        image.pixels = pixels
        
        # Сохраняем изображение
        image.filepath_raw = texture_path
        image.file_format = 'JPEG'
        image.save()
        
        print(f"Текстура кирпича успешно создана: {texture_path}")
        return texture_path
        
    except Exception as e:
        print(f"Ошибка при создании текстуры кирпича: {e}")
        return None

def load_wall_coordinates(json_path):
    """
    Загружает данные о стенах из JSON файла
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке JSON файла: {e}")
        return None

def get_junction_by_id(junctions, junction_id):
    """
    Находит соединение по его ID
    """
    for junction in junctions:
        if junction["id"] == junction_id:
            return junction
    return None

def create_wall_mesh(segment_data, wall_height, wall_thickness, scale_factor=1.0, is_external=False, brick_material=None):
    """
    Создает 3D меш для сегмента стены на основе bbox данных
    
    Args:
        segment_data: данные сегмента стены
        wall_height: высота стены
        wall_thickness: толщина стены
        scale_factor: масштабный коэффициент
        is_external: является ли стена внешней
        brick_material: материал с текстурой кирпича для внешних стен
    """
    # Получаем bbox данные
    bbox = segment_data["bbox"]
    orientation = segment_data.get("orientation", "horizontal")
    
    # Получаем координаты с применением масштабирования
    x = bbox["x"] * scale_factor
    y = bbox["y"] * scale_factor
    width = bbox["width"] * scale_factor
    height = bbox["height"] * scale_factor
    
    # Создаем меш
    bm = bmesh.new()
    
    if orientation == "horizontal":
        # Горизонтальная стена (толщина по оси Y)
        # Вершины нижней грани
        v1 = bm.verts.new((x, y, 0))
        v2 = bm.verts.new((x + width, y, 0))
        v3 = bm.verts.new((x + width, y + height, 0))
        v4 = bm.verts.new((x, y + height, 0))
        
        # Вершины верхней грани
        v5 = bm.verts.new((x, y, wall_height))
        v6 = bm.verts.new((x + width, y, wall_height))
        v7 = bm.verts.new((x + width, y + height, wall_height))
        v8 = bm.verts.new((x, y + height, wall_height))
        
    else:  # vertical
        # Вертикальная стена
        # Вершины нижней грани
        v1 = bm.verts.new((x, y, 0))
        v2 = bm.verts.new((x + width, y, 0))
        v3 = bm.verts.new((x + width, y + height, 0))
        v4 = bm.verts.new((x, y + height, 0))
        
        # Вершины верхней грани
        v5 = bm.verts.new((x, y, wall_height))
        v6 = bm.verts.new((x + width, y, wall_height))
        v7 = bm.verts.new((x + width, y + height, wall_height))
        v8 = bm.verts.new((x, y + height, wall_height))
    
    # Создаем грани
    face_bottom = bm.faces.new((v1, v2, v3, v4))
    face_top = bm.faces.new((v5, v6, v7, v8))
    face_front = bm.faces.new((v1, v2, v6, v5))
    face_back = bm.faces.new((v4, v3, v7, v8))
    face_left = bm.faces.new((v1, v5, v8, v4))
    face_right = bm.faces.new((v2, v3, v7, v6))
    
    # Добавляем UV-развертку для текстур
    if is_external and brick_material:
        # Создаем UV-слой
        bm.loops.layers.uv.new()
        uv_layer = bm.loops.layers.uv.active
        
        # Для каждой грани создаем UV-координаты
        for face in bm.faces:
            # Определяем ориентацию грани для правильного наложения текстуры
            if orientation == "horizontal":
                if face.normal.z > 0.5:  # Верхняя грань
                    # UV для верхней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/2, 0)  # Увеличиваем масштаб для лучшей видимости
                    face.loops[2][uv_layer].uv = (width/2, height/2)
                    face.loops[3][uv_layer].uv = (0, height/2)
                elif face.normal.z < -0.5:  # Нижняя грань
                    # UV для нижней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/2, 0)
                    face.loops[2][uv_layer].uv = (width/2, height/2)
                    face.loops[3][uv_layer].uv = (0, height/2)
                elif abs(face.normal.x) > 0.5:  # Боковые грани по X (внешняя сторона)
                    # UV для боковых граней - ВНЕШНЯЯ СТОРОНА
                    # Определяем, какая сторона внешняя (направлена от центра здания)
                    # Для горизонтальных стен внешняя сторона обычно по оси Y
                    if face.normal.y > 0:  # Передняя сторона (возможно внешняя)
                        face.loops[0][uv_layer].uv = (0, 0)
                        face.loops[1][uv_layer].uv = (width/2, 0)
                        face.loops[2][uv_layer].uv = (width/2, wall_height/2)
                        face.loops[3][uv_layer].uv = (0, wall_height/2)
                    else:  # Задняя сторона (возможно внутренняя)
                        face.loops[0][uv_layer].uv = (0, 0)
                        face.loops[1][uv_layer].uv = (width/2, 0)
                        face.loops[2][uv_layer].uv = (width/2, wall_height/2)
                        face.loops[3][uv_layer].uv = (0, wall_height/2)
                else:  # Боковые грани по Y - НАИБОЛЕЕ ВЕРОЯТНАЯ ВНЕШНЯЯ СТОРОНА
                    # UV для боковых граней с улучшенным масштабом
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (height/2, 0)
                    face.loops[2][uv_layer].uv = (height/2, wall_height/2)
                    face.loops[3][uv_layer].uv = (0, wall_height/2)
            else:  # vertical
                if face.normal.z > 0.5:  # Верхняя грань
                    # UV для верхней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/2, 0)
                    face.loops[2][uv_layer].uv = (width/2, height/2)
                    face.loops[3][uv_layer].uv = (0, height/2)
                elif face.normal.z < -0.5:  # Нижняя грань
                    # UV для нижней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/2, 0)
                    face.loops[2][uv_layer].uv = (width/2, height/2)
                    face.loops[3][uv_layer].uv = (0, height/2)
                elif abs(face.normal.x) > 0.5:  # Боковые грани по X - НАИБОЛЕЕ ВЕРОЯТНАЯ ВНЕШНЯЯ СТОРОНА
                    # UV для боковых граней с улучшенным масштабом
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (height/2, 0)
                    face.loops[2][uv_layer].uv = (height/2, wall_height/2)
                    face.loops[3][uv_layer].uv = (0, wall_height/2)
                else:  # Боковые грани по Y
                    # UV для боковых граней
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/2, 0)
                    face.loops[2][uv_layer].uv = (width/2, wall_height/2)
                    face.loops[3][uv_layer].uv = (0, wall_height/2)
    
    # Обновляем нормали
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Создаем объект
    mesh = bpy.data.meshes.new(name=f"Wall_{segment_data['segment_id']}")
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Применяем материал в зависимости от типа стены
    if is_external:
        # Внешняя стена с желтым цветом для визуальной проверки
        yellow_mat = bpy.data.materials.new(name="ExternalWallMaterial")
        yellow_mat.diffuse_color = (1.0, 1.0, 0.0, 1.0)  # Ярко-желтый
        yellow_mat.use_nodes = True
        
        # Настраиваем материал для лучшей видимости
        nodes = yellow_mat.node_tree.nodes
        links = yellow_mat.node_tree.links
        
        # Очищаем стандартные узлы
        nodes.clear()
        
        # Создаем основные узлы
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Настраиваем эмиссию для лучшей видимости
        principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 0.0, 1.0)  # Ярко-желтый
        
        # Проверяем наличие входов эмиссии (зависит от версии Blender)
        if 'Emission' in principled_bsdf.inputs:
            principled_bsdf.inputs['Emission'].default_value = (1.0, 1.0, 0.0, 1.0)  # Желтая эмиссия
            if 'Emission Strength' in principled_bsdf.inputs:
                principled_bsdf.inputs['Emission Strength'].default_value = 0.5  # Умеренная сила эмиссии
        else:
            # Если входа эмиссии нет, используем только базовый цвет
            principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 0.0, 1.0)  # Ярко-желтый
            # Увеличиваем шероховатость для лучшей видимости
            if 'Roughness' in principled_bsdf.inputs:
                principled_bsdf.inputs['Roughness'].default_value = 0.8
        
        # Соединяем узлы
        links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
        
        if obj.data.materials:
            obj.data.materials[0] = yellow_mat
        else:
            obj.data.materials.append(yellow_mat)
            
        print(f"    ВНЕШНЯЯ СТЕНА (отмечена желтым): {segment_data['segment_id']}")
    else:
        # Внутренняя стена или стандартный материал
        mat = bpy.data.materials.new(name="InternalWallMaterial")
        mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)  # Светло-серый
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    
    return obj

def create_opening_mesh(opening_data, wall_height, wall_thickness, scale_factor=1.0, is_external=False):
    """
    Создает 3D меш для проема (окна или двери) используя точные координаты из JSON
    без расширения за счет толщины стен, чтобы проемы были вертикальными без углублений
    
    Args:
        opening_data: данные проема из JSON
        wall_height: высота стены
        wall_thickness: толщина стены
        scale_factor: масштабный коэффициент
        is_external: является ли проем внешним (для выбора материала)
    """
    bbox = opening_data["bbox"]
    orientation = opening_data.get("orientation", "horizontal")
    
    # Создаем меш для проема
    bm = bmesh.new()
    
    # Используем значения из JSON вместо вычисляемых пропорций
    if "bottom_height" in opening_data and "opening_height" in opening_data:
        opening_bottom = opening_data["bottom_height"]
        opening_height = opening_data["opening_height"]
    else:
        # Резервный вариант с вычисляемыми пропорциями
        # Определяем параметры проема
        if opening_data["type"] == "door":
            opening_height = 2.0  # Фиксированная высота двери - 2 метра
            opening_bottom = 0.1  # Небольшой зазор снизу
        else:  # window
            opening_height = wall_height * 0.6  # Высота окна - 60% от высоты стены
            opening_bottom = wall_height * 0.3  # Высота от пола до низа окна - 30% от высоты стены
    
    if orientation == "horizontal":
        # Горизонтальный проем (окно или дверь)
        x = bbox["x"] * scale_factor
        y = bbox["y"] * scale_factor
        width = bbox["width"] * scale_factor
        height = bbox["height"] * scale_factor
        
        # Создаем куб для проема без расширения
        v1 = bm.verts.new((x, y, opening_bottom))
        v2 = bm.verts.new((x + width, y, opening_bottom))
        v3 = bm.verts.new((x + width, y + height, opening_bottom))
        v4 = bm.verts.new((x, y + height, opening_bottom))
        
        v5 = bm.verts.new((x, y, opening_bottom + opening_height))
        v6 = bm.verts.new((x + width, y, opening_bottom + opening_height))
        v7 = bm.verts.new((x + width, y + height, opening_bottom + opening_height))
        v8 = bm.verts.new((x, y + height, opening_bottom + opening_height))
        
    else:  # vertical
        # Вертикальный проем
        x = bbox["x"] * scale_factor
        y = bbox["y"] * scale_factor
        width = bbox["width"] * scale_factor
        height = bbox["height"] * scale_factor
        
        # Создаем куб для проема без расширения
        v1 = bm.verts.new((x, y, opening_bottom))
        v2 = bm.verts.new((x + width, y, opening_bottom))
        v3 = bm.verts.new((x + width, y + height, opening_bottom))
        v4 = bm.verts.new((x, y + height, opening_bottom))
        
        v5 = bm.verts.new((x, y, opening_bottom + opening_height))
        v6 = bm.verts.new((x + width, y, opening_bottom + opening_height))
        v7 = bm.verts.new((x + width, y + height, opening_bottom + opening_height))
        v8 = bm.verts.new((x, y + height, opening_bottom + opening_height))
    
    # Создаем грани
    face_bottom = bm.faces.new((v1, v2, v3, v4))
    face_top = bm.faces.new((v5, v6, v7, v8))
    face_front = bm.faces.new((v1, v2, v6, v5))
    face_back = bm.faces.new((v4, v3, v7, v8))
    face_left = bm.faces.new((v1, v5, v8, v4))
    face_right = bm.faces.new((v2, v3, v7, v6))
    
    # Обновляем нормали
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Создаем объект
    mesh = bpy.data.meshes.new(name=f"Opening_{opening_data['id']}")
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Применяем материал в зависимости от типа проема и расположения
    if opening_data["type"] == "door":
        if is_external:
            # Тёмно-зелёный материал для внешних дверей
            mat = bpy.data.materials.new(name="ExternalDoorMaterial")
            mat.diffuse_color = (0.0, 0.6, 0.0, 1.0)  # Тёмно-зелёный
        else:
            # Зелёный материал для внутренних дверей
            mat = bpy.data.materials.new(name="InternalDoorMaterial")
            mat.diffuse_color = (0.0, 0.8, 0.0, 1.0)  # Зелёный
    else:  # window
        if is_external:
            # Тёмно-голубой материал для внешних окон
            mat = bpy.data.materials.new(name="ExternalWindowMaterial")
            mat.diffuse_color = (0.3, 0.5, 0.8, 1.0)  # Тёмно-голубой
        else:
            # Голубой материал для внутренних окон
            mat = bpy.data.materials.new(name="InternalWindowMaterial")
            mat.diffuse_color = (0.5, 0.7, 1.0, 1.0)  # Голубой
    
    # Применяем материал к объекту
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return obj

def create_pillar_mesh(pillar_data, wall_height, wall_thickness, scale_factor=1.0):
    """
    Создает 3D меш для колонны с коричневым материалом
    """
    bbox = pillar_data["bbox"]
    
    # Получаем координаты с применением масштабирования
    x = bbox["x"] * scale_factor
    y = bbox["y"] * scale_factor
    width = bbox["width"] * scale_factor
    height = bbox["height"] * scale_factor
    
    # Создаем меш
    bm = bmesh.new()
    
    # Вершины нижней грани
    v1 = bm.verts.new((x, y, 0))
    v2 = bm.verts.new((x + width, y, 0))
    v3 = bm.verts.new((x + width, y + height, 0))
    v4 = bm.verts.new((x, y + height, 0))
    
    # Вершины верхней грани
    v5 = bm.verts.new((x, y, wall_height))
    v6 = bm.verts.new((x + width, y, wall_height))
    v7 = bm.verts.new((x + width, y + height, wall_height))
    v8 = bm.verts.new((x, y + height, wall_height))
    
    # Создаем грани
    face_bottom = bm.faces.new((v1, v2, v3, v4))
    face_top = bm.faces.new((v5, v6, v7, v8))
    face_front = bm.faces.new((v1, v2, v6, v5))
    face_back = bm.faces.new((v4, v3, v7, v8))
    face_left = bm.faces.new((v1, v5, v8, v4))
    face_right = bm.faces.new((v2, v3, v7, v6))
    
    # Обновляем нормали
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Создаем объект
    mesh = bpy.data.meshes.new(name=f"Pillar_{pillar_data['id']}")
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Создаем коричневый материал для колонн
    mat = bpy.data.materials.new(name="PillarMaterial")
    mat.diffuse_color = (0.6, 0.3, 0.1, 1.0)  # Коричневый
    
    # Применяем материал к объекту
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return obj

def create_fill_for_opening(opening_data, wall_height, wall_thickness, scale_factor=1.0, wall_obj=None):
    """
    Создает 3D-заполнение для проема (окна/двери) - над и под проемом на всю ширину проема
    
    Args:
        opening_data: данные проема из JSON
        wall_height: высота стены в метрах
        wall_thickness: толщина стены в метрах
        scale_factor: масштабный коэффициент
        wall_obj: объект стены для интеграции через булевы операции
    
    Returns:
        list: список объектов заполнения (над и под проемом)
    """
    fill_objects = []
    
    # Получаем bbox данные проема
    bbox = opening_data["bbox"]
    orientation = opening_data.get("orientation", "horizontal")
    opening_id = opening_data.get("id", "unknown")
    
    print(f"    Создание заполнения для проема {opening_id} (ориентация: {orientation})")
    
    # Используем значения из JSON вместо вычисляемых пропорций
    if "bottom_height" in opening_data and "opening_height" in opening_data:
        opening_bottom = opening_data["bottom_height"]
        opening_height = opening_data["opening_height"]
        print(f"    Параметры из JSON: высота={opening_height:.2f}м, низ={opening_bottom:.2f}м")
    else:
        # Резервный вариант с вычисляемыми пропорциями
        print(f"    Предупреждение: отсутствуют параметры в JSON для проема {opening_id}, используем стандартные значения")
        
        # Определяем параметры проема в зависимости от типа
        if opening_data["type"] == "door":
            opening_height = 2.0  # Фиксированная высота двери - 2 метра
            opening_bottom = 0.1  # Небольшой зазор снизу
            print(f"    Параметры двери: высота={opening_height}м, низ={opening_bottom}м")
        else:  # window
            opening_height = wall_height * 0.6  # Высота окна - 60% от высоты стены
            opening_bottom = wall_height * 0.3  # Высота от пола до низа окна - 30% от высоты стены
            print(f"    Параметры окна: высота={opening_height:.2f}м, низ={opening_bottom:.2f}м")
    
    # Получаем координаты с применением масштабирования
    x = bbox["x"] * scale_factor
    y = bbox["y"] * scale_factor
    width = bbox["width"] * scale_factor
    height = bbox["height"] * scale_factor
    
    print(f"    Координаты проема: x={x:.3f}, y={y:.3f}, ширина={width:.3f}, высота={height:.3f}")
    
    # Определяем реальные размеры заполнения на основе данных из JSON
    # Используем полную ширину проема из JSON
    if orientation == "horizontal":
        fill_width = width  # Полная ширина проема
        fill_depth = wall_thickness  # Толщина стены
        fill_x = x  # Начальная позиция X
        fill_y = y  # Начальная позиция Y
        print(f"    Горизонтальный проем: заполнение шириной={fill_width:.3f}, глубиной={fill_depth:.3f}")
    else:  # vertical
        fill_width = wall_thickness  # Толщина стены
        fill_depth = height  # Полная высота проема в плане
        fill_x = x  # Начальная позиция X
        fill_y = y  # Начальная позиция Y
        print(f"    Вертикальный проем: заполнение шириной={fill_width:.3f}, глубиной={fill_depth:.3f}")
    
    # Шаг 1: Создаем заполнение ОТ 0 ДО НИЖНЕГО КРАЯ проема
    if opening_bottom > 0:
        fill_below_height = opening_bottom
        fill_below_z = fill_below_height / 2
        
        print(f"    Создание заполнения под проемом: высота={fill_below_height:.3f}м")
        
        # Создаем куб для заполнения под проемом
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        fill_below_obj = bpy.context.active_object
        fill_below_obj.name = f"Fill_Below_{opening_id}"
        
        # Масштабируем куб на полную ширину проема
        # Используем полный размер без деления на 2
        scale_x = fill_width
        scale_y = fill_depth
        scale_z = fill_below_height
        fill_below_obj.scale = (scale_x, scale_y, scale_z)
        
        # Позиционируем центр куба
        if orientation == "horizontal":
            fill_below_obj.location = (fill_x + fill_width/2.0, fill_y + fill_depth/2.0, fill_below_z)
        else:  # vertical
            fill_below_obj.location = (fill_x + fill_width/2.0, fill_y + fill_depth/2.0, fill_below_z)
        
        # Добавляем UV-развертку для текстуры
        bm = bmesh.new()
        bm.from_mesh(fill_below_obj.data)
        
        # Создаем UV-слой
        bm.loops.layers.uv.new()
        uv_layer = bm.loops.layers.uv.active
        
        # Для каждой грани создаем UV-координаты
        for face in bm.faces:
            # Определяем ориентацию грани для правильного наложения текстуры
            if face.normal.z > 0.5:  # Верхняя грань
                # UV для верхней грани
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_width/10, 0)  # Масштаб 10 пикселей = 1 единица UV
                face.loops[2][uv_layer].uv = (fill_width/10, fill_depth/10)
                face.loops[3][uv_layer].uv = (0, fill_depth/10)
            elif face.normal.z < -0.5:  # Нижняя грань
                # UV для нижней грани
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_width/10, 0)
                face.loops[2][uv_layer].uv = (fill_width/10, fill_depth/10)
                face.loops[3][uv_layer].uv = (0, fill_depth/10)
            elif abs(face.normal.x) > 0.5:  # Боковые грани по X
                # UV для боковых граней
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_width/10, 0)
                face.loops[2][uv_layer].uv = (fill_width/10, fill_below_height/10)
                face.loops[3][uv_layer].uv = (0, fill_below_height/10)
            else:  # Боковые грани по Y
                # UV для боковых граней
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_depth/10, 0)
                face.loops[2][uv_layer].uv = (fill_depth/10, fill_below_height/10)
                face.loops[3][uv_layer].uv = (0, fill_below_height/10)
        
        # Обновляем нормали
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        
        # Применяем изменения к мешу
        bm.to_mesh(fill_below_obj.data)
        bm.free()
        
        # Применяем материал как у соседних стен с полным копированием настроек
        if wall_obj and wall_obj.data.materials:
            wall_material = wall_obj.data.materials[0]
            
            # Создаем копию материала для заглушки
            fill_material = wall_material.copy()
            fill_material.name = f"FillMaterial_{opening_id}_Below"
            
            # Если это материал с текстурой кирпича, копируем узлы текстуры
            if wall_material.use_nodes and "BrickWallMaterial" in wall_material.name:
                fill_material.use_nodes = True
                wall_nodes = wall_material.node_tree.nodes
                fill_nodes = fill_material.node_tree.nodes
                
                # Находим узел текстуры в стене
                texture_node = None
                for node in wall_nodes:
                    if node.type == 'TEX_IMAGE':
                        texture_node = node
                        break
                
                if texture_node:
                    # Находим соответствующий узел в заглушке
                    fill_texture_node = None
                    for node in fill_nodes:
                        if node.type == 'TEX_IMAGE':
                            fill_texture_node = node
                            break
                    
                    if fill_texture_node:
                        # Копируем изображение текстуры
                        fill_texture_node.image = texture_node.image
                        
                        # Находим узел масштабирования и копируем настройки
                        for node in wall_nodes:
                            if node.type == 'MAPPING':
                                for fill_node in fill_nodes:
                                    if fill_node.type == 'MAPPING':
                                        # Копируем настройки масштабирования
                                        fill_node.inputs['Scale'].default_value = node.inputs['Scale'].default_value
                                        break
                                break
            
            # Применяем скопированный материал
            fill_below_obj.data.materials.append(fill_material)
        else:
            # Стандартный материал стен
            mat = bpy.data.materials.new(name="FillMaterial")
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)  # Светло-серый
            fill_below_obj.data.materials.append(mat)
        
        fill_objects.append(fill_below_obj)
        print(f"    Создан объект заполнения под проемом: {fill_below_obj.name}")
    else:
        print(f"    Пропуск заполнения под проемом: opening_bottom={opening_bottom} <= 0")
    
    # Шаг 2: Создаем заполнение ОТ ВЕРХНЕГО КРАЯ проема ДО ПОТОЛКА
    if opening_bottom + opening_height < wall_height:
        fill_above_height = wall_height - (opening_bottom + opening_height)
        fill_above_z = opening_bottom + opening_height + fill_above_height / 2
        
        print(f"    Создание заполнения над проемом: высота={fill_above_height:.3f}м")
        
        # Создаем куб для заполнения над проемом
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        fill_above_obj = bpy.context.active_object
        fill_above_obj.name = f"Fill_Above_{opening_id}"
        
        # Масштабируем куб на полную ширину проема
        # Используем полный размер без деления на 2
        scale_x = fill_width
        scale_y = fill_depth
        scale_z = fill_above_height
        fill_above_obj.scale = (scale_x, scale_y, scale_z)
        
        # Позиционируем центр куба
        if orientation == "horizontal":
            fill_above_obj.location = (fill_x + fill_width/2.0, fill_y + fill_depth/2.0, fill_above_z)
        else:  # vertical
            fill_above_obj.location = (fill_x + fill_width/2.0, fill_y + fill_depth/2.0, fill_above_z)
        
        # Добавляем UV-развертку для текстуры
        bm = bmesh.new()
        bm.from_mesh(fill_above_obj.data)
        
        # Создаем UV-слой
        bm.loops.layers.uv.new()
        uv_layer = bm.loops.layers.uv.active
        
        # Для каждой грани создаем UV-координаты
        for face in bm.faces:
            # Определяем ориентацию грани для правильного наложения текстуры
            if face.normal.z > 0.5:  # Верхняя грань
                # UV для верхней грани
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_width/10, 0)  # Масштаб 10 пикселей = 1 единица UV
                face.loops[2][uv_layer].uv = (fill_width/10, fill_depth/10)
                face.loops[3][uv_layer].uv = (0, fill_depth/10)
            elif face.normal.z < -0.5:  # Нижняя грань
                # UV для нижней грани
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_width/10, 0)
                face.loops[2][uv_layer].uv = (fill_width/10, fill_depth/10)
                face.loops[3][uv_layer].uv = (0, fill_depth/10)
            elif abs(face.normal.x) > 0.5:  # Боковые грани по X
                # UV для боковых граней
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_width/10, 0)
                face.loops[2][uv_layer].uv = (fill_width/10, fill_above_height/10)
                face.loops[3][uv_layer].uv = (0, fill_above_height/10)
            else:  # Боковые грани по Y
                # UV для боковых граней
                face.loops[0][uv_layer].uv = (0, 0)
                face.loops[1][uv_layer].uv = (fill_depth/10, 0)
                face.loops[2][uv_layer].uv = (fill_depth/10, fill_above_height/10)
                face.loops[3][uv_layer].uv = (0, fill_above_height/10)
        
        # Обновляем нормали
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        
        # Применяем изменения к мешу
        bm.to_mesh(fill_above_obj.data)
        bm.free()
        
        # Применяем материал как у соседних стен с полным копированием настроек
        if wall_obj and wall_obj.data.materials:
            wall_material = wall_obj.data.materials[0]
            
            # Создаем копию материала для заглушки
            fill_material = wall_material.copy()
            fill_material.name = f"FillMaterial_{opening_id}_Above"
            
            # Если это материал с текстурой кирпича, копируем узлы текстуры
            if wall_material.use_nodes and "BrickWallMaterial" in wall_material.name:
                fill_material.use_nodes = True
                wall_nodes = wall_material.node_tree.nodes
                fill_nodes = fill_material.node_tree.nodes
                
                # Находим узел текстуры в стене
                texture_node = None
                for node in wall_nodes:
                    if node.type == 'TEX_IMAGE':
                        texture_node = node
                        break
                
                if texture_node:
                    # Находим соответствующий узел в заглушке
                    fill_texture_node = None
                    for node in fill_nodes:
                        if node.type == 'TEX_IMAGE':
                            fill_texture_node = node
                            break
                    
                    if fill_texture_node:
                        # Копируем изображение текстуры
                        fill_texture_node.image = texture_node.image
                        
                        # Находим узел масштабирования и копируем настройки
                        for node in wall_nodes:
                            if node.type == 'MAPPING':
                                for fill_node in fill_nodes:
                                    if fill_node.type == 'MAPPING':
                                        # Копируем настройки масштабирования
                                        fill_node.inputs['Scale'].default_value = node.inputs['Scale'].default_value
                                        break
                                break
            
            # Применяем скопированный материал
            fill_above_obj.data.materials.append(fill_material)
        else:
            # Стандартный материал стен
            mat = bpy.data.materials.new(name="FillMaterial")
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)  # Светло-серый
            fill_above_obj.data.materials.append(mat)
        
        fill_objects.append(fill_above_obj)
        print(f"    Создан объект заполнения над проемом: {fill_above_obj.name}")
    else:
        print(f"    Пропуск заполнения над проемом: opening_bottom+opening_height={opening_bottom+opening_height:.3f} >= wall_height={wall_height}")
    
    # Отключаем интеграцию с существующей стеной через булевы операции
    # Оставляем заглушки как отдельные объекты с красным материалом
    if wall_obj and fill_objects:
        print(f"    Создано {len(fill_objects)} отдельных элементов заполнения для проема {opening_id}")
    elif not wall_obj:
        print(f"    Предупреждение: отсутствует стена для проема {opening_id}")
    
    return fill_objects

def create_fill_walls_between_openings(wall_segments, openings, wall_thickness, scale_factor, wall_height, wall_objects=None):
    """
    Создает заполнение для всех проемов - над и под каждым проемом
    
    Args:
        wall_segments: список сегментов стен от проемов
        openings: список проемов
        wall_thickness: толщина стен
        scale_factor: масштабный коэффициент
        wall_height: высота стен
        wall_objects: список объектов стен для интеграции
    
    Returns:
        list: список всех объектов заполнения
    """
    all_fill_objects = []
    
    print(f"Начинаем создание заполнения для {len(openings)} проемов...")
    
    # Находим соответствующие стены для каждого проема
    wall_map = {}
    if wall_objects:
        for wall_obj in wall_objects:
            wall_map[wall_obj.name] = wall_obj
    
    # Создаем заполнение для каждого проема
    for i, opening in enumerate(openings):
        opening_id = opening.get("id", f"unknown_{i}")
        opening_type = opening.get("type", "unknown")
        opening_orientation = opening.get("orientation", "horizontal")
        
        print(f"Обработка проема {i+1}/{len(openings)}: {opening_id} (тип: {opening_type}, ориентация: {opening_orientation})")
        
        # Ищем соответствующую стену для проема
        related_wall = None
        
        if wall_objects:
            # Простая эвристика: находим стену, которая пересекается с проемом
            # Создаем временный объект проема для проверки пересечения
            temp_opening = create_opening_mesh(opening, wall_height, wall_thickness, scale_factor)
            if temp_opening:
                for wall_obj in wall_objects:
                    if check_wall_opening_intersection(wall_obj, temp_opening):
                        related_wall = wall_obj
                        print(f"  Найдена соответствующая стена: {wall_obj.name}")
                        break
                
                if not related_wall:
                    print(f"  Предупреждение: не найдена соответствующая стена для проема {opening_id}")
                
                # Удаляем временный объект проема
                bpy.data.objects.remove(temp_opening, do_unlink=True)
        else:
            print(f"  Предупреждение: отсутствуют объекты стен для проема {opening_id}")
        
        # Используем ориентацию проема непосредственно из данных JSON
        # Это гарантирует правильное определение ориентации для заполнения
        print(f"  Используем ориентацию проема из JSON: {opening_orientation}")
        
        # Создаем заполнение для проема с ориентацией из данных проема
        # Копируем данные проема, чтобы не изменять оригинал
        opening_copy = opening.copy()
        opening_copy["orientation"] = opening_orientation
        
        fill_objects = create_fill_for_opening(
            opening_copy,
            wall_height,
            wall_thickness,
            scale_factor,
            related_wall
        )
        
        if fill_objects:
            print(f"  Создано {len(fill_objects)} элементов заполнения для проема {opening_id}")
        else:
            print(f"  Предупреждение: не создано элементов заполнения для проема {opening_id}")
        
        all_fill_objects.extend(fill_objects)
    
    print(f"Завершено создание заполнения. Всего создано {len(all_fill_objects)} элементов для {len(openings)} проемов.")
    return all_fill_objects

def apply_boolean_difference(wall_obj, opening_obj):
    """
    Применяет булеву операцию разности к стене и проему
    """
    try:
        # Выделяем стену
        bpy.context.view_layer.objects.active = wall_obj
        wall_obj.select_set(True)
        opening_obj.select_set(False)
        
        # Добавляем модификатор Boolean
        bool_mod = wall_obj.modifiers.new(name="Boolean", type='BOOLEAN')
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.object = opening_obj
        bool_mod.solver = 'FAST'  # Используем быстрый решатель
        
        # Применяем модификатор
        bpy.context.view_layer.objects.active = wall_obj
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
        
        # Обновляем геометрию стены после вырезания проема
        wall_obj.data.update()
        wall_obj.data.update_tag()
        bpy.context.view_layer.update()
        
        # Удаляем объект проема
        bpy.data.objects.remove(opening_obj, do_unlink=True)
        
        return True
    except Exception as e:
        print(f"Ошибка при применении булевой операции: {e}")
        return False

def check_wall_opening_intersection(wall_obj, opening_obj):
    """
    Проверяет, пересекается ли стена с проемом по ограничивающей рамке
    """
    try:
        # Получаем ограничивающие рамки
        wall_bounds = [tuple(coord) for coord in wall_obj.bound_box]
        opening_bounds = [tuple(coord) for coord in opening_obj.bound_box]
        
        # Находим минимальные и максимальные координаты
        wall_min_x = min(coord[0] for coord in wall_bounds)
        wall_max_x = max(coord[0] for coord in wall_bounds)
        wall_min_y = min(coord[1] for coord in wall_bounds)
        wall_max_y = max(coord[1] for coord in wall_bounds)
        
        opening_min_x = min(coord[0] for coord in opening_bounds)
        opening_max_x = max(coord[0] for coord in opening_bounds)
        opening_min_y = min(coord[1] for coord in opening_bounds)
        opening_max_y = max(coord[1] for coord in opening_bounds)
        
        # Проверяем пересечение по осям X и Y
        x_intersect = (wall_min_x <= opening_max_x and wall_max_x >= opening_min_x)
        y_intersect = (wall_min_y <= opening_max_y and wall_max_y >= opening_min_y)
        
        return x_intersect and y_intersect
    except:
        return False

def merge_wall_objects(wall_objects):
    """
    Объединяет все стены в один объект
    """
    if not wall_objects:
        return None
    
    # Выделяем все стены
    bpy.ops.object.select_all(action='DESELECT')
    for obj in wall_objects:
        obj.select_set(True)
    
    # Активируем первую стену
    bpy.context.view_layer.objects.active = wall_objects[0]
    
    # Объединяем
    bpy.ops.object.join()
    
    return bpy.context.active_object

def invert_x_coordinates(obj, center_x=None):
    """
    Инвертирует X-координаты объекта относительно центра
    """
    if center_x is None:
        # Находим центр X-координат объекта
        min_x = min(v.co.x for v in obj.data.vertices)
        max_x = max(v.co.x for v in obj.data.vertices)
        center_x = (min_x + max_x) / 2
    
    # Инвертируем X-координаты относительно центра
    for v in obj.data.vertices:
        v.co.x = 2 * center_x - v.co.x

def setup_isometric_camera_and_render(output_path="isometric_view.jpg"):
    """
    Настраивает камеру для изометрического вида и выполняет рендер
    
    Args:
        output_path: путь для сохранения изображения
    """
    try:
        # Находим центр и границы всех объектов
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        
        # Проходим по всем меш-объектам
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                for vertex in obj.data.vertices:
                    # Преобразуем координаты вершины в мировые
                    world_coord = obj.matrix_world @ vertex.co
                    min_x = min(min_x, world_coord.x)
                    max_x = max(max_x, world_coord.x)
                    min_y = min(min_y, world_coord.y)
                    max_y = max(max_y, world_coord.y)
                    min_z = min(min_z, world_coord.z)
                    max_z = max(max_z, world_coord.z)
        
        # Вычисляем центр сцены
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        # Вычисляем размеры сцены
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        
        # Определяем расстояние до камеры (чтобы все объекты помещались в кадр)
        max_dimension = max(size_x, size_y, size_z)
        camera_distance = max_dimension * 2.0  # Расстояние до камеры
        
        # Создаем камеру
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "IsometricCamera"
        
        # Устанавливаем камеру как активную
        bpy.context.scene.camera = camera
        
        # Настраиваем изометрический ракурс
        # Позиция камеры для изометрического вида (сверху-сбоку, центрируем по сцене)
        # Используем диагональное позиционирование для классического изометрического вида
        offset_x = camera_distance * 0.7071  # cos(45°)
        offset_y = -camera_distance * 0.7071  # -sin(45°)
        offset_z = camera_distance * 0.5774  # 1/sqrt(3) для правильной высоты
        
        camera.location = (center_x + offset_x, center_y + offset_y, center_z + offset_z)
        
        # Направляем камеру на центр сцены
        # Создаем матрицу направления от камеры к центру
        direction = Vector((center_x, center_y, center_z)) - camera.location
        direction.normalize()
        
        # Используем track_to_constraint для надежного наведения камеры на центр
        # Создаем constraint
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = None  # Будем использовать пустой объект
        
        # Создаем пустой объект в центре сцены для наведения
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(center_x, center_y, center_z))
        empty = bpy.context.active_object
        empty.name = "CameraTarget"
        
        # Устанавливаем пустой объект как цель для камеры
        constraint.target = empty
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        # Устанавливаем тип камеры на ортографическую для лучшего изометрического вида
        camera.data.type = 'ORTHO'
        # Увеличиваем масштаб, чтобы гарантированно захватить всю сцену
        camera.data.ortho_scale = max(size_x, size_y) * 1.5
        
        # Настраиваем параметры рендера
        scene = bpy.context.scene
        
        # Устанавливаем движок рендера (EEVEE_NEXT работает быстрее в фоновом режиме)
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        
        # Настраиваем параметры EEVEE_NEXT для качественного отображения текстур
        scene.eevee.taa_render_samples = 128  # Увеличиваем количество сэмплов для лучшего качества
        
        # Настраиваем параметры теней для лучшего отображения текстур
        try:
            # Включаем тени
            scene.eevee.use_shadows = True
            
            # Устанавливаем качество теней
            if hasattr(scene.eevee, 'shadow_cube_size'):
                scene.eevee.shadow_cube_size = '2048'  # Увеличиваем качество теней
            if hasattr(scene.eevee, 'shadow_cascade_size'):
                scene.eevee.shadow_cascade_size = '2048'  # Увеличиваем качество каскадных теней
            
            # Настраиваем параметры освещения для текстур
            if hasattr(scene.eevee, 'shadow_softness'):
                scene.eevee.shadow_softness = 0.2  # Мягкие тени
            
            # Включаем экранное пространственное затенение (SSAO) для лучшего восприятия текстур
            if hasattr(scene.eevee, 'use_ssr'):
                scene.eevee.use_ssr = True  # Включаем отражения
            if hasattr(scene.eevee, 'use_sao'):
                scene.eevee.use_sao = True  # Включаем затенение окружения
                
            # Настраиваем параметрыAmbient Occlusion для лучшего восприятия текстур
            if hasattr(scene.eevee, 'gtao_distance'):
                scene.eevee.gtao_distance = 0.2  # Расстояние для затенения
            if hasattr(scene.eevee, 'gtao_thickness'):
                scene.eevee.gtao_thickness = 0.1  # Толщина для затенения
                
        except Exception as e:
            print(f"Предупреждение: Не удалось установить все параметры рендера EEVEE: {e}")
            print("Используем базовые параметры")
        
        # Настраиваем параметры файла
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = 90
        scene.render.filepath = output_path
        scene.render.image_settings.color_depth = '8'
        
        # Устанавливаем разрешение
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.resolution_percentage = 100
        
        # Убеждаемся, что путь к файлу абсолютный
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
            scene.render.filepath = output_path
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Настраиваем освещение для EEVEE_NEXT с улучшенным отображением текстур
        # Удаляем существующее освещение
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()
        
        # Добавляем основной солнечный свет для яркого освещения текстур
        bpy.ops.object.light_add(type='SUN')
        sun_light = bpy.context.active_object
        sun_light.name = "SunLight"
        sun_light.location = (5, 5, 10)
        sun_light.rotation_euler = (0.5, -0.5, 0.5)
        sun_light.data.energy = 5.0  # Увеличиваем энергию для лучшей видимости текстур
        sun_light.data.angle = 0.1  # Уменьшаем угол для более четких теней
        
        # Добавляем большой заполняющий свет с мягкими тенями
        bpy.ops.object.light_add(type='AREA')
        fill_light = bpy.context.active_object
        fill_light.name = "FillLight"
        fill_light.location = (0, 0, 8)
        fill_light.rotation_euler = (0, 0, 0)
        fill_light.data.energy = 3.0  # Увеличиваем энергию
        fill_light.data.size = 20.0  # Увеличиваем размер для более мягкого света
        
        # Добавляем свет сбоку для подсветки текстур на боковых поверхностях
        bpy.ops.object.light_add(type='AREA')
        side_light = bpy.context.active_object
        side_light.name = "SideLight"
        side_light.location = (-10, 0, 5)
        side_light.rotation_euler = (0, 1.57, 0)  # Поворот на 90 градусов
        side_light.data.energy = 2.5  # Увеличиваем энергию для подсветки боковых поверхностей
        side_light.data.size = 15.0  # Большой размер для мягкости
        
        # Добавляем второй свет сбоку с другой стороны
        bpy.ops.object.light_add(type='AREA')
        side_light2 = bpy.context.active_object
        side_light2.name = "SideLight2"
        side_light2.location = (10, 0, 5)
        side_light2.rotation_euler = (0, -1.57, 0)  # Поворот на -90 градусов
        side_light2.data.energy = 2.5  # Увеличиваем энергию для подсветки боковых поверхностей
        side_light2.data.size = 15.0  # Большой размер для мягкости
        
        # Добавляем свет спереди для подсветки передних поверхностей
        bpy.ops.object.light_add(type='AREA')
        front_light = bpy.context.active_object
        front_light.name = "FrontLight"
        front_light.location = (0, -10, 5)
        front_light.rotation_euler = (1.57, 0, 0)  # Направлен вперед
        front_light.data.energy = 2.0  # Энергия для подсветки передних поверхностей
        front_light.data.size = 15.0  # Большой размер для мягкости
        
        # Выделяем все объекты для рендера
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type in ('MESH', 'CAMERA', 'LIGHT'):
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
        
        # Выполняем рендер с дополнительными параметрами для фонового режима
        print(f"Выполняем рендер изометрического вида: {output_path}")
        
        # Принудительно обновляем сцену (в новой версии Blender метод update() устарел)
        # Используем dg.update() вместо scene.update()
        try:
            bpy.context.view_layer.update()
        except:
            # Если и это не сработает, просто продолжаем
            pass
        
        # Выполняем рендер
        result = bpy.ops.render.render(write_still=True)
        
        # Проверяем, был ли создан файл
        if os.path.exists(output_path):
            print(f"Изометрический вид сохранен: {output_path}")
            print(f"Размер файла: {os.path.getsize(output_path)} байт")
            return True
        else:
            print(f"Ошибка: файл не был создан: {output_path}")
            # Пробуем альтернативный путь
            alt_path = os.path.splitext(output_path)[0] + "_alt.jpg"
            scene.render.filepath = alt_path
            bpy.ops.render.render(write_still=True)
            if os.path.exists(alt_path):
                print(f"Изометрический вид сохранен по альтернативному пути: {alt_path}")
                return True
            return False
        
    except Exception as e:
        print(f"Ошибка при создании изометрического вида: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_3d_walls_from_json(json_path, wall_height=2.0, export_obj=True, clear_scene=True, brick_texture_path=None, render_isometric=False, show_junction_labels=True):
    """
    Основная функция для создания 3D стен из JSON файла
    
    Args:
        json_path (str): Путь к JSON файлу с координатами стен
        wall_height (float): Высота стен в метрах (по умолчанию 2.0)
        export_obj (bool): Экспортировать результат в OBJ файл (по умолчанию True)
        clear_scene (bool): Очищать сцену перед созданием стен (по умолчанию True)
        brick_texture_path (str): Путь к файлу текстуры кирпича для внешних стен
        render_isometric (bool): Создавать изометрический рендер (по умолчанию False)
        show_junction_labels (bool): Отображать номера junctions над стенами (по умолчанию True)
    
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    try:
        print("=" * 60)
        print("СОЗДАНИЕ 3D СТЕН ВЫСОТОЙ 1.33 МЕТРА")
        print("=" * 60)
        
        # Параметры
        wall_height_meters = wall_height
        scale_factor = 0.01  # 1 пиксель = 0.01 метра
        invert_x = False  # Отключаем инверсию X-координаты (координаты уже инвертированы в JSON)
        
        # Устанавливаем единицы измерения в метры
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.scale_length = 1.0  # 1 BU = 1m
        
        # Очищаем сцену если нужно
        if clear_scene:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
        
        # Загружаем данные
        print(f"Загрузка данных из {json_path}...")
        data = load_wall_coordinates(json_path)
        if not data:
            print("Ошибка: не удалось загрузить данные из JSON файла")
            return False
        
        # Получаем параметры
        wall_thickness = data["metadata"]["wall_thickness"] * scale_factor
        junctions = data["junctions"]
        wall_segments_from_openings = data["wall_segments_from_openings"]
        wall_segments_from_junctions = data["wall_segments_from_junctions"]
        openings = data["openings"]
        pillars = data.get("pillar_squares", [])
        
        print(f"Параметры: толщина стен={wall_thickness}м, высота стен={wall_height_meters}м")
        print(f"Масштабный коэффициент: {scale_factor}")
        print(f"Сегментов стен от проемов: {len(wall_segments_from_openings)}")
        print(f"Сегментов стен от соединений: {len(wall_segments_from_junctions)}")
        print(f"Проемов (окон и дверей): {len(openings)}")
        print(f"Колонн: {len(pillars)}")
        
        # Загружаем текстуру кирпича если указан путь
        brick_material = None
        if brick_texture_path:
            # Если текстура не существует, пытаемся создать ее
            if not os.path.exists(brick_texture_path):
                print(f"Текстура не найдена, пытаемся создать: {brick_texture_path}")
                create_brick_texture_if_needed(brick_texture_path)
            
            print(f"Загрузка текстуры кирпича: {brick_texture_path}")
            brick_material = load_brick_texture(brick_texture_path)
        else:
            # Попробуем найти текстуру в той же директории, что и скрипт
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_brick_path = os.path.join(script_dir, "brick_texture.jpg")
            
            # Если текстура не существует, создаем ее
            if not os.path.exists(default_brick_path):
                print(f"Текстура не найдена, создаем автоматически: {default_brick_path}")
                create_brick_texture_if_needed(default_brick_path)
            
            if os.path.exists(default_brick_path):
                print(f"Используем текстуру кирпича: {default_brick_path}")
                brick_material = load_brick_texture(default_brick_path)
            else:
                print("Текстура кирпича не найдена, внешние стены будут иметь стандартный материал")
        
        # Проверяем наличие контура здания в данных
        if "building_outline" not in data:
            print("Ошибка: в данных отсутствует контур здания (building_outline)")
            return False
        
        building_outline = data["building_outline"]
        outline_vertices = building_outline["vertices"]
        print(f"Загружен контур здания с {len(outline_vertices)} вершинами")
        
        # Строим внешний контур на основе привязки junctions к стенам
        print("Построение внешнего контура на основе привязки junctions к стенам...")
        wall_based_outline = build_outline_from_wall_connections(
            junctions,
            wall_segments_from_openings,
            wall_segments_from_junctions,
            building_outline,
            scale_factor
        )
        
        # Определяем внешние стены на основе порядка junctions и их привязок к стенам
        print("Определение внешних стен на основе порядка junctions и их привязок к стенам...")
        external_wall_ids = mark_external_walls_by_junctions_and_walls(
            junctions,
            wall_segments_from_openings,
            wall_segments_from_junctions,
            building_outline
        )
        print(f"  Найдено внешних стен: {len(external_wall_ids)}")
        
        # Дополнительно определяем внешние проемы на основе контура (для совместимости)
        print("Определение внешних проемов по контуру...")
        tolerance_multiplier = 2.0  # Увеличиваем tolerance в 2 раза
        _, external_opening_ids = mark_external_elements_by_outline(
            wall_segments_from_openings,
            wall_segments_from_junctions,
            openings,
            outline_vertices,
            wall_thickness * tolerance_multiplier,  # Увеличиваем tolerance
            scale_factor
        )
        print(f"  Найдено внешних проемов: {len(external_opening_ids)}")
        
        # Создаем стены из сегментов от проемов
        wall_objects = []
        external_walls_count = 0
        internal_walls_count = 0
        print("Создание стен...")
        for i, segment in enumerate(wall_segments_from_openings):
            # Проверяем, является ли стена внешней
            is_ext = is_external_wall(segment, external_wall_ids)
            
            # Выводим отладочную информацию
            if is_ext:
                print(f"  Внешняя стена {i+1}/{len(wall_segments_from_openings)}: {segment['segment_id']}")
            else:
                print(f"  Внутренняя стена {i+1}/{len(wall_segments_from_openings)}: {segment['segment_id']}")
            
            # Создаем стену с соответствующим материалом
            wall_obj = create_wall_mesh(segment, wall_height_meters, wall_thickness, scale_factor,
                                      is_external=is_ext, brick_material=brick_material)
            if wall_obj:
                wall_objects.append(wall_obj)
                if is_ext:
                    external_walls_count += 1
                    wall_obj.name = f"External_Wall_{segment['segment_id']}"
                else:
                    internal_walls_count += 1
                    wall_obj.name = f"Internal_Wall_{segment['segment_id']}"
        
        # Создаем стены из сегментов от соединений
        for i, segment in enumerate(wall_segments_from_junctions):
            # Проверяем, является ли стена внешней
            is_ext = is_external_wall(segment, external_wall_ids)
            
            # Выводим отладочную информацию
            if is_ext:
                print(f"  Внешняя стена соединения {i+1}/{len(wall_segments_from_junctions)}: {segment['segment_id']}")
            else:
                print(f"  Внутренняя стена соединения {i+1}/{len(wall_segments_from_junctions)}: {segment['segment_id']}")
            
            # Создаем стену с соответствующим материалом
            wall_obj = create_wall_mesh(segment, wall_height_meters, wall_thickness, scale_factor,
                                      is_external=is_ext, brick_material=brick_material)
            if wall_obj:
                wall_objects.append(wall_obj)
                if is_ext:
                    external_walls_count += 1
                    wall_obj.name = f"External_Wall_{segment['segment_id']}"
                else:
                    internal_walls_count += 1
                    wall_obj.name = f"Internal_Wall_{segment['segment_id']}"
        
        # Создаем заполняющие элементы для проемов (над и под проемами)
        print("Создание заполняющих элементов для проемов...")
        fill_objects = create_fill_walls_between_openings(
            wall_segments_from_openings,
            openings,
            wall_thickness,
            scale_factor,
            wall_height_meters,
            wall_objects
        )
        
        print(f"Создано заполняющих элементов: {len(fill_objects)}")
        
        print(f"Всего создано стен: {len(wall_objects)}")
        print(f"Внешних стен: {external_walls_count}")
        print(f"Внутренних стен: {internal_walls_count}")
        
        # Создаем проемы как отдельные цветные объекты
        print("Создание проемов как отдельных объектов...")
        opening_objects = []
        external_openings_count = 0
        internal_openings_count = 0
        
        for opening in openings:
            is_ext = is_external_opening(opening, external_opening_ids)
            opening_obj = create_opening_mesh(opening, wall_height_meters, wall_thickness, scale_factor, is_external=is_ext)
            if opening_obj:
                opening_objects.append(opening_obj)
                if is_ext:
                    external_openings_count += 1
                    opening_obj.name = f"External_{opening['type']}_{opening['id']}"
                else:
                    internal_openings_count += 1
                    opening_obj.name = f"Internal_{opening['type']}_{opening['id']}"
        
        print(f"Создано проемов: {len(opening_objects)}/{len(openings)}")
        print(f"Внешних проемов: {external_openings_count}")
        print(f"Внутренних проемов: {internal_openings_count}")
        
        # Создаем колонны как отдельные цветные объекты
        print("Создание колонн как отдельных объектов...")
        pillar_objects = []
        for pillar in pillars:
            pillar_obj = create_pillar_mesh(pillar, wall_height_meters, wall_thickness, scale_factor)
            if pillar_obj:
                pillar_objects.append(pillar_obj)
        
        print(f"Создано колонн: {len(pillar_objects)}/{len(pillars)}")
        
        # Создаем метки с номерами junctions если включено
        if show_junction_labels:
            print("Создание меток с номерами junctions...")
            junction_label_objects = create_junction_labels(junctions, wall_segments_from_openings, wall_segments_from_junctions, scale_factor, wall_height_meters)
            print(f"Создано меток junctions: {len(junction_label_objects)}")
        else:
            print("Отображение номеров junctions отключено")
        
        # Инвертируем X-координаты всех объектов если нужно
        if invert_x:
            print("Инверсия X-координат...")
            
            # Находим общий центр X для всех объектов
            all_objects = wall_objects + opening_objects + pillar_objects
            if all_objects:
                all_min_x = min(v.co.x for obj in all_objects for v in obj.data.vertices)
                all_max_x = max(v.co.x for obj in all_objects for v in obj.data.vertices)
                center_x = (all_min_x + all_max_x) / 2
                
                # Применяем инверсию ко всем объектам
                for obj in all_objects:
                    invert_x_coordinates(obj, center_x)
        
        # Не объединяем стены, чтобы сохранить отдельные объекты для лучшей визуализации
        print("Сохранение отдельных объектов стен...")
        for i, wall_obj in enumerate(wall_objects):
            if wall_obj.name in bpy.data.objects:
                wall_obj.name = f"Wall_{i}"
        
        # Экспорт в OBJ
        if export_obj:
            print("Экспорт в OBJ формат...")
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            output_path = os.path.join(os.path.dirname(json_path), f"{base_name}_3d.obj")
            
            # Выделяем все объекты для экспорта
            bpy.ops.object.select_all(action='DESELECT')
            for obj in wall_objects:
                if obj and obj.name in bpy.data.objects:
                    obj.select_set(True)
            
            # Добавляем проёмы к выделению
            for obj in opening_objects:
                if obj and obj.name in bpy.data.objects:
                    obj.select_set(True)
            
            # Добавляем колонны к выделению
            for obj in pillar_objects:
                if obj and obj.name in bpy.data.objects:
                    obj.select_set(True)
            
            # Обновляем сцену перед экспортом
            bpy.context.view_layer.update()
            
            # Дополнительное обновление всех мешей
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH' and obj.data:
                    obj.data.update()
                    obj.data.update_tag()
            
            # Принудительное обновление зависимостей сцены
            bpy.context.view_layer.depsgraph.update()
            
            # Пробуем новый оператор экспорта, если доступен
            try:
                bpy.ops.wm.obj_export(
                    filepath=output_path,
                    export_materials=True,  # Включаем экспорт материалов для цвета
                    export_uv=True,  # Включаем экспорт UV-координат для текстур
                    export_normals=True
                )
            except AttributeError:
                # Если новый оператор недоступен, пробуем старый
                try:
                    bpy.ops.export_scene.obj(
                        filepath=output_path,
                        use_materials=True,  # Включаем экспорт материалов для цвета
                        use_uvs=True,  # Включаем экспорт UV-координат для текстур
                        use_normals=True
                    )
                except AttributeError:
                    print("Предупреждение: Экспорт OBJ недоступен в этой версии Blender")
            print(f"Сохранено: {output_path}")
        
        # Создаем изометрический рендер если нужно
        if render_isometric:
            print("Создание изометрического рендера...")
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            isometric_path = os.path.join(os.path.dirname(json_path), f"{base_name}_isometric.jpg")
            setup_isometric_camera_and_render(isometric_path)
        
        print("=" * 60)
        print("ЗАВЕРШЕНО!")
        print(f"Создано стен: {len(wall_objects)}")
        print(f"Внешних стен: {external_walls_count}")
        print(f"Внутренних стен: {internal_walls_count}")
        print(f"Создано проемов: {len(opening_objects)}")
        print(f"Внешних проемов: {external_openings_count}")
        print(f"Внутренних проемов: {internal_openings_count}")
        print(f"Создано колонн: {len(pillar_objects)}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при создании 3D стен: {e}")
        return False


# Использование
if __name__ == "__main__":
    # Путь к файлу с координатами стен
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "wall_coordinates.json")
    
    # Путь к текстуре кирпича (опционально)
    brick_texture_path = os.path.join(script_dir, "brick_texture.jpg")
    
    # Создаем 3D стены с указанной высотой и текстурой кирпича для внешних стен
    # Также создаем изометрический рендер и отображаем номера junctions
    create_3d_walls_from_json(json_path, wall_height=3.0, export_obj=True,
                             brick_texture_path=brick_texture_path, render_isometric=True,
                             show_junction_labels=True)
    