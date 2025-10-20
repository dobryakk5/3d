import bpy
import bmesh
import json
import os
from mathutils import Vector, Matrix
import random

def is_external_wall(segment_data, openings, wall_segments_from_openings, pillars=None):
    """
    Определяет, является ли стена внешней на основе наличия окон и расположения
    
    Args:
        segment_data: данные сегмента стены
        openings: список всех проемов
        wall_segments_from_openings: список сегментов стен от проемов
        pillars: список колонн (игнорируется при определении внешних стен)
    
    Returns:
        bool: True если стена внешняя, False если внутренняя
    """
    # Проверяем, связан ли сегмент стены с окном
    if "opening_id" in segment_data:
        opening_id = segment_data["opening_id"]
        for opening in openings:
            if opening["id"] == opening_id and opening["type"] == "window":
                return True
    
    # Собираем все координаты стен для определения границ (исключая колонны)
    all_x_coords = []
    all_y_coords = []
    
    for segment in wall_segments_from_openings:
        seg_bbox = segment["bbox"]
        all_x_coords.extend([seg_bbox["x"], seg_bbox["x"] + seg_bbox["width"]])
        all_y_coords.extend([seg_bbox["y"], seg_bbox["y"] + seg_bbox["height"]])
    
    # Добавляем координаты текущего сегмента
    bbox = segment_data["bbox"]
    all_x_coords.extend([bbox["x"], bbox["x"] + bbox["width"]])
    all_y_coords.extend([bbox["y"], bbox["y"] + bbox["height"]])
    
    if all_x_coords and all_y_coords:
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        
        # Проверяем, находится ли стена на границе
        wall_left = bbox["x"]
        wall_right = bbox["x"] + bbox["width"]
        wall_top = bbox["y"]
        wall_bottom = bbox["y"] + bbox["height"]
        
        # Добавляем небольшой допуск (5 пикселей)
        tolerance = 5
        is_boundary = (
            abs(wall_left - min_x) <= tolerance or
            abs(wall_right - max_x) <= tolerance or
            abs(wall_top - min_y) <= tolerance or
            abs(wall_bottom - max_y) <= tolerance
        )
        
        # Колонны игнорируются при определении внешних стен
        # Стена на границе всегда считается внешней
        
        if is_boundary:
            return True
    
    return False

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
    mapping.inputs['Scale'].default_value = (5.0, 5.0, 1.0)  # Масштаб кирпича (в два раза больше)
    
    # Соединяем узлы
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], texture_node.inputs['Vector'])
    links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
    
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
                    face.loops[1][uv_layer].uv = (width/10, 0)  # Масштаб 10 пикселей = 1 единица UV
                    face.loops[2][uv_layer].uv = (width/10, height/10)
                    face.loops[3][uv_layer].uv = (0, height/10)
                elif face.normal.z < -0.5:  # Нижняя грань
                    # UV для нижней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/10, 0)
                    face.loops[2][uv_layer].uv = (width/10, height/10)
                    face.loops[3][uv_layer].uv = (0, height/10)
                elif abs(face.normal.x) > 0.5:  # Боковые грани по X
                    # UV для боковых граней
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/10, 0)
                    face.loops[2][uv_layer].uv = (width/10, wall_height/10)
                    face.loops[3][uv_layer].uv = (0, wall_height/10)
                else:  # Боковые грани по Y
                    # UV для боковых граней
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (height/10, 0)
                    face.loops[2][uv_layer].uv = (height/10, wall_height/10)
                    face.loops[3][uv_layer].uv = (0, wall_height/10)
            else:  # vertical
                if face.normal.z > 0.5:  # Верхняя грань
                    # UV для верхней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/10, 0)
                    face.loops[2][uv_layer].uv = (width/10, height/10)
                    face.loops[3][uv_layer].uv = (0, height/10)
                elif face.normal.z < -0.5:  # Нижняя грань
                    # UV для нижней грани
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/10, 0)
                    face.loops[2][uv_layer].uv = (width/10, height/10)
                    face.loops[3][uv_layer].uv = (0, height/10)
                elif abs(face.normal.x) > 0.5:  # Боковые грани по X
                    # UV для боковых граней
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (width/10, 0)
                    face.loops[2][uv_layer].uv = (width/10, wall_height/10)
                    face.loops[3][uv_layer].uv = (0, wall_height/10)
                else:  # Боковые грани по Y
                    # UV для боковых граней
                    face.loops[0][uv_layer].uv = (0, 0)
                    face.loops[1][uv_layer].uv = (height/10, 0)
                    face.loops[2][uv_layer].uv = (height/10, wall_height/10)
                    face.loops[3][uv_layer].uv = (0, wall_height/10)
    
    # Обновляем нормали
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Создаем объект
    mesh = bpy.data.meshes.new(name=f"Wall_{segment_data['segment_id']}")
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Применяем материал в зависимости от типа стены
    if is_external and brick_material:
        # Внешняя стена с текстурой кирпича
        if obj.data.materials:
            obj.data.materials[0] = brick_material
        else:
            obj.data.materials.append(brick_material)
    else:
        # Внутренняя стена или стандартный материал
        mat = bpy.data.materials.new(name="WallMaterial")
        mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)  # Светло-серый
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    
    return obj

def create_opening_mesh(opening_data, wall_height, wall_thickness, scale_factor=1.0):
    """
    Создает 3D меш для проема (окна или двери) используя точные координаты из JSON
    без расширения за счет толщины стен, чтобы проемы были вертикальными без углублений
    """
    bbox = opening_data["bbox"]
    orientation = opening_data.get("orientation", "horizontal")
    
    # Создаем меш для проема
    bm = bmesh.new()
    
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
    
    # Применяем материал в зависимости от типа проема
    if opening_data["type"] == "door":
        # Зелёный материал для дверей
        mat = bpy.data.materials.new(name="DoorMaterial")
        mat.diffuse_color = (0.0, 0.8, 0.0, 1.0)  # Зелёный
    else:  # window
        # Голубой материал для окон
        mat = bpy.data.materials.new(name="WindowMaterial")
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

def create_fill_walls_between_openings(wall_segments, openings, wall_thickness, scale_factor):
    """
    Создает сегменты стен для заполнения пустых пространств между проемами
    
    Args:
        wall_segments: список сегментов стен от проемов
        openings: список проемов
        wall_thickness: толщина стен
        scale_factor: масштабный коэффициент
    
    Returns:
        list: список сегментов заполняющих стен
    """
    fill_segments = []
    
    # Группируем сегменты стен по ориентации и линии
    horizontal_segments = {}  # {y: [segments]}
    vertical_segments = {}    # {x: [segments]}
    
    for segment in wall_segments:
        bbox = segment["bbox"]
        orientation = segment.get("orientation", "horizontal")
        
        if orientation == "horizontal":
            y = bbox["y"]
            if y not in horizontal_segments:
                horizontal_segments[y] = []
            horizontal_segments[y].append(segment)
        else:
            x = bbox["x"]
            if x not in vertical_segments:
                vertical_segments[x] = []
            vertical_segments[x].append(segment)
    
    # Обрабатываем горизонтальные сегменты
    for y, segments in horizontal_segments.items():
        # Сортируем сегменты по x координате
        segments.sort(key=lambda s: s["bbox"]["x"])
        
        # Находим промежутки между сегментами
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]
            
            current_bbox = current_seg["bbox"]
            next_bbox = next_seg["bbox"]
            
            # Правый край текущего сегмента
            current_right = current_bbox["x"] + current_bbox["width"]
            # Левый край следующего сегмента
            next_left = next_bbox["x"]
            
            # Если есть промежуток больше, чем толщина стены
            if next_left - current_right > wall_thickness * scale_factor * 1.5:
                # Создаем заполняющий сегмент
                fill_width = next_left - current_right
                fill_segment = {
                    "segment_id": f"fill_h_{y}_{i}",
                    "bbox": {
                        "x": current_right,
                        "y": y,
                        "width": fill_width,
                        "height": wall_thickness * scale_factor
                    },
                    "orientation": "horizontal"
                }
                fill_segments.append(fill_segment)
    
    # Обрабатываем вертикальные сегменты
    for x, segments in vertical_segments.items():
        # Сортируем сегменты по y координате
        segments.sort(key=lambda s: s["bbox"]["y"])
        
        # Находим промежутки между сегментами
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]
            
            current_bbox = current_seg["bbox"]
            next_bbox = next_seg["bbox"]
            
            # Нижний край текущего сегмента
            current_bottom = current_bbox["y"] + current_bbox["height"]
            # Верхний край следующего сегмента
            next_top = next_bbox["y"]
            
            # Если есть промежуток больше, чем толщина стены
            if next_top - current_bottom > wall_thickness * scale_factor * 1.5:
                # Создаем заполняющий сегмент
                fill_height = next_top - current_bottom
                fill_segment = {
                    "segment_id": f"fill_v_{x}_{i}",
                    "bbox": {
                        "x": x,
                        "y": current_bottom,
                        "width": wall_thickness * scale_factor,
                        "height": fill_height
                    },
                    "orientation": "vertical"
                }
                fill_segments.append(fill_segment)
    
    return fill_segments

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
        
        # Настраиваем базовые параметры EEVEE_NEXT
        scene.eevee.taa_render_samples = 64
        # Устанавливаем только базовые параметры, которые гарантированно работают
        try:
            # Пробуем установить параметры теней
            if hasattr(scene.eevee, 'shadow_cube_size'):
                scene.eevee.shadow_cube_size = '1024'
            if hasattr(scene.eevee, 'shadow_cascade_size'):
                scene.eevee.shadow_cascade_size = '2048'
        except:
            print("Предупреждение: Не удалось установить параметры теней EEVEE")
        
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
        
        # Настраиваем освещение для EEVEE_NEXT
        # Удаляем существующее освещение
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()
        
        # Добавляем мягкий солнечный свет с меньшей энергией
        bpy.ops.object.light_add(type='SUN')
        sun_light = bpy.context.active_object
        sun_light.name = "SunLight"
        sun_light.location = (5, 5, 10)
        sun_light.rotation_euler = (0.5, -0.5, 0.5)
        sun_light.data.energy = 3.0  # Уменьшаем энергию для мягкости
        sun_light.data.angle = 0.2  # Добавляем небольшой угол для мягкости
        
        # Добавляем большой заполняющий свет с мягкими тенями
        bpy.ops.object.light_add(type='AREA')
        fill_light = bpy.context.active_object
        fill_light.name = "FillLight"
        fill_light.location = (0, 0, 8)
        fill_light.rotation_euler = (0, 0, 0)
        fill_light.data.energy = 2.0  # Уменьшаем энергию
        fill_light.data.size = 20.0  # Увеличиваем размер для более мягкого света
        
        # Добавляем мягкий свет сбоку для подсветки деталей
        bpy.ops.object.light_add(type='AREA')
        side_light = bpy.context.active_object
        side_light.name = "SideLight"
        side_light.location = (10, 0, 5)
        side_light.rotation_euler = (0, 1.57, 0)  # Поворот на 90 градусов
        side_light.data.energy = 1.5  # Низкая энергия для мягкой подсветки
        side_light.data.size = 15.0  # Большой размер для мягкости
        
        # Добавляем свет снизу для уменьшения контраста
        bpy.ops.object.light_add(type='AREA')
        bottom_light = bpy.context.active_object
        bottom_light.name = "BottomLight"
        bottom_light.location = (0, 0, 1)
        bottom_light.rotation_euler = (3.14, 0, 0)  # Направлен вверх
        bottom_light.data.energy = 0.8  # Очень низкая энергия
        bottom_light.data.size = 30.0  # Большой размер для очень мягкого света
        
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

def create_3d_walls_from_json(json_path, wall_height=2.0, export_obj=True, clear_scene=True, brick_texture_path=None, render_isometric=False):
    """
    Основная функция для создания 3D стен из JSON файла
    
    Args:
        json_path (str): Путь к JSON файлу с координатами стен
        wall_height (float): Высота стен в метрах (по умолчанию 2.0)
        export_obj (bool): Экспортировать результат в OBJ файл (по умолчанию True)
        clear_scene (bool): Очищать сцену перед созданием стен (по умолчанию True)
        brick_texture_path (str): Путь к файлу текстуры кирпича для внешних стен
        render_isometric (bool): Создавать изометрический рендер (по умолчанию False)
    
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
        invert_x = True  # Инвертировать X-координату
        
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
        
        # Создаем стены из сегментов от проемов
        wall_objects = []
        external_walls_count = 0
        internal_walls_count = 0
        print("Создание стен...")
        for i, segment in enumerate(wall_segments_from_openings):
            # Проверяем, является ли стена внешней
            is_ext = is_external_wall(segment, openings, wall_segments_from_openings, pillars)
            
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
        for segment in wall_segments_from_junctions:
            # Проверяем, является ли стена внешней
            is_ext = is_external_wall(segment, openings, wall_segments_from_openings, pillars)
            
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
        
        # Создаем заполняющие стены для пустых пространств между проемов
        # print("Создание заполняющих стен для пустых пространств...")
        # fill_wall_segments = create_fill_walls_between_openings(wall_segments_from_openings, openings, wall_thickness, scale_factor)
        #
        # for segment in fill_wall_segments:
        #     # Проверяем, является ли стена внешней
        #     is_ext = is_external_wall(segment, openings, wall_segments_from_openings, pillars)
        #
        #     # Создаем стену с соответствующим материалом
        #     wall_obj = create_wall_mesh(segment, wall_height_meters, wall_thickness, scale_factor,
        #                               is_external=is_ext, brick_material=brick_material)
        #     if wall_obj:
        #         wall_objects.append(wall_obj)
        #         if is_ext:
        #             external_walls_count += 1
        #             wall_obj.name = f"External_Fill_Wall_{segment['segment_id']}"
        #         else:
        #             internal_walls_count += 1
        #             wall_obj.name = f"Internal_Fill_Wall_{segment['segment_id']}"
        #
        # print(f"Создано заполняющих стен: {len(fill_wall_segments)}")
        print("Заполнение проемов временно отключено")
        
        print(f"Всего создано стен: {len(wall_objects)}")
        print(f"Внешних стен: {external_walls_count}")
        print(f"Внутренних стен: {internal_walls_count}")
        
        # Создаем проемы как отдельные цветные объекты
        print("Создание проемов как отдельных объектов...")
        opening_objects = []
        for opening in openings:
            opening_obj = create_opening_mesh(opening, wall_height_meters, wall_thickness, scale_factor)
            if opening_obj:
                opening_objects.append(opening_obj)
        
        print(f"Создано проемов: {len(opening_objects)}/{len(openings)}")
        
        # Создаем колонны как отдельные цветные объекты
        print("Создание колонн как отдельных объектов...")
        pillar_objects = []
        for pillar in pillars:
            pillar_obj = create_pillar_mesh(pillar, wall_height_meters, wall_thickness, scale_factor)
            if pillar_obj:
                pillar_objects.append(pillar_obj)
        
        print(f"Создано колонн: {len(pillar_objects)}/{len(pillars)}")
        
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
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            
            # Добавляем проёмы к выделению
            for obj in opening_objects:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            
            # Добавляем колонны к выделению
            for obj in pillar_objects:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            
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
    # Также создаем изометрический рендер
    create_3d_walls_from_json(json_path, wall_height=3.0, export_obj=True,
                             brick_texture_path=brick_texture_path, render_isometric=True)