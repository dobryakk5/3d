#!/usr/bin/env python3
"""
Объединение Building_Outline + Fill объектов с использованием Voxel Remesh (FINE)
Voxel размер: 0.02м (2см) - мелкая сетка для острых углов

Организация объектов в коллекции:
- NEW_MODEL: Контур 1-го этажа + Фундамент (новая модель)
- OPENINGS: Окна/Двери/Колонны из внешнего контура

Экспортирует:
- Контур 1-го этажа (Building_Outline + Fill объекты, объединенные через voxel remesh)
- Фундамент (из JSON)
- Окна outline (только на внешнем контуре здания)
- Двери outline (только на внешнем контуре здания)
- Колонны (Pillar_*)
"""

import bpy
import bmesh
import os
import json
import time
from mathutils import Vector

def create_collection(collection_name):
    """Создает новую коллекцию или возвращает существующую"""
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]

    new_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(new_collection)
    return new_collection

def move_to_collection(obj, collection):
    """Перемещает объект в указанную коллекцию"""
    # Удаляем объект из всех коллекций
    for coll in obj.users_collection:
        coll.objects.unlink(obj)
    # Добавляем в новую коллекцию
    collection.objects.link(obj)

def load_outline_openings_from_json(json_path):
    """Загружает список outline openings из JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    outline_junction_ids = {v['junction_id'] for v in data['building_outline']['vertices']}
    outline_openings = []

    for opening in data['openings']:
        edge_junction_ids = {ej['junction_id'] for ej in opening['edge_junctions']}
        if edge_junction_ids.issubset(outline_junction_ids):
            outline_openings.append(opening['id'])

    return outline_openings

def identify_and_number_walls_from_mesh(obj, position_tolerance=0.15, normal_threshold=0.7):
    """
    Анализирует меш и группирует грани в стены на основе нормалей и позиций

    Координатная система Blender:
    - X - горизонталь (лево/право)
    - Y - горизонталь (вперед/назад)
    - Z - высота (вертикаль) - ИГНОРИРУЕТСЯ в нормалях

    Алгоритм:
    1. Для каждой грани получить нормаль (normal.x, normal.y), игнорировать normal.z
    2. Определить доминантную ось: |normal.x| > 0.7 или |normal.y| > 0.7
    3. Группировать грани:
       - Стены ±X: грани с |normal.x| > 0.7, группировать по близкому Y (допуск 0.15м)
       - Стены ±Y: грани с |normal.y| > 0.7, группировать по близкому X (допуск 0.15м)
    4. Присвоить номер каждой стене (integer layer "wall_number")

    Args:
        obj: объект Blender после Voxel Remesh
        position_tolerance: допуск для группировки по позиции (м)
        normal_threshold: минимальное значение доминантной компоненты нормали

    Returns:
        dict: {wall_number: {
            'direction': (nx, ny) - направление нормали стены,
            'position': float - позиция стены (X или Y),
            'axis': 'X' или 'Y' - по какой оси ориентирована стена,
            'face_count': int - количество граней,
            'center': (x, y, z) - центр стены
        }}
    """
    # Переключаемся в Edit mode для работы с BMesh
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Создаём integer layer для номеров стен
    wall_num_layer = bm.faces.layers.int.get("wall_number")
    if wall_num_layer is None:
        wall_num_layer = bm.faces.layers.int.new("wall_number")

    # Обнуляем все метки
    for f in bm.faces:
        f[wall_num_layer] = -1  # -1 означает "не принадлежит стене"

    # Группировка граней по стенам
    # Структура: {(axis, sign, position): [face1, face2, ...]}
    # axis: 'X' или 'Y'
    # sign: +1 или -1 (направление нормали)
    # position: округленная позиция (Y для стен ±X, X для стен ±Y)
    wall_groups = {}

    print(f"      Анализ граней для определения стен...")
    print(f"        Допуск по позиции: {position_tolerance}м")
    print(f"        Порог нормали: {normal_threshold}")

    # Проходим по всем граням
    faces_with_dominant_normal = 0
    faces_without_dominant_normal = 0

    for face in bm.faces:
        # Получаем нормализованную нормаль
        normal = face.normal.copy()
        normal.normalize()

        # Работаем только с (normal.x, normal.y), игнорируем normal.z
        nx = normal.x
        ny = normal.y

        # Определяем доминантную ось
        dominant_axis = None
        sign = 0

        if abs(nx) > normal_threshold:
            # Стена с нормалью по X
            dominant_axis = 'X'
            sign = 1 if nx > 0 else -1
        elif abs(ny) > normal_threshold:
            # Стена с нормалью по Y
            dominant_axis = 'Y'
            sign = 1 if ny > 0 else -1

        if dominant_axis is None:
            # Грань без доминантной нормали (например, наклонная или горизонтальная)
            faces_without_dominant_normal += 1
            continue

        faces_with_dominant_normal += 1

        # Вычисляем центр грани
        center = face.calc_center_median()

        # Определяем позицию стены
        if dominant_axis == 'X':
            # Стена с нормалью ±X: группируем по Y
            position = center.y
        else:  # dominant_axis == 'Y'
            # Стена с нормалью ±Y: группируем по X
            position = center.x

        # Округляем позицию для группировки (кластеризация)
        # Ищем существующую группу с близкой позицией
        group_key = None
        for key in wall_groups.keys():
            key_axis, key_sign, key_position = key
            if key_axis == dominant_axis and key_sign == sign:
                if abs(position - key_position) < position_tolerance:
                    group_key = key
                    break

        # Если не нашли группу - создаём новую
        if group_key is None:
            group_key = (dominant_axis, sign, position)
            wall_groups[group_key] = []

        wall_groups[group_key].append(face)

    print(f"        Граней с доминантной нормалью: {faces_with_dominant_normal}")
    print(f"        Граней без доминантной нормали: {faces_without_dominant_normal}")
    print(f"        Найдено групп (стен): {len(wall_groups)}")

    # Присваиваем номера стенам и собираем информацию
    wall_info = {}
    wall_number = 0

    for group_key, faces in wall_groups.items():
        axis, sign, position = group_key

        # Присваиваем номер всем граням этой стены
        for face in faces:
            face[wall_num_layer] = wall_number

        # Вычисляем центр стены (средняя позиция всех граней)
        center_x = sum(f.calc_center_median().x for f in faces) / len(faces)
        center_y = sum(f.calc_center_median().y for f in faces) / len(faces)
        center_z = sum(f.calc_center_median().z for f in faces) / len(faces)

        # Направление нормали стены
        if axis == 'X':
            direction = (sign, 0)
        else:  # axis == 'Y'
            direction = (0, sign)

        # Сохраняем информацию о стене
        wall_info[wall_number] = {
            'direction': direction,
            'position': position,
            'axis': axis,
            'face_count': len(faces),
            'center': (center_x, center_y, center_z),
            'sign': '+' if sign > 0 else '-'
        }

        print(f"          Стена #{wall_number}: ось {axis}{'+' if sign > 0 else '-'}, "
              f"позиция={position:.2f}м, граней={len(faces)}")

        wall_number += 1

    # Обновляем меш и возвращаемся в Object mode
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    return wall_info

def analyze_connectivity(obj):
    """Анализирует связность меша"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    islands = []
    all_faces = set(bm.faces)

    while all_faces:
        face = all_faces.pop()
        island = set([face])
        queue = [face]

        while queue:
            current = queue.pop(0)
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if linked_face in all_faces:
                        all_faces.remove(linked_face)
                        island.add(linked_face)
                        queue.append(linked_face)

        islands.append(island)

    bpy.ops.object.mode_set(mode='OBJECT')
    return len(islands)

def export_obj(obj, output_path):
    """Экспортирует объект или список объектов в OBJ файл"""
    bpy.ops.object.select_all(action='DESELECT')

    # Поддержка как одного объекта, так и списка объектов
    if isinstance(obj, list):
        for o in obj:
            if o is not None:
                o.select_set(True)
        if obj:
            bpy.context.view_layer.objects.active = obj[0]
    else:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.wm.obj_export(
            filepath=output_path,
            export_selected_objects=True,
            export_materials=True,
            export_normals=True,
            export_uv=True
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=output_path,
            use_selection=True,
            use_materials=True,
            use_normals=True,
            use_uvs=True
        )

    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        return True, file_size
    return False, 0

def create_procedural_brick_material():
    """
    Создает процедурный материал кирпича без использования файла текстуры
    """
    # ВАЖНО: Удаляем существующий материал, если он есть
    mat_name = "ProceduralBrickMaterial"
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    # Создаем новый материал
    brick_mat = bpy.data.materials.new(name=mat_name)
    brick_mat.use_nodes = True

    # ВАЖНО: Устанавливаем diffuse_color для правильного экспорта в OBJ/MTL
    brick_mat.diffuse_color = (0.6, 0.3, 0.1, 1.0)  # Кирпичный цвет (коричнево-красный)

    # Получаем узлы материала
    nodes = brick_mat.node_tree.nodes
    links = brick_mat.node_tree.links

    # Очищаем стандартные узлы
    nodes.clear()

    # Создаем основные узлы (простой материал без процедурной текстуры)
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    # ВАЖНО: Устанавливаем цвет напрямую в Principled BSDF для правильного экспорта OBJ
    principled_bsdf.inputs['Base Color'].default_value = (0.6, 0.3, 0.1, 1.0)  # Кирпичный цвет
    principled_bsdf.inputs['Roughness'].default_value = 0.8  # Шероховатость

    # Соединяем узлы
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # Specular может называться по-разному в разных версиях Blender
    if 'Specular IOR Level' in principled_bsdf.inputs:
        principled_bsdf.inputs['Specular IOR Level'].default_value = 0.2
    elif 'Specular' in principled_bsdf.inputs:
        principled_bsdf.inputs['Specular'].default_value = 0.2

    print("      Создан процедурный материал кирпича")

    return brick_mat

def create_white_material():
    """
    Создает белый материал для внутренних стен
    """
    # ВАЖНО: Удаляем существующий материал, если он есть
    mat_name = "WhiteInteriorMaterial"
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    # Создаем новый материал
    white_mat = bpy.data.materials.new(name=mat_name)
    white_mat.use_nodes = True

    # ВАЖНО: Устанавливаем diffuse_color для правильного экспорта в OBJ/MTL
    white_mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)  # Белый

    # Получаем узлы материала
    nodes = white_mat.node_tree.nodes
    links = white_mat.node_tree.links

    # Очищаем стандартные узлы
    nodes.clear()

    # Создаем основные узлы
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Настраиваем белый цвет
    principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Белый
    principled_bsdf.inputs['Roughness'].default_value = 0.5  # Небольшая шероховатость

    # Specular может называться по-разному в разных версиях Blender
    if 'Specular IOR Level' in principled_bsdf.inputs:
        principled_bsdf.inputs['Specular IOR Level'].default_value = 0.3
    elif 'Specular' in principled_bsdf.inputs:
        principled_bsdf.inputs['Specular'].default_value = 0.3

    # Соединяем узлы
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    print("      Создан белый материал для внутренних стен")

    return white_mat

def create_red_material():
    """
    Создает красный материал для текстовых меток с номерами стен
    """
    # ВАЖНО: Удаляем существующий материал, если он есть
    mat_name = "RedLabelMaterial"
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    # Создаем новый материал
    red_mat = bpy.data.materials.new(name=mat_name)
    red_mat.use_nodes = True

    # ВАЖНО: Устанавливаем diffuse_color для правильного экспорта в OBJ/MTL
    red_mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # Красный

    # Получаем узлы материала
    nodes = red_mat.node_tree.nodes
    links = red_mat.node_tree.links

    # Очищаем стандартные узлы
    nodes.clear()

    # Создаем основные узлы
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Настраиваем красный цвет
    principled_bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)  # Красный
    principled_bsdf.inputs['Roughness'].default_value = 0.3

    # Specular может называться по-разному в разных версиях Blender
    if 'Specular IOR Level' in principled_bsdf.inputs:
        principled_bsdf.inputs['Specular IOR Level'].default_value = 0.5
    elif 'Specular' in principled_bsdf.inputs:
        principled_bsdf.inputs['Specular'].default_value = 0.5

    # Соединяем узлы
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    print("      Создан красный материал для меток с номерами стен")

    return red_mat

def create_wall_number_labels(wall_info, red_material, label_size=0.5):
    """
    Создает 3D текстовые объекты с номерами стен для визуализации

    Для каждой стены:
    1. Создает текстовый объект с номером стены
    2. Размещает его в центре стены
    3. Применяет красный материал

    Args:
        wall_info: словарь с информацией о стенах (из identify_and_number_walls_from_mesh)
        red_material: красный материал для текста
        label_size: размер текстовых меток (м)

    Returns:
        list: список созданных текстовых объектов
    """
    labels = []

    print(f"      Создание текстовых меток для {len(wall_info)} стен...")

    for wall_num, info in wall_info.items():
        # Получаем центр стены
        center_x, center_y, center_z = info['center']

        # Создаем текстовую кривую
        text_curve = bpy.data.curves.new(name=f"WallLabel_{wall_num}", type='FONT')
        text_curve.body = str(wall_num)

        # Параметры текста
        text_curve.size = label_size
        text_curve.align_x = 'CENTER'
        text_curve.align_y = 'CENTER'

        # Создаем объект из кривой
        text_obj = bpy.data.objects.new(f"Wall_Number_{wall_num}", text_curve)
        bpy.context.collection.objects.link(text_obj)

        # Позиционируем метку в центре стены
        text_obj.location = (center_x, center_y, center_z)

        # Применяем красный материал
        if text_obj.data.materials:
            text_obj.data.materials[0] = red_material
        else:
            text_obj.data.materials.append(red_material)

        labels.append(text_obj)

        print(f"        Создана метка для стены #{wall_num} в позиции ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

    print(f"      ✅ Создано меток: {len(labels)}")

    return labels

def apply_materials_by_wall_numbers(obj, wall_info, brick_mat, white_mat):
    """
    Применяет материалы к граням объекта на основе номеров стен и направления нормалей

    Логика:
    - Для каждой грани читаем номер стены из integer layer "wall_number"
    - Определяем направление нормали стены (из wall_info)
    - External: нормаль грани совпадает с нормалью стены (dot > 0)
    - Internal: нормаль грани противоположна нормали стены (dot < 0)

    Args:
        obj: объект Blender (после identify_and_number_walls_from_mesh)
        wall_info: словарь с информацией о стенах
        brick_mat: материал кирпича (для внешних граней)
        white_mat: белый материал (для внутренних граней)
    """
    # ВАЖНО: Очищаем старые материалы перед добавлением новых
    obj.data.materials.clear()

    # Добавляем материалы к объекту (2 материала как в оригинале)
    obj.data.materials.append(brick_mat)  # index 0 - external (кирпич)
    obj.data.materials.append(white_mat)  # index 1 - internal (белый)

    print(f"      Материалы добавлены к объекту: {len(obj.data.materials)}")

    # Переключаемся в Edit mode для чтения номеров стен из BMesh
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    wall_num_layer = bm.faces.layers.int.get("wall_number")

    # Создаём словарь: индекс грани -> (номер стены, нормаль грани)
    face_data = {}

    if wall_num_layer:
        for face in bm.faces:
            wall_num = face[wall_num_layer]
            face_normal = face.normal.copy()
            face_normal.normalize()
            face_data[face.index] = (wall_num, face_normal)
    else:
        print(f"      ⚠️  BMesh layer 'wall_number' не найден!")

    # Выходим из Edit Mode перед применением материалов
    bpy.ops.object.mode_set(mode='OBJECT')

    # Счетчики для статистики
    brick_faces = 0
    white_faces = 0
    no_wall_faces = 0

    # Теперь применяем материалы в Object Mode
    for i, polygon in enumerate(obj.data.polygons):
        if i not in face_data:
            # Грань без данных - external по умолчанию
            polygon.material_index = 0
            brick_faces += 1
            continue

        wall_num, face_normal = face_data[i]

        if wall_num == -1 or wall_num not in wall_info:
            # Грань не принадлежит стене - external по умолчанию
            polygon.material_index = 0
            brick_faces += 1
            no_wall_faces += 1
            continue

        # Получаем направление нормали стены
        wall_direction = wall_info[wall_num]['direction']
        # Преобразуем направление (nx, ny) в 3D вектор (nx, ny, 0)
        wall_normal_3d = Vector((wall_direction[0], wall_direction[1], 0.0))
        wall_normal_3d.normalize()

        # Вычисляем dot product между нормалью грани и нормалью стены
        # Используем только (x, y) компоненты, игнорируем z
        face_normal_2d = Vector((face_normal.x, face_normal.y, 0.0))
        face_normal_2d.normalize()

        dot = face_normal_2d.dot(wall_normal_3d)

        # Применяем материал на основе dot product
        if dot > 0:
            # Нормаль грани совпадает с нормалью стены → внешняя сторона
            polygon.material_index = 0  # Кирпич
            brick_faces += 1
        else:
            # Нормаль грани противоположна нормали стены → внутренняя сторона
            polygon.material_index = 1  # Белый
            white_faces += 1

    # Обновляем меш для применения изменений
    obj.data.update()

    print(f"      Материалы применены на основе номеров стен:")
    print(f"        - Всего граней в меше: {len(obj.data.polygons)}")
    print(f"        - Кирпич (external): {brick_faces} граней")
    print(f"        - Белый (internal): {white_faces} граней")
    print(f"        - Граней без стены: {no_wall_faces}")

def create_foundation(foundation_data, z_offset=-0.75, thickness=0.3, scale_factor=0.01):
    """
    Создает 3D меш фундамента из данных JSON

    Args:
        foundation_data: Данные фундамента из JSON (словарь с ключом 'vertices')
        z_offset: Смещение по оси Z (по умолчанию -0.75м = -75см)
        thickness: Толщина фундамента (по умолчанию 0.3м = 30см)
        scale_factor: Коэффициент масштабирования из пикселей в метры (по умолчанию 0.01 = 1px=1см)

    Returns:
        Объект фундамента в Blender
    """
    if not foundation_data or 'vertices' not in foundation_data:
        print("ОШИБКА: Данные фундамента не найдены в JSON")
        return None

    vertices_2d = foundation_data['vertices']

    # Создаем новый меш и объект
    mesh = bpy.data.meshes.new(name="Foundation_Mesh")
    foundation_obj = bpy.data.objects.new("Foundation", mesh)
    bpy.context.collection.objects.link(foundation_obj)

    # Создаем вершины (верхний слой фундамента на z_offset)
    # В Blender: X=право, Y=глубина, Z=высота
    # ИНВЕРСИЯ Y: y вместо -y для зеркалирования по оси Y
    # Маппинг: JSON(x,y) в пикселях -> Blender(x*scale, y*scale, z_offset) в метрах
    vertices = []
    for v in vertices_2d:
        vertices.append((v['x'] * scale_factor, v['y'] * scale_factor, z_offset))

    # Добавляем нижний слой вершин (с учетом толщины)
    for v in vertices_2d:
        vertices.append((v['x'] * scale_factor, v['y'] * scale_factor, z_offset - thickness))

    # Создаем грани
    # Верхняя грань (полигон из верхних вершин)
    num_verts = len(vertices_2d)
    faces = []

    # Верхняя грань (0, 1, 2, 3, ...)
    top_face = list(range(num_verts))
    faces.append(top_face)

    # Нижняя грань (в обратном порядке для правильных нормалей)
    bottom_face = list(range(num_verts, 2 * num_verts))
    bottom_face.reverse()
    faces.append(bottom_face)

    # Боковые грани (соединяем верхний и нижний слои)
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        # Четырехугольник: верхний i, верхний next_i, нижний next_i, нижний i
        face = [i, next_i, next_i + num_verts, i + num_verts]
        faces.append(face)

    # Применяем геометрию к мешу
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Пересчитываем нормали
    bpy.context.view_layer.objects.active = foundation_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Создаем темно-серый материал для фундамента
    mat = bpy.data.materials.new(name="Foundation_Material_Dark_Gray")
    mat.use_nodes = True
    # Темно-серый цвет: RGB (0.2, 0.2, 0.2)
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
    foundation_obj.data.materials.append(mat)

    print(f"      Создан фундамент: {len(vertices)} вершин, {len(faces)} граней")
    print(f"      Позиция Y: {z_offset}м до {z_offset - thickness}м, толщина: {thickness}м")
    print(f"      Инверсия Y: ДА, цвет: темно-серый, касается низа здания")

    return foundation_obj

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    obj_path = os.path.join(script_dir, "wall_coordinates_inverted_3d.obj")
    json_path = os.path.join(script_dir, "wall_coordinates_inverted.json")
    output_path = os.path.join(script_dir, "complete_outline_VOXEL_REMESH_FINE.obj")

    print("=" * 70)
    print("МЕТОД 3: VOXEL REMESH FINE (острые углы)")
    print("=" * 70)

    start_time = time.time()

    # Очищаем сцену
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Создаем коллекции для организации объектов
    print(f"\n[1/6] Создание коллекций")
    new_model_collection = create_collection("NEW_MODEL")
    openings_collection = create_collection("OPENINGS")
    print(f"      ✅ Создана коллекция: NEW_MODEL (контур + фундамент)")
    print(f"      ✅ Создана коллекция: OPENINGS (окна, двери, колонны)")

    # Загружаем OBJ
    print(f"\n[2/6] Загрузка OBJ: {obj_path}")
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)
    print(f"      Загружено объектов: {len(bpy.data.objects)}")

    # Загружаем outline openings из JSON
    print(f"\n[3/7] Загрузка outline openings из JSON")
    outline_opening_ids = load_outline_openings_from_json(json_path)
    print(f"      Найдено outline openings: {len(outline_opening_ids)}")

    # Создаем фундамент из JSON
    print(f"\n[4/7] Создание фундамента из JSON")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    foundation_obj = None
    if 'foundation' in json_data:
        # scale_factor = 0.01 (1 пиксель = 1 см = 0.01 метра)
        # z_offset = 0 (верх фундамента на уровне низа здания)
        foundation_obj = create_foundation(json_data['foundation'], z_offset=0.0, thickness=0.75, scale_factor=0.01)
        if foundation_obj:
            print(f"      ✅ Фундамент создан успешно (масштаб 1px = 1см)")
            # Перемещаем фундамент в коллекцию NEW_MODEL
            move_to_collection(foundation_obj, new_model_collection)
            print(f"      ✅ Фундамент перемещен в коллекцию NEW_MODEL")
        else:
            print(f"      ⚠️  Не удалось создать фундамент")
    else:
        print(f"      ⚠️  Данные фундамента не найдены в JSON")

    # Находим Building_Outline_Merged
    outline_obj = bpy.data.objects.get("Building_Outline_Merged")
    if not outline_obj:
        print("ОШИБКА: Building_Outline_Merged не найден!")
        return

    print(f"      Building_Outline_Merged: {len(outline_obj.data.vertices)} вершин, {len(outline_obj.data.polygons)} граней")

    # Находим Fill объекты
    print(f"\n[5/7] Поиск Fill объектов")
    fill_objects = []
    for opening_id in outline_opening_ids:
        for prefix in ['Fill_Above_', 'Fill_Below_']:
            obj_name = f"{prefix}{opening_id}"
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH':
                fill_objects.append(obj)

    print(f"      Найдено Fill объектов: {len(fill_objects)}")

    # Находим окна, двери и колонны для экспорта (только outline проемы)
    print(f"\n[5.5/7] Поиск окон, дверей и колонн (только outline)")
    windows = []
    doors = []
    pillars = []

    # Создаем set для быстрого поиска
    outline_set = set(outline_opening_ids)

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Проверяем, принадлежит ли проем к outline
            is_outline_opening = False
            for opening_id in outline_set:
                if opening_id in obj.name:
                    is_outline_opening = True
                    break

            if is_outline_opening:
                # Окна (Internal_window_*, External_window_*)
                if obj.name.startswith('Internal_window_') or obj.name.startswith('External_window_'):
                    windows.append(obj)
                    move_to_collection(obj, openings_collection)
                # Двери (Internal_door_*, External_door_*)
                elif obj.name.startswith('Internal_door_') or obj.name.startswith('External_door_'):
                    doors.append(obj)
                    move_to_collection(obj, openings_collection)

            # Колонны всегда добавляем (они не зависят от outline)
            if obj.name.startswith('Pillar_'):
                pillars.append(obj)
                move_to_collection(obj, openings_collection)

    print(f"      Найдено окон (outline): {len(windows)}")
    print(f"      Найдено дверей (outline): {len(doors)}")
    print(f"      Найдено колонн: {len(pillars)}")
    print(f"      ✅ Объекты перемещены в коллекцию OPENINGS")

    # Объединяем все объекты через Join
    print(f"\n[6/7] Объединение объектов (Join)")
    bpy.ops.object.select_all(action='DESELECT')
    outline_obj.select_set(True)
    for obj in fill_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = outline_obj

    bpy.ops.object.join()
    merged_obj = bpy.context.active_object
    merged_obj.name = "Complete_Building_Outline_Remesh_Fine"

    # Перемещаем объединенный контур в коллекцию NEW_MODEL
    move_to_collection(merged_obj, new_model_collection)

    verts_before = len(merged_obj.data.vertices)
    faces_before = len(merged_obj.data.polygons)

    print(f"      ДО Remesh:")
    print(f"        Вершины: {verts_before}")
    print(f"        Грани: {faces_before}")

    islands_before = analyze_connectivity(merged_obj)
    print(f"        Отдельных частей: {islands_before}")

    # Применяем Voxel Remesh с МЕЛКИМИ вокселями для острых углов
    print(f"\n[6.5/7] Применение Voxel Remesh (FINE)")
    print(f"      Voxel размер: 0.02м (2см) - мелкая сетка для острых углов")

    remesh_mod = merged_obj.modifiers.new('Remesh', 'REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = 0.02  # 2cm воксели (мелко, острые углы)
    remesh_mod.use_remove_disconnected = False
    remesh_mod.use_smooth_shade = False  # Плоское затенение для острых углов

    bpy.context.view_layer.objects.active = merged_obj
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    # Удаление внутренних граней
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_interior_faces()
    bpy.ops.object.mode_set(mode='OBJECT')
    interior_faces = sum(1 for face in merged_obj.data.polygons if face.select)
    if interior_faces > 0:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Пересчет нормалей
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    merged_obj.data.update()

    verts_after = len(merged_obj.data.vertices)
    faces_after = len(merged_obj.data.polygons)

    print(f"      ПОСЛЕ Remesh:")
    print(f"        Вершины: {verts_after} (изменение: {verts_after - verts_before:+d})")
    print(f"        Грани: {faces_after} (изменение: {faces_after - faces_before:+d})")

    islands_after = analyze_connectivity(merged_obj)
    print(f"        Отдельных частей: {islands_after}")

    if islands_after == 1:
        print(f"        ✅ ОДНА СВЯЗНАЯ ПОВЕРХНОСТЬ!")
    else:
        print(f"        ⚠️  Все еще {islands_after} отдельных частей")

    # Определение стен из меша (новый метод)
    print(f"\n[6.7/7] Идентификация стен из меша")
    wall_info = identify_and_number_walls_from_mesh(
        merged_obj,
        position_tolerance=0.15,  # половина толщины стены
        normal_threshold=0.7      # |компонента| > 0.7
    )

    print(f"      Найдено стен: {len(wall_info)}")
    for wall_num, info in wall_info.items():
        direction_str = f"{info['axis']}{info['sign']}"
        print(f"        Стена #{wall_num}: направление {direction_str}, "
              f"позиция={info['position']:.2f}м, граней={info['face_count']}")

    # Экспорт wall_info в JSON для отладки
    print(f"\n[6.75/7] Экспорт информации о стенах в JSON")
    wall_info_json_path = os.path.join(script_dir, "wall_numbers_debug.json")

    # Преобразуем wall_info в JSON-сериализуемый формат
    wall_info_serializable = {}
    for wall_num, info in wall_info.items():
        wall_info_serializable[str(wall_num)] = {
            'direction': list(info['direction']),
            'position': float(info['position']),
            'axis': info['axis'],
            'face_count': info['face_count'],
            'center': list(info['center']),
            'sign': info['sign']
        }

    with open(wall_info_json_path, 'w', encoding='utf-8') as f:
        json.dump(wall_info_serializable, f, indent=2, ensure_ascii=False)

    print(f"      ✅ Информация о стенах сохранена: {wall_info_json_path}")

    # Визуализация номеров стен
    print(f"\n[6.8/7] Создание меток с номерами стен")
    red_mat = create_red_material()
    wall_labels = create_wall_number_labels(wall_info, red_mat, label_size=0.5)
    print(f"      ✅ Создано меток: {len(wall_labels)}")

    # Применение материалов на основе номеров стен
    print(f"\n[6.9/7] Применение материалов (кирпич/белый)")

    # Создаем материалы
    brick_mat = create_procedural_brick_material()
    white_mat = create_white_material()

    # Применяем материалы к объединенному контуру на основе номеров стен
    apply_materials_by_wall_numbers(merged_obj, wall_info, brick_mat, white_mat)
    print(f"      ✅ Материалы применены успешно")

    # Экспорт
    print(f"\n[7/7] Экспорт результата")
    # Экспортируем merged_obj, foundation_obj, windows, doors, pillars вместе
    objects_to_export = [merged_obj]

    if foundation_obj is not None:
        objects_to_export.append(foundation_obj)

    # Добавляем окна, двери и колонны
    objects_to_export.extend(windows)
    objects_to_export.extend(doors)
    objects_to_export.extend(pillars)

    total_objects = len(objects_to_export)
    print(f"      Объекты для экспорта:")
    print(f"        - Контур здания: 1")
    if foundation_obj:
        print(f"        - Фундамент: 1")
    print(f"        - Окна outline: {len(windows)}")
    print(f"        - Двери outline: {len(doors)}")
    print(f"        - Колонны: {len(pillars)}")
    print(f"        Всего объектов: {total_objects}")

    success, file_size = export_obj(objects_to_export, output_path)

    if success:
        print(f"      ✅ Экспорт завершен: {output_path}")
        print(f"      Размер файла: {file_size} байт ({file_size / 1024:.2f} KB)")
        print(f"      ✅ Экспортировано объектов: {total_objects}")
        if foundation_obj:
            print(f"        ✅ Фундамент")
        if windows:
            print(f"        ✅ Окна outline ({len(windows)})")
        if doors:
            print(f"        ✅ Двери outline ({len(doors)})")
        if pillars:
            print(f"        ✅ Колонны ({len(pillars)})")
    else:
        print(f"      ❌ ОШИБКА экспорта")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"ИТОГОВАЯ СТАТИСТИКА (Контур + Фундамент + Объекты)")
    print(f"{'=' * 70}")
    print(f"Время выполнения: {elapsed:.2f} сек")
    print(f"Voxel размер: 0.02м (2см) - для острых углов")
    print(f"\nКоллекция NEW_MODEL (новая модель):")
    print(f"  Контур здания:")
    print(f"    Вершины: {verts_before} → {verts_after} ({(verts_after/verts_before*100 - 100):+.1f}%)")
    print(f"    Грани: {faces_before} → {faces_after} ({(faces_after/faces_before*100 - 100):+.1f}%)")
    print(f"    Связность: {islands_before} частей → {islands_after} {'✅' if islands_after == 1 else '⚠️'}")
    if foundation_obj:
        print(f"  Фундамент: ✅ Создан (Y = 0 до -0.75м, толщина = 0.75м)")
    else:
        print(f"  Фундамент: ⚠️  Не создан")
    print(f"\nКоллекция OPENINGS (проемы на внешнем контуре):")
    print(f"  Окна outline: {len(windows)} {'✅' if windows else '❌'}")
    print(f"  Двери outline: {len(doors)} {'✅' if doors else '❌'}")
    print(f"  Колонны: {len(pillars)} {'✅' if pillars else '❌'}")
    print(f"\nВсего объектов в экспорте: {total_objects}")
    print(f"Файл: {output_path}")
    print(f"\n💡 Совет: В Blender включите/выключите коллекции NEW_MODEL и OPENINGS")
    print(f"   чтобы показать/скрыть контур или проемы")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
