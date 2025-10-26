#!/usr/bin/env python3
"""
Объединение Building_Outline + Fill объектов с использованием Voxel Remesh (FINE)
Voxel размер: 0.02м (2см) - мелкая сетка для острых углов

Экспортирует:
- Контур 1-го этажа (Building_Outline + Fill объекты, объединенные через voxel remesh)
- Фундамент (из JSON)
- Окна (Internal_window_*, External_window_*)
- Двери (Internal_door_*, External_door_*)
- Колонны (Pillar_*)
"""

import bpy
import bmesh
import os
import json
import time

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
    old_model_collection = create_collection("OLD_MODEL")
    print(f"      ✅ Создана коллекция: NEW_MODEL (контур + фундамент)")
    print(f"      ✅ Создана коллекция: OLD_MODEL (окна, двери, колонны)")

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

    # Находим окна, двери и колонны для экспорта
    print(f"\n[5.5/7] Поиск окон, дверей и колонн")
    windows = []
    doors = []
    pillars = []

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Окна (Internal_window_*, External_window_*)
            if obj.name.startswith('Internal_window_') or obj.name.startswith('External_window_'):
                windows.append(obj)
                move_to_collection(obj, old_model_collection)
            # Двери (Internal_door_*, External_door_*)
            elif obj.name.startswith('Internal_door_') or obj.name.startswith('External_door_'):
                doors.append(obj)
                move_to_collection(obj, old_model_collection)
            # Колонны/столбы (Pillar_*)
            elif obj.name.startswith('Pillar_'):
                pillars.append(obj)
                move_to_collection(obj, old_model_collection)

    print(f"      Найдено окон: {len(windows)}")
    print(f"      Найдено дверей: {len(doors)}")
    print(f"      Найдено колонн: {len(pillars)}")
    print(f"      ✅ Объекты перемещены в коллекцию OLD_MODEL")

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
    print(f"        - Окна: {len(windows)}")
    print(f"        - Двери: {len(doors)}")
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
            print(f"        ✅ Окна ({len(windows)})")
        if doors:
            print(f"        ✅ Двери ({len(doors)})")
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
    print(f"\nКоллекция OLD_MODEL (старая модель):")
    print(f"  Окна: {len(windows)} {'✅' if windows else '❌'}")
    print(f"  Двери: {len(doors)} {'✅' if doors else '❌'}")
    print(f"  Колонны: {len(pillars)} {'✅' if pillars else '❌'}")
    print(f"\nВсего объектов в экспорте: {total_objects}")
    print(f"Файл: {output_path}")
    print(f"\n💡 Совет: В Blender включите/выключите коллекции NEW_MODEL и OLD_MODEL")
    print(f"   чтобы показать/скрыть новую или старую модель")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
