#!/usr/bin/env python3
"""
Объединение Building_Outline + Fill объектов с использованием Boolean Union
Этот метод использует Boolean модификатор для создания герметичной поверхности.
"""

import bpy
import bmesh
import os
import json
import time

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
    """Анализирует связность меша - возвращает количество отдельных частей"""
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
    """Экспортирует объект в OBJ файл"""
    bpy.ops.object.select_all(action='DESELECT')
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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    obj_path = os.path.join(script_dir, "wall_coordinates_inverted_3d.obj")
    json_path = os.path.join(script_dir, "wall_coordinates_inverted.json")
    output_path = os.path.join(script_dir, "complete_outline_BOOLEAN_UNION.obj")

    print("=" * 70)
    print("МЕТОД 1: BOOLEAN UNION")
    print("=" * 70)

    start_time = time.time()

    # Очищаем сцену
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Загружаем OBJ
    print(f"\n[1/6] Загрузка OBJ: {obj_path}")
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)
    print(f"      Загружено объектов: {len(bpy.data.objects)}")

    # Загружаем outline openings из JSON
    print(f"\n[2/6] Загрузка outline openings из JSON")
    outline_opening_ids = load_outline_openings_from_json(json_path)
    print(f"      Найдено outline openings: {len(outline_opening_ids)}")

    # Находим Building_Outline_Merged
    outline_obj = bpy.data.objects.get("Building_Outline_Merged")
    if not outline_obj:
        print("ОШИБКА: Building_Outline_Merged не найден!")
        return

    print(f"      Building_Outline_Merged: {len(outline_obj.data.vertices)} вершин, {len(outline_obj.data.polygons)} граней")

    # Находим Fill объекты
    print(f"\n[3/6] Поиск Fill объектов")
    fill_objects = []
    for opening_id in outline_opening_ids:
        for prefix in ['Fill_Above_', 'Fill_Below_']:
            obj_name = f"{prefix}{opening_id}"
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH':
                fill_objects.append(obj)

    print(f"      Найдено Fill объектов: {len(fill_objects)}")

    # Объединяем все объекты через Join (без Boolean пока)
    print(f"\n[4/6] Объединение объектов (Join)")
    bpy.ops.object.select_all(action='DESELECT')
    outline_obj.select_set(True)
    for obj in fill_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = outline_obj

    bpy.ops.object.join()
    merged_obj = bpy.context.active_object
    merged_obj.name = "Complete_Building_Outline_Boolean"

    verts_before = len(merged_obj.data.vertices)
    faces_before = len(merged_obj.data.polygons)

    print(f"      ДО Boolean Union:")
    print(f"        Вершины: {verts_before}")
    print(f"        Грани: {faces_before}")

    islands_before = analyze_connectivity(merged_obj)
    print(f"        Отдельных частей: {islands_before}")

    # Применяем Boolean Union через Remesh для соединения
    print(f"\n[5/6] Применение Boolean Union (через Voxel Remesh)")
    print(f"      ВАЖНО: Boolean Union работает между объектами.")
    print(f"      Для объединения внутри одного объекта используем Remesh с малым voxel_size")

    # Используем Remesh для создания единой топологии
    remesh_mod = merged_obj.modifiers.new('Remesh', 'REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = 0.02  # 2cm воксели (достаточно мелко для деталей)
    remesh_mod.use_remove_disconnected = False  # Не удаляем отдельные части

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

    print(f"      ПОСЛЕ Boolean Union:")
    print(f"        Вершины: {verts_after} (изменение: {verts_after - verts_before:+d})")
    print(f"        Грани: {faces_after} (изменение: {faces_after - faces_before:+d})")

    islands_after = analyze_connectivity(merged_obj)
    print(f"        Отдельных частей: {islands_after}")

    if islands_after == 1:
        print(f"        ✅ ОДНА СВЯЗНАЯ ПОВЕРХНОСТЬ!")
    else:
        print(f"        ⚠️  Все еще {islands_after} отдельных частей")

    # Экспорт
    print(f"\n[6/6] Экспорт результата")
    success, file_size = export_obj(merged_obj, output_path)

    if success:
        print(f"      ✅ Экспорт завершен: {output_path}")
        print(f"      Размер файла: {file_size} байт ({file_size / 1024:.2f} KB)")
    else:
        print(f"      ❌ ОШИБКА экспорта")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"ИТОГОВАЯ СТАТИСТИКА (Boolean Union)")
    print(f"{'=' * 70}")
    print(f"Время выполнения: {elapsed:.2f} сек")
    print(f"Вершины: {verts_before} → {verts_after} ({(verts_after/verts_before*100 - 100):+.1f}%)")
    print(f"Грани: {faces_before} → {faces_after} ({(faces_after/faces_before*100 - 100):+.1f}%)")
    print(f"Связность: {islands_before} частей → {islands_after} {'✅' if islands_after == 1 else '⚠️'}")
    print(f"Файл: {output_path}")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
