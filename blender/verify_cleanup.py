#!/usr/bin/env python3
"""
Скрипт для визуальной проверки результата очистки геометрии.
Сравнивает оригинальный и очищенный объекты.
"""

import bpy
import os


def compare_objects():
    """
    Сравнивает два OBJ файла: оригинальный и очищенный.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Пути к файлам
    original_path = os.path.join(script_dir, "wall_coordinates_inverted_3d.obj")
    cleaned_path = os.path.join(script_dir, "wall_outline_cleaned.obj")

    print("=" * 60)
    print("СРАВНЕНИЕ ГЕОМЕТРИИ: ДО и ПОСЛЕ ОЧИСТКИ")
    print("=" * 60)

    # Проверяем наличие файлов
    if not os.path.exists(original_path):
        print(f"ОШИБКА: Оригинальный файл не найден: {original_path}")
        return

    if not os.path.exists(cleaned_path):
        print(f"ОШИБКА: Очищенный файл не найден: {cleaned_path}")
        return

    # Очищаем сцену
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Загружаем оригинальный файл
    print(f"\nЗагрузка оригинала: {original_path}")
    try:
        bpy.ops.wm.obj_import(filepath=original_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=original_path)

    # Находим объект Building_Outline_Merged
    original_obj = bpy.data.objects.get("Building_Outline_Merged")

    if not original_obj:
        print("ОШИБКА: Объект 'Building_Outline_Merged' не найден в оригинальном файле")
        return

    # Сохраняем статистику оригинала
    original_stats = {
        "vertices": len(original_obj.data.vertices),
        "faces": len(original_obj.data.polygons),
        "edges": len(original_obj.data.edges)
    }

    print(f"\nОРИГИНАЛ (до очистки):")
    print(f"  Вершины: {original_stats['vertices']}")
    print(f"  Грани: {original_stats['faces']}")
    print(f"  Ребра: {original_stats['edges']}")

    # Переименуем оригинал для различия
    original_obj.name = "Original_Outline"
    original_obj.location.x = -5  # Сдвигаем влево

    # Загружаем очищенный файл
    print(f"\nЗагрузка очищенного: {cleaned_path}")
    try:
        bpy.ops.wm.obj_import(filepath=cleaned_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=cleaned_path)

    # Находим очищенный объект
    cleaned_obj = bpy.data.objects.get("Building_Outline_Merged")

    if not cleaned_obj:
        print("ОШИБКА: Объект 'Building_Outline_Merged' не найден в очищенном файле")
        return

    # Сохраняем статистику очищенного
    cleaned_stats = {
        "vertices": len(cleaned_obj.data.vertices),
        "faces": len(cleaned_obj.data.polygons),
        "edges": len(cleaned_obj.data.edges)
    }

    print(f"\nОЧИЩЕННЫЙ (после очистки):")
    print(f"  Вершины: {cleaned_stats['vertices']}")
    print(f"  Грани: {cleaned_stats['faces']}")
    print(f"  Ребра: {cleaned_stats['edges']}")

    # Переименуем очищенный для различия
    cleaned_obj.name = "Cleaned_Outline"
    cleaned_obj.location.x = 5  # Сдвигаем вправо

    # Вычисляем разницу
    print("\n" + "=" * 60)
    print("РАЗНИЦА:")
    print(f"  Удалено вершин: {original_stats['vertices'] - cleaned_stats['vertices']}")
    print(f"  Удалено граней: {original_stats['faces'] - cleaned_stats['faces']}")
    print(f"  Удалено ребер: {original_stats['edges'] - cleaned_stats['edges']}")

    # Процент улучшения
    vertex_improvement = (1 - cleaned_stats['vertices'] / original_stats['vertices']) * 100
    face_improvement = (1 - cleaned_stats['faces'] / original_stats['faces']) * 100

    print(f"\n  Уменьшение вершин: {vertex_improvement:.1f}%")
    print(f"  Уменьшение граней: {face_improvement:.1f}%")

    print("=" * 60)

    # Создаем материалы для визуального различия
    # Оригинал - красноватый (с проблемами)
    original_mat = bpy.data.materials.new(name="Original_Material")
    original_mat.diffuse_color = (1.0, 0.3, 0.3, 1.0)  # Красноватый
    original_mat.use_nodes = True

    if original_obj.data.materials:
        original_obj.data.materials[0] = original_mat
    else:
        original_obj.data.materials.append(original_mat)

    # Очищенный - зеленоватый (чистый)
    cleaned_mat = bpy.data.materials.new(name="Cleaned_Material")
    cleaned_mat.diffuse_color = (0.3, 1.0, 0.3, 1.0)  # Зеленоватый
    cleaned_mat.use_nodes = True

    if cleaned_obj.data.materials:
        cleaned_obj.data.materials[0] = cleaned_mat
    else:
        cleaned_obj.data.materials.append(cleaned_mat)

    print("\n✅ Объекты загружены и готовы к визуальному сравнению:")
    print(f"   - Оригинал (красный): {original_obj.name}")
    print(f"   - Очищенный (зеленый): {cleaned_obj.name}")
    print("\nОткройте сцену в Blender UI для визуального сравнения.")

    # Сохраняем blend файл для просмотра
    blend_path = os.path.join(script_dir, "outline_comparison.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"\n💾 Сцена сохранена: {blend_path}")


if __name__ == "__main__":
    compare_objects()
