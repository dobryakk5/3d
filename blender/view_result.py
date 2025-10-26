#!/usr/bin/env python3
"""
Открывает результат VOXEL_REMESH_FINE в Blender для просмотра
"""

import bpy
import os

# Очищаем сцену
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Путь к OBJ файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
obj_path = os.path.join(script_dir, "complete_outline_VOXEL_REMESH_FINE.obj")

print(f"Загрузка: {obj_path}")

# Импортируем OBJ
try:
    bpy.ops.wm.obj_import(filepath=obj_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=obj_path)

print(f"Загружено объектов: {len(bpy.data.objects)}")

# Настраиваем камеру для просмотра
for obj in bpy.data.objects:
    print(f"  - {obj.name} ({obj.type})")
    if obj.type == 'MESH':
        # Центрируем вид на объекте
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

# Переключаемся в режим solid для лучшего просмотра
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'SOLID'

print("\n✅ Модель загружена! Используйте мышь для осмотра:")
print("   - Средняя кнопка мыши: вращение")
print("   - Shift + средняя кнопка: панорама")
print("   - Колесико: зум")
