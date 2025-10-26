#!/usr/bin/env python3
"""
Создает изометрический рендер здания с фундаментом
"""

import bpy
import os
import math

# Очищаем сцену
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Удаляем все камеры и источники света
for obj in list(bpy.data.cameras):
    bpy.data.cameras.remove(obj)
for obj in list(bpy.data.lights):
    bpy.data.lights.remove(obj)

# Путь к OBJ файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
obj_path = os.path.join(script_dir, "complete_outline_VOXEL_REMESH_FINE.obj")

print(f"\n{'='*60}")
print(f"РЕНДЕР ЗДАНИЯ С ФУНДАМЕНТОМ")
print(f"{'='*60}\n")

# Импортируем OBJ
try:
    bpy.ops.wm.obj_import(filepath=obj_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=obj_path)

# Настраиваем материалы
building_obj = None
foundation_obj = None

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        print(f"Объект: {obj.name}")

        if 'Foundation' in obj.name:
            foundation_obj = obj
            # Темно-серый материал для фундамента
            mat = bpy.data.materials.new(name="Foundation_Dark_Gray")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            print(f"  Материал: темно-серый")
        else:
            building_obj = obj
            # Белый материал для здания
            mat = bpy.data.materials.new(name="Building_White")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.95, 0.95, 0.95, 1.0)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            print(f"  Материал: белый")

# Создаем камеру для изометрического вида
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Позиционируем камеру для изометрического вида
# Центр сцены приблизительно (0, -6, 0)
cam_obj.location = (15, -15, 10)
cam_obj.rotation_euler = (math.radians(60), 0, math.radians(45))

# Создаем источник света
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 2.0
light_obj = bpy.data.objects.new("Sun", light_data)
bpy.context.scene.collection.objects.link(light_obj)
light_obj.location = (10, -10, 20)
light_obj.rotation_euler = (math.radians(45), 0, math.radians(30))

# Настройки рендера
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Путь для сохранения рендера
render_path = os.path.join(script_dir, "foundation_render.png")
bpy.context.scene.render.filepath = render_path

# Рендерим
print(f"\nРендеринг...")
bpy.ops.render.render(write_still=True)

print(f"\n✅ Рендер сохранен: {render_path}")
print(f"\n{'='*60}\n")
