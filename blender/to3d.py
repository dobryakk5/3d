import bpy
import bmesh
from mathutils import Vector

def floor_plan_to_3d(svg_path, wall_height=3.0):
    """
    Конвертирует 2D floor plan (SVG/DWG) в 3D mesh
    """
    # Очистить сцену
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Импорт 2D чертежа (SVG)
    bpy.ops.import_curve.svg(filepath=svg_path)
    
    # Получить импортированную кривую
    curve_obj = bpy.context.selected_objects[0]
    
    # Конвертировать в mesh
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.convert(target='MESH')
    
    # Экструдировать вверх (создать стены)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, wall_height)}
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Экспорт как OBJ (БЕЗ материалов/текстур)
    output_path = svg_path.replace('.svg', '_3d.obj')
    bpy.ops.export_scene.obj(
        filepath=output_path,
        use_materials=False,  # ← БЕЗ материалов
        use_uvs=False        # ← БЕЗ UV-координат
    )
    
    print(f"Сохранено: {output_path}")

# Использование
floor_plan_to_3d("floor_plan.svg", wall_height=3.0)