аimport bpy
import bmesh
import json
import os
from mathutils import Vector

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

def create_wall_mesh(segment_data, wall_height, wall_thickness, scale_factor=1.0):
    """
    Создает 3D меш для сегмента стены на основе bbox данных
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
    
    # Обновляем нормали
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Создаем объект
    mesh = bpy.data.meshes.new(name=f"Wall_{segment_data['segment_id']}")
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    
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
    
    # Определяем параметры проема для стен высотой 1.33 метра
    if opening_data["type"] == "door":
        opening_height = 1.2  # Высота двери для стен 1.33м
        opening_bottom = 0.1  # Небольшой зазор снизу
    else:  # window
        opening_height = 0.8  # Высота окна для стен 1.33м
        opening_bottom = 0.4  # Высота от пола до низа окна
    
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
    
    return obj

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

def create_3d_walls_from_json(json_path, wall_height=2.0, export_obj=True, clear_scene=True):
    """
    Основная функция для создания 3D стен из JSON файла
    
    Args:
        json_path (str): Путь к JSON файлу с координатами стен
        wall_height (float): Высота стен в метрах (по умолчанию 2.0)
        export_obj (bool): Экспортировать результат в OBJ файл (по умолчанию True)
        clear_scene (bool): Очищать сцену перед созданием стен (по умолчанию True)
    
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
        
        print(f"Параметры: толщина стен={wall_thickness}м, высота стен={wall_height_meters}м")
        print(f"Масштабный коэффициент: {scale_factor}")
        print(f"Сегментов стен от проемов: {len(wall_segments_from_openings)}")
        print(f"Сегментов стен от соединений: {len(wall_segments_from_junctions)}")
        print(f"Проемов (окон и дверей): {len(openings)}")
        
        # Создаем стены из сегментов от проемов
        wall_objects = []
        print("Создание стен...")
        for i, segment in enumerate(wall_segments_from_openings):
            wall_obj = create_wall_mesh(segment, wall_height_meters, wall_thickness, scale_factor)
            if wall_obj:
                wall_objects.append(wall_obj)
        
        # Создаем стены из сегментов от соединений
        for segment in wall_segments_from_junctions:
            wall_obj = create_wall_mesh(segment, wall_height_meters, wall_thickness, scale_factor)
            if wall_obj:
                wall_objects.append(wall_obj)
        
        print(f"Всего создано стен: {len(wall_objects)}")
        
        # Создаем проемы и применяем их к стенам
        print("Создание проемов и применение к стенам...")
        successful_openings = 0
        for opening in openings:
            opening_obj = create_opening_mesh(opening, wall_height_meters, wall_thickness, scale_factor)
            opening_applied = False
            
            if opening_obj:
                # Сохраняем имя объекта для проверки
                opening_name = opening_obj.name
                
                # Находим стены, к которым нужно применить проем
                for wall_obj in wall_objects:
                    if check_wall_opening_intersection(wall_obj, opening_obj):
                        if apply_boolean_difference(wall_obj, opening_obj):
                            successful_openings += 1
                            opening_applied = True
                            break  # Проем применен к одной стене
                
                # Если проем не был применен, удаляем его
                if not opening_applied and opening_name in bpy.data.objects:
                    bpy.data.objects.remove(bpy.data.objects[opening_name], do_unlink=True)
        
        print(f"Успешно применено проемов: {successful_openings}/{len(openings)}")
        
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
            
            # Пробуем новый оператор экспорта, если доступен
            try:
                bpy.ops.wm.obj_export(
                    filepath=output_path,
                    export_materials=False,
                    export_uv=False
                )
            except AttributeError:
                # Если новый оператор недоступен, пробуем старый
                try:
                    bpy.ops.export_scene.obj(
                        filepath=output_path,
                        use_materials=False,
                        use_uvs=False
                    )
                except AttributeError:
                    print("Предупреждение: Экспорт OBJ недоступен в этой версии Blender")
            print(f"Сохранено: {output_path}")
        
        print("=" * 60)
        print("ЗАВЕРШЕНО!")
        print(f"Создано стен: {len(wall_objects)}")
        print(f"Обработано проемов: {successful_openings}")
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
    
    # Создаем 3D стены высотой 1.33 метра (2.0 / 1.5)
    create_3d_walls_from_json(json_path, wall_height=1.33, export_obj=True)