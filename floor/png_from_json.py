import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

def load_json_data(json_path):
    """
    Загружает данные из JSON файла
    """
    with open(json_path, 'r') as file:
        return json.load(file)

def create_wall_mesh(segment_data, wall_height, scale_factor=1.0):
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
    
    # Создаём вершины
    if orientation == "horizontal":
        # Горизонтальная стена
        vertices = [
            [x, y, 0],
            [x + width, y, 0],
            [x + width, y + height, 0],
            [x, y + height, 0],
            [x, y, wall_height],
            [x + width, y, wall_height],
            [x + width, y + height, wall_height],
            [x, y + height, wall_height]
        ]
    else:  # vertical
        # Вертикальная стена
        vertices = [
            [x, y, 0],
            [x + width, y, 0],
            [x + width, y + height, 0],
            [x, y + height, 0],
            [x, y, wall_height],
            [x + width, y, wall_height],
            [x + width, y + height, wall_height],
            [x, y + height, wall_height]
        ]
    
    # Создаём грани
    faces = [
        [0, 1, 2, 3],  # нижняя грань
        [4, 5, 6, 7],  # верхняя грань
        [0, 1, 5, 4],  # передняя грань
        [2, 3, 7, 6],  # задняя грань
        [0, 3, 7, 4],  # левая грань
        [1, 2, 6, 5]   # правая грань
    ]
    
    return vertices, faces

def create_opening_mesh(opening_data, wall_height, scale_factor=1.0):
    """
    Создает 3D меш для проема (окна или двери)
    """
    bbox = opening_data["bbox"]
    orientation = opening_data.get("orientation", "horizontal")
    
    # Определяем параметры проема
    if opening_data["type"] == "door":
        opening_height = 0.20  # Высота двери для стен 0.222м (1.2 / 6)
        opening_bottom = 0.017  # Небольшой зазор снизу (0.1 / 6)
    else:  # window
        opening_height = 0.133  # Высота окна для стен 0.222м (0.8 / 6)
        opening_bottom = 0.067  # Высота от пола до низа окна (0.4 / 6)
    
    # Получаем координаты с применением масштабирования
    x = bbox["x"] * scale_factor
    y = bbox["y"] * scale_factor
    width = bbox["width"] * scale_factor
    height = bbox["height"] * scale_factor
    
    # Создаём вершины
    vertices = [
        [x, y, opening_bottom],
        [x + width, y, opening_bottom],
        [x + width, y + height, opening_bottom],
        [x, y + height, opening_bottom],
        [x, y, opening_bottom + opening_height],
        [x + width, y, opening_bottom + opening_height],
        [x + width, y + height, opening_bottom + opening_height],
        [x, y + height, opening_bottom + opening_height]
    ]
    
    # Создаём грани
    faces = [
        [0, 1, 2, 3],  # нижняя грань
        [4, 5, 6, 7],  # верхняя грань
        [0, 1, 5, 4],  # передняя грань
        [2, 3, 7, 6],  # задняя грань
        [0, 3, 7, 4],  # левая грань
        [1, 2, 6, 5]   # правая грань
    ]
    
    return vertices, faces

def create_pillar_mesh(pillar_data, wall_height, scale_factor=1.0):
    """
    Создает 3D меш для колонны
    """
    bbox = pillar_data["bbox"]
    
    # Получаем координаты с применением масштабирования
    x = bbox["x"] * scale_factor
    y = bbox["y"] * scale_factor
    width = bbox["width"] * scale_factor
    height = bbox["height"] * scale_factor
    
    # Создаём вершины
    vertices = [
        [x, y, 0],
        [x + width, y, 0],
        [x + width, y + height, 0],
        [x, y + height, 0],
        [x, y, wall_height],
        [x + width, y, wall_height],
        [x + width, y + height, wall_height],
        [x, y + height, wall_height]
    ]
    
    # Создаём грани
    faces = [
        [0, 1, 2, 3],  # нижняя грань
        [4, 5, 6, 7],  # верхняя грань
        [0, 1, 5, 4],  # передняя грань
        [2, 3, 7, 6],  # задняя грань
        [0, 3, 7, 4],  # левая грань
        [1, 2, 6, 5]   # правая грань
    ]
    
    return vertices, faces

def visualize_3d_model_from_json(json_data):
    """
    Визуализирует 3D модель из JSON данных с раскрашиванием элементов
    """
    # Параметры (можно изменять вручную)
    wall_height = 0.222  # Высота стен в метрах (1.33 / 6)
    scale_factor = 0.01  # 1 пиксель = 0.01 метра
    z_scale_factor = 0.5  # Коэффициент уменьшения Z-оси (1.0 = нормальный, 0.5 = в 2 раза меньше)
    
    # Получаем данные
    wall_segments_from_openings = json_data["wall_segments_from_openings"]
    wall_segments_from_junctions = json_data["wall_segments_from_junctions"]
    openings = json_data["openings"]
    pillars = json_data.get("pillar_squares", [])
    
    # Создаём фигуру
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Определяем цвета
    wall_color = 'lightgray'
    door_color = 'green'
    window_color = 'skyblue'
    pillar_color = 'brown'
    
    # Собираем все вершины и грани
    all_vertices = []
    all_faces = []
    face_colors = []
    
    # Добавляем стены
    for segment in wall_segments_from_openings:
        vertices, faces = create_wall_mesh(segment, wall_height, scale_factor)
        start_idx = len(all_vertices)
        all_vertices.extend(vertices)
        
        for face in faces:
            all_faces.append([idx + start_idx for idx in face])
            face_colors.append(wall_color)
    
    for segment in wall_segments_from_junctions:
        vertices, faces = create_wall_mesh(segment, wall_height, scale_factor)
        start_idx = len(all_vertices)
        all_vertices.extend(vertices)
        
        for face in faces:
            all_faces.append([idx + start_idx for idx in face])
            face_colors.append(wall_color)
    
    # Добавляем проёмы
    for opening in openings:
        vertices, faces = create_opening_mesh(opening, wall_height, scale_factor)
        start_idx = len(all_vertices)
        all_vertices.extend(vertices)
        
        color = door_color if opening["type"] == "door" else window_color
        
        for face in faces:
            all_faces.append([idx + start_idx for idx in face])
            face_colors.append(color)
    
    # Добавляем колонны
    for pillar in pillars:
        vertices, faces = create_pillar_mesh(pillar, wall_height, scale_factor)
        start_idx = len(all_vertices)
        all_vertices.extend(vertices)
        
        for face in faces:
            all_faces.append([idx + start_idx for idx in face])
            face_colors.append(pillar_color)
    
    # Визуализируем грани с соответствующими цветами
    for i, face in enumerate(all_faces):
        poly = [all_vertices[idx] for idx in face]
        color = face_colors[i]
        
        # Создаём коллекцию для одной грани
        face_collection = Poly3DCollection([poly], alpha=0.9, facecolor=color, edgecolor='k')
        ax.add_collection3d(face_collection)
    
    # Устанавливаем пределы осей с изменённым соотношением
    all_vertices = np.array(all_vertices)
    x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
    
    # Инвертируем ось X, чтобы она строилась вправо
    ax.set_xlim([x_max, x_min])  # Инвертированный порядок для оси X
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([0, wall_height])
    
    # Изменяем соотношение осей для более плоского вида
    ax.set_box_aspect([1, 1, z_scale_factor])  # Z-ось с настраиваемым коэффициентом
    
    # Добавляем метки осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Устанавливаем угол обзора для лучшей визуализации (вид сверху)
    ax.view_init(elev=80, azim=45)
    
    # Добавляем легенду
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=wall_color, alpha=0.7, label='Стены'),
        Patch(facecolor=door_color, alpha=0.9, label='Двери'),
        Patch(facecolor=window_color, alpha=0.9, label='Окна'),
        Patch(facecolor=pillar_color, alpha=0.9, label='Колонны')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title('3D модель стен с раскрашенными элементами')
    plt.tight_layout()
    plt.savefig('png_from_json.png', dpi=300)
    plt.show()

def main():
    # Загружаем данные из JSON
    json_file = 'blender/wall_coordinates.json'
    json_data = load_json_data(json_file)
    
    # Получаем статистику
    wall_segments_from_openings = json_data["wall_segments_from_openings"]
    wall_segments_from_junctions = json_data["wall_segments_from_junctions"]
    openings = json_data["openings"]
    pillars = json_data.get("pillar_squares", [])
    
    print(f"Сегментов стен от проемов: {len(wall_segments_from_openings)}")
    print(f"Сегментов стен от соединений: {len(wall_segments_from_junctions)}")
    print(f"Проемов (окон и дверей): {len(openings)}")
    print(f"Окон: {sum(1 for o in openings if o['type'] == 'window')}, Дверей: {sum(1 for o in openings if o['type'] == 'door')}")
    print(f"Колонн: {len(pillars)}")
    
    # Визуализируем модель
    visualize_3d_model_from_json(json_data)

if __name__ == "__main__":
    main()