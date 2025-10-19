# План модификации visualize_polygons.py для хранения координат в JSON

## Обзор задачи

Модифицировать скрипт `visualize_polygons.py` для хранения всех координат в JSON формате с последующим построением SVG на основе этих данных.

## Структура JSON для хранения данных

```json
{
  "metadata": {
    "source_file": "plan_floor1_objects.json",
    "wall_thickness": 20.0,
    "timestamp": "2025-10-18T07:30:00.000Z",
    "version": "1.0"
  },
  "junctions": [
    {
      "id": 1,
      "x": 780.0,
      "y": 618.0,
      "junction_type": "wall_junction_3way",
      "detected_type": "T",
      "directions": ["left", "up", "down"],
      "confidence": 1.0
    }
  ],
  "wall_segments_from_openings": [
    {
      "segment_id": "wall_window_2_left_1_to_2",
      "opening_id": "window_2",
      "edge_side": "left",
      "start_junction_id": 1,
      "end_junction_id": 2,
      "orientation": "horizontal",
      "bbox": {
        "x": 100.0,
        "y": 200.0,
        "width": 150.0,
        "height": 20.0
      },
      "alignment_info": {}
    }
  ],
  "wall_segments_from_junctions": [
    {
      "segment_id": "wall_junction_1_to_2_left",
      "start_junction_id": 1,
      "end_junction_id": 2,
      "direction": "left",
      "orientation": "horizontal",
      "bbox": {
        "x": 100.0,
        "y": 200.0,
        "width": 150.0,
        "height": 20.0
      }
    }
  ],
  "openings": [
    {
      "id": "window_2",
      "type": "window",
      "bbox": {
        "x": 286.0,
        "y": 139.0,
        "width": 156.0,
        "height": 22.0
      },
      "orientation": "horizontal",
      "edge_junctions": [
        {
          "edge_side": "left",
          "junction_id": 1
        }
      ]
    }
  ],
  "statistics": {
    "total_junctions": 45,
    "total_wall_segments_from_openings": 50,
    "total_wall_segments_from_junctions": 25,
    "total_openings": 13,
    "extended_segments": 5
  }
}
```

## Модификации кода

### 1. Добавление новых функций

#### Функция инициализации структуры JSON
```python
def initialize_json_data(input_path: str, wall_thickness: float) -> Dict:
    """Инициализирует структуру JSON для хранения данных"""
    return {
        "metadata": {
            "source_file": input_path,
            "wall_thickness": wall_thickness,
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0"
        },
        "junctions": [],
        "wall_segments_from_openings": [],
        "wall_segments_from_junctions": [],
        "openings": [],
        "statistics": {
            "total_junctions": 0,
            "total_wall_segments_from_openings": 0,
            "total_wall_segments_from_junctions": 0,
            "total_openings": 0,
            "extended_segments": 0
        }
    }
```

#### Функция добавления junction в JSON
```python
def add_junction_to_json(json_data: Dict, junction: JunctionPoint) -> None:
    """Добавляет junction в JSON структуру"""
    junction_data = {
        "id": junction.id,
        "x": junction.x,
        "y": junction.y,
        "junction_type": junction.junction_type,
        "detected_type": junction.detected_type,
        "directions": junction.directions,
        "confidence": junction.confidence
    }
    json_data["junctions"].append(junction_data)
    json_data["statistics"]["total_junctions"] += 1
```

#### Функция добавления сегмента стены от проема в JSON
```python
def add_wall_segment_from_opening_to_json(json_data: Dict, segment: WallSegmentFromOpening) -> None:
    """Добавляет сегмент стены от проема в JSON структуру"""
    segment_data = {
        "segment_id": segment.segment_id,
        "opening_id": segment.opening_id,
        "edge_side": segment.edge_side,
        "start_junction_id": segment.start_junction.id,
        "end_junction_id": segment.end_junction.id,
        "orientation": segment.orientation,
        "bbox": segment.bbox,
        "alignment_info": segment.bbox.get("alignment_info", {})
    }
    json_data["wall_segments_from_openings"].append(segment_data)
    json_data["statistics"]["total_wall_segments_from_openings"] += 1
```

#### Функция добавления сегмента стены между junctions в JSON
```python
def add_wall_segment_from_junction_to_json(json_data: Dict, segment: WallSegmentFromJunction) -> None:
    """Добавляет сегмент стены между junctions в JSON структуру"""
    segment_data = {
        "segment_id": segment.segment_id,
        "start_junction_id": segment.start_junction.id,
        "end_junction_id": segment.end_junction.id,
        "direction": segment.direction,
        "orientation": segment.orientation,
        "bbox": segment.bbox
    }
    json_data["wall_segments_from_junctions"].append(segment_data)
    json_data["statistics"]["total_wall_segments_from_junctions"] += 1
```

#### Функция добавления проема в JSON
```python
def add_opening_to_json(json_data: Dict, opening: Dict, edge_junctions: List[Tuple[str, JunctionPoint]]) -> None:
    """Добавляет проем в JSON структуру"""
    opening_data = {
        "id": opening.get('id', ''),
        "type": opening.get('type', ''),
        "bbox": opening.get('bbox', {}),
        "orientation": detect_opening_orientation(opening.get('bbox', {})),
        "edge_junctions": [
            {
                "edge_side": edge_side,
                "junction_id": junction.id
            }
            for edge_side, junction in edge_junctions
        ]
    }
    json_data["openings"].append(opening_data)
    json_data["statistics"]["total_openings"] += 1
```

#### Функция сохранения JSON данных в файл
```python
def save_json_data(json_data: Dict, output_path: str) -> None:
    """Сохраняет JSON данные в файл"""
    print(f"Сохранение данных в JSON: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ Данные успешно сохранены")
```

#### Функция построения SVG из JSON данных
```python
def create_svg_from_json(json_path: str, svg_output_path: str) -> None:
    """Создает SVG файл на основе сохраненных JSON данных"""
    print(f"Создание SVG из JSON: {json_path}")
    
    # Загружаем данные из JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Вычисляем размеры SVG
    max_x, max_y = 0, 0
    for segment in json_data["wall_segments_from_openings"]:
        bbox = segment["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
    
    for segment in json_data["wall_segments_from_junctions"]:
        bbox = segment["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
    
    for opening in json_data["openings"]:
        bbox = opening["bbox"]
        max_x = max(max_x, bbox["x"] + bbox["width"])
        max_y = max(max_y, bbox["y"] + bbox["height"])
    
    # Добавляем отступы
    padding = 50
    svg_width = int(max_x + padding * 2)
    svg_height = int(max_y + padding * 2)
    
    # Создаем SVG документ
    dwg = svgwrite.Drawing(svg_output_path, size=(svg_width, svg_height), profile='full')
    dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill='white'))
    
    # Определяем стили
    styles = define_styles()
    
    # Отрисовываем junctions
    junctions_group = dwg.add(dwg.g(id='junctions'))
    for junction in json_data["junctions"]:
        svg_x, svg_y = junction["x"] + padding, junction["y"] + padding
        junction_style = get_junction_style(junction["detected_type"])
        circle = dwg.circle(center=(svg_x, svg_y), r=5, **junction_style)
        junctions_group.add(circle)
        
        # Добавляем номер
        text = dwg.text(
            f"J{junction['id']}",
            insert=(svg_x + 10, svg_y - 5),
            text_anchor='start',
            fill='black',
            font_size='8px',
            font_weight='bold'
        )
        junctions_group.add(text)
    
    # Отрисовываем сегменты стен от junctions
    junction_walls_group = dwg.add(dwg.g(id='junction_based_walls'))
    junction_wall_style = {
        'stroke': '#FF6347',
        'stroke_width': 2,
        'fill': 'none',
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round'
    }
    
    for segment in json_data["wall_segments_from_junctions"]:
        bbox = segment["bbox"]
        x, y = bbox["x"] + padding, bbox["y"] + padding
        width, height = bbox["width"], bbox["height"]
        
        rect = dwg.rect(insert=(x, y), size=(width, height), **junction_wall_style)
        junction_walls_group.add(rect)
    
    # Отрисовываем сегменты стен от проемов
    opening_walls_group = dwg.add(dwg.g(id='opening_based_walls'))
    for segment in json_data["wall_segments_from_openings"]:
        bbox = segment["bbox"]
        x, y = bbox["x"] + padding, bbox["y"] + padding
        width, height = bbox["width"], bbox["height"]
        
        rect = dwg.rect(insert=(x, y), size=(width, height), **styles['wall'])
        opening_walls_group.add(rect)
    
    # Отрисовываем проемы
    openings_group = dwg.add(dwg.g(id='openings'))
    for opening in json_data["openings"]:
        bbox = opening["bbox"]
        opening_type = opening["type"]
        orientation = opening["orientation"]
        
        x, y = bbox["x"] + padding, bbox["y"] + padding
        width, height = bbox["width"], bbox["height"]
        
        # Определяем толщину
        wall_thickness = json_data["metadata"]["wall_thickness"]
        if orientation == 'horizontal':
            svg_height = wall_thickness
            y = y + (bbox["height"] - wall_thickness) / 2
        else:
            svg_width = wall_thickness
            x = x + (bbox["width"] - wall_thickness) / 2
        
        style = styles['window'] if opening_type == 'window' else styles['door']
        rect = dwg.rect(insert=(x, y), size=(width if orientation == 'horizontal' else wall_thickness, 
                                       height if orientation == 'vertical' else wall_thickness), **style)
        openings_group.add(rect)
    
    # Сохраняем SVG
    dwg.save()
    print(f"  ✓ SVG сохранен: {svg_output_path}")
```

### 2. Модификация основной функции

Изменить функцию `visualize_polygons_opening_based_with_junction_types()`:

1. Инициализировать JSON структуру в начале
2. Добавлять данные в JSON по мере их создания
3. Сохранять JSON в файл перед созданием SVG
4. Создавать SVG на основе JSON данных

### 3. Порядок модификации

1. Добавить новые функции в скрипт
2. Модифицировать функцию `process_openings_with_junctions` для сохранения данных в JSON
3. Модифицировать функцию `build_missing_wall_segments_for_junctions` для сохранения данных в JSON
4. Модифицировать основную функцию `visualize_polygons_opening_based_with_junction_types`
5. Добавить вызов функции сохранения JSON
6. Заменить прямое создание SVG на вызов `create_svg_from_json`

### 4. Имена файлов

- Входной JSON: `plan_floor1_objects.json`
- Выходной JSON с координатами: `wall_coordinates.json`
- Выходной SVG: `wall_polygons.svg`

## Преимущества подхода

1. **Разделение данных и визуализации**: Данные хранятся независимо от их визуализации
2. **Возможность повторного использования**: Данные можно использовать для других целей
3. **Отладка**: Легко проверить промежуточные результаты
4. **Модульность**: Функции визуализации отделены от функций обработки данных
5. **Гибкость**: Легко изменить стиль визуализации без изменения данных

## Тестирование

1. Запустить модифицированный скрипт
2. Проверить созданный JSON файл на полноту данных
3. Сравнить SVG файл с оригинальным
4. Убедиться, что все элементы отображаются корректно