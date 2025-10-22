# План реализации отображения номеров junctions в Blender

## Обзор
Необходимо добавить функциональность отображения номеров junctions в Blender над стенами для визуальной отслеживания привязки стен к junctions, а также убрать отдельную желтую контурную стену, которая строится по координатам.

## Текущее состояние
В файле `create_walls_2m.py` уже существует функция `create_junction_labels()` (строки 203-292), которая создает текстовые метки с номерами junctions, но она не вызывается в основной функции `create_3d_walls_from_json()`.

## План изменений

### 1. Модификация функции `create_3d_walls_from_json()`

#### 1.1. Добавление нового параметра
Добавить опциональный параметр `show_junction_labels` в сигнатуру функции:

```python
def create_3d_walls_from_json(json_path, wall_height=2.0, export_obj=True, clear_scene=True, 
                             brick_texture_path=None, render_isometric=False, 
                             show_junction_labels=True):
```

#### 1.2. Обновление документации функции
Обновить docstring для описания нового параметра:

```python
"""
Основная функция для создания 3D стен из JSON файла

Args:
    json_path (str): Путь к JSON файлу с координатами стен
    wall_height (float): Высота стен в метрах (по умолчанию 2.0)
    export_obj (bool): Экспортировать результат в OBJ файл (по умолчанию True)
    clear_scene (bool): Очищать сцену перед созданием стен (по умолчанию True)
    brick_texture_path (str): Путь к файлу текстуры кирпича для внешних стен
    render_isometric (bool): Создавать изометрический рендер (по умолчанию False)
    show_junction_labels (bool): Отображать номера junctions над стенами (по умолчанию True)

Returns:
    bool: True если успешно, False в случае ошибки
"""
```

#### 1.3. Добавление вызова функции
Добавить вызов `create_junction_labels()` после создания колонн (примерно строка 1835) и перед инверсией координат:

```python
# Создаем метки с номерами junctions если включено
if show_junction_labels:
    print("Создание меток с номерами junctions...")
    junction_label_objects = create_junction_labels(junctions, scale_factor, wall_height_meters)
    print(f"Создано меток junctions: {len(junction_label_objects)}")
else:
    print("Отображение номеров junctions отключено")
```

### 2. Модификация основного блока выполнения

#### 2.1. Обновление вызова функции в __main__
В конце файла (строка ~2080) обновить вызов функции:

```python
# Создаем 3D стены с указанной высотой и текстурой кирпича для внешних стен
# Также создаем изометрический рендер и отображаем номера junctions
create_3d_walls_from_json(json_path, wall_height=3.0, export_obj=True,
                         brick_texture_path=brick_texture_path, render_isometric=True,
                         show_junction_labels=True)
```

### 3. Улучшения функции `create_junction_labels()`

#### 3.1. Опционально: Улучшение видимости меток
Можно увеличить размер шрифта и улучшить материал для лучшей видимости:

```python
# Увеличиваем размер шрифта для лучшей видимости
text_obj.data.size = 0.5  # Увеличить с 0.3 до 0.5

# Улучшаем материал для лучшей контрастности
text_material.diffuse_color = (1.0, 1.0, 0.0, 1.0)  # Желтый цвет вместо белого
```

## Ожидаемый результат

После реализации изменений:
1. При вызове `create_3d_walls_from_json()` с параметром `show_junction_labels=True` (по умолчанию) будут создаваться текстовые метки с номерами junctions
2. Метки будут отображаться как 3D-текст на высоте `wall_height + 0.5` метров над каждым junction
3. Каждая метка будет иметь уникальное имя `Junction_Label_{junction_id}`
4. Метки будут экспортироваться вместе с другими объектами в OBJ файл
5. Параметр `show_junction_labels=False` позволит отключить создание меток при необходимости

## Тестирование

После внесения изменений рекомендуется протестировать:
1. Создание стен с включенными метками junctions
2. Создание стен с выключенными метками junctions
3. Корректность позиционирования меток относительно стен
4. Экспорт в OBJ файл с метками
5. Изометрический рендер с метками

## Места в коде для изменений

1. **Строка ~1623**: Сигнатура функции `create_3d_walls_from_json()`
2. **Строка ~1635**: Docstring функции
3. **Строка ~1835**: Добавление вызова `create_junction_labels()`
4. **Строка ~1879-1901**: Удаление кода создания контурной стены (блок с create_outline_wall)
5. **Строка ~2080**: Обновление вызова в __main__ блоке
6. **Строка ~2083-2104**: Удаление кода создания контурной стены в __main__ блоке

### 4. Удаление контурной стены

#### 4.1. Удаление в функции экспорта
Удалить блок кода (строки ~1879-1901), который создает контурную стену для отладки:

```python
# УДАЛИТЬ ЭТОТ БЛОК:
# Создаем контурную стену для отладки ПЕРЕД экспортом
outline_wall = None
try:
    if data and "building_outline" in data:
        building_outline = data["building_outline"]
        outline_vertices = building_outline["vertices"]
        
        # Создаем контурную стену с увеличенной высотой для лучшей видимости
        outline_wall = create_outline_wall(outline_vertices, wall_height_meters + 1.0, scale_factor)
        if outline_wall:
            print(f"Контурная стена создана для отладки: {outline_wall.name}")
            # Добавляем контурную стену к выделению
            outline_wall.select_set(True)
            print("Контурная стена добавлена к экспорту")
except Exception as e:
    print(f"Ошибка при создании контурной стены: {e}")
```

#### 4.2. Удаление в __main__ блоке
Удалить блок кода (строки ~2083-2104), который создает контурную стену:

```python
# УДАЛИТЬ ЭТОТ БЛОК:
# Дополнительно создаем контурную стену для отладки ПЕРЕД экспортом
outline_wall = None
try:
    data = load_wall_coordinates(json_path)
    if data and "building_outline" in data:
        building_outline = data["building_outline"]
        outline_vertices = building_outline["vertices"]
        scale_factor = 0.01  # 1 пиксель = 0.01 метра
        wall_height = 3.0
        
        # Создаем контурную стену с увеличенной высотой для лучшей видимости
        outline_wall = create_outline_wall(outline_vertices, wall_height + 1.0, scale_factor)
        if outline_wall:
            print(f"Контурная стена создана для отладки: {outline_wall.name}")
            print(f"Тип объекта: {outline_wall.type}")
            print(f"Количество вершин: {len(outline_wall.data.vertices)}")
            print(f"Количество полигонов: {len(outline_wall.data.polygons)}")
            print(f"Положение: {outline_wall.location}")
        else:
            print("ОШИБКА: Контурная стена не создана")
except Exception as e:
    print(f"Ошибка при создании контурной стены: {e}")
```

## Примечания

- Функция `create_junction_labels()` уже реализована и требует только вызова
- Метки создаются с белым материалом и эмиссией для лучшей видимости
- Высота позиционирования меток рассчитывается как `wall_height + 0.5` метров
- Все метки добавляются в коллекцию объектов Blender
- Контурная желтая стена будет полностью удалена из кода для упрощения визуализации
- Функция `create_outline_wall()` может остаться в коде, но не будет вызываться