# План исправления проблемы с ориентацией junctions

## Описание проблемы

Дверь D5 (горизонтальная) и стена W34 (вертикальная) пересекаются в junction J19, что некорректно. Проблема в том, что junction J19 находится рядом с горизонтальной дверью D5, но от этого junction отходят сегменты стен с разной ориентацией, включая вертикальную стену W34.

## Корень проблемы

1. Junction J19 (x=280, y=616) находится очень близко к двери D5 (x=278, y=605)
2. Junction обрабатывается как обычный junction, без учета того, что рядом с ним находится горизонтальная дверь
3. В результате от этого junction отходят сегменты стен с несовместимой ориентацией

## Решение

Нужно усилить проверку совместимости ориентации при создании сегментов от junctions:

1. **Определить ориентацию ближайшего проема к junction**
2. **Ограничить ориентацию сегментов стен, выходящих из junction, ориентацией ближайшего проема**
3. **Если рядом с junction нет проемов, оставить текущую логику без изменений**

## План реализации

### Шаг 1: Создание функции определения ближайшего проема к junction

```python
def find_nearest_opening_to_junction(junction: JunctionPoint, 
                                   openings: List[Dict], 
                                   threshold: float = 30.0) -> Optional[Dict]:
    """
    Находит ближайший проем к junction
    
    Args:
        junction: Точка junction
        openings: Список всех проемов
        threshold: Максимальное расстояние для определения соседства
    
    Returns:
        Ближайший проем или None, если проем не найден
    """
    nearest_opening = None
    min_distance = float('inf')
    
    for opening in openings:
        bbox = opening.get('bbox', {})
        if not bbox:
            continue
        
        # Вычисляем центр проема
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        
        # Вычисляем расстояние от junction до центра проема
        distance = math.sqrt((junction.x - center_x)**2 + (junction.y - center_y)**2)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            nearest_opening = opening
    
    return nearest_opening
```

### Шаг 2: Создание функции определения ориентации проема

```python
def get_opening_orientation(opening: Dict) -> str:
    """
    Определяет ориентацию проема
    
    Args:
        opening: Проем с bbox
    
    Returns:
        'horizontal' или 'vertical'
    """
    bbox = opening.get('bbox', {})
    width = bbox.get('width', 0)
    height = bbox.get('height', 0)
    
    return 'horizontal' if width > height else 'vertical'
```

### Шаг 3: Модификация функции split_polygon_into_segments

Добавить проверку ориентации ближайшего проема при создании сегментов от junction:

```python
def split_polygon_into_segments(polygon_vertices: List[Dict], 
                              used_opening_sides: Dict[str, set] = None,
                              used_junctions: set = None,
                              junctions: List[JunctionPoint] = None,
                              openings: List[Dict] = None) -> Tuple[List[WallSegment], Dict[str, set], set]:
    # ... существующий код ...
    
    # При обработке junction:
    if vertex_type == 'junction':
        # Находим ближайший проем к junction
        nearest_opening = find_nearest_opening_to_junction(junction, openings)
        
        # Если найден ближайший проем, проверяем совместимость ориентации
        if nearest_opening:
            opening_orientation = get_opening_orientation(nearest_opening)
            segment_orientation = analyze_segment_orientation(segment_vertices)
            
            # Если ориентации не совпадают, пропускаем создание сегмента
            if opening_orientation != segment_orientation:
                segment_vertices = [vertex]
                continue
```

### Шаг 4: Модификация функции create_junction_aware_t_junctions

Добавить проверку ориентации ближайшего проема при создании T-соединений:

```python
def create_junction_aware_t_junctions(segments: List[Dict],
                                     junctions: List[JunctionPoint],
                                     wall_thickness: float,
                                     openings: List[Dict] = None) -> List[Dict]:
    # ... существующий код ...
    
    # При обработке junction:
    for junction in junctions_to_process:
        # Находим ближайший проем к junction
        nearest_opening = find_nearest_opening_to_junction(junction, openings)
        
        # Если найден ближайший проем, ограничиваем ориентацию сегментов
        if nearest_opening:
            opening_orientation = get_opening_orientation(nearest_opening)
            
            # Фильтруем сегменты по ориентации
            connected_segments = [s for s in connected_segments 
                                if s['orientation'] == opening_orientation]
```

## Ожидаемый результат

После применения этого исправления:
1. От junction J19 будут отходить только горизонтальные сегменты стен
2. Вертикальная стена W34 не будет создаваться от этого junction
3. Дверь D5 и стена W34 больше не будут пересекаться

## Тестирование

1. Запустить исправленный код
2. Проверить, что дверь D5 и стена W34 больше не пересекаются
3. Убедиться, что от junction J19 отходят только горизонтальные сегменты стен
4. Проверить, что общее количество сегментов стен уменьшилось