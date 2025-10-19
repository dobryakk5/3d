# Обновленный план исправления проблемы с ориентацией junctions

## Описание проблемы

Дверь D5 (горизонтальная) и стена W34 (вертикальная) пересекаются. Проблема в том, что стена W34v создается не из junction J19, а из полигонных стен, которые проходят рядом с дверью.

## Корень проблемы

1. Junction J19 (x=280, y=616) находится очень близко к двери D5 (x=278, y=605)
2. Но стена W34v создается не из этого junction, а из полигонных стен
3. Полигоны стен обрабатываются независимо от ориентации ближайших проемов

## Решение

Нужно усилить проверку совместимости ориентации не только при создании сегментов от junctions, но и при создании сегментов из полигонов:

1. **Определить ориентацию ближайшего проема к каждой вершине полигона**
2. **Ограничить ориентацию сегментов стен, выходящих из вершин полигона, ориентацией ближайшего проема**
3. **Если рядом с вершиной полигона нет проемов, оставить текущую логику без изменений**

## План реализации

### Шаг 1: Модификация функции split_polygon_into_segments

Добавить проверку ориентации ближайшего проема при создании сегментов из полигонов:

```python
def split_polygon_into_segments(polygon_vertices: List[Dict], 
                              used_opening_sides: Dict[str, set] = None,
                              used_junctions: set = None,
                              junctions: List[JunctionPoint] = None,
                              openings: List[Dict] = None) -> Tuple[List[WallSegment], Dict[str, set], set]:
    # ... существующий код ...
    
    # При обработке любой вершины:
    if vertex_type in ['junction', 'opening', 'vertex'] and i > 0:
        # НОВОЕ: Находим ближайший проем к текущей вершине
        nearest_opening = find_nearest_opening_to_vertex(vertex, openings)
        
        # Если найден ближайший проем, проверяем совместимость ориентации
        if nearest_opening:
            opening_orientation = get_opening_orientation(nearest_opening)
            segment_orientation = analyze_segment_orientation(segment_vertices)
            
            # Если ориентации не совпадают, пропускаем создание сегмента
            if opening_orientation != segment_orientation:
                segment_vertices = [vertex]
                continue
```

### Шаг 2: Создание функции find_nearest_opening_to_vertex

```python
def find_nearest_opening_to_vertex(vertex: Dict, 
                                 openings: List[Dict], 
                                 threshold: float = 30.0) -> Optional[Dict]:
    """
    Находит ближайший проем к вершине
    
    Args:
        vertex: Вершина полигона
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
        
        # Вычисляем расстояние от вершины до центра проема
        distance = math.sqrt((vertex['x'] - center_x)**2 + (vertex['y'] - center_y)**2)
        
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            nearest_opening = opening
    
    return nearest_opening
```

### Шаг 3: Модификация функции create_junction_aware_t_junctions

Оставить существующую логику без изменений, так как она уже работает правильно.

## Ожидаемый результат

После применения этого исправления:
1. От вершин полигонов, находящихся рядом с горизонтальной дверью D5, будут отходить только горизонтальные сегменты стен
2. Вертикальные сегменты стен не будут создаваться рядом с горизонтальной дверью
3. Дверь D5 и стена W34 больше не будут пересекаться

## Тестирование

1. Запустить исправленный код
2. Проверить, что дверь D5 и стена W34 больше не пересекаются
3. Убедиться, что от junction J19 отходят только горизонтальные сегменты стен
4. Проверить, что общее количество сегментов стен уменьшилось