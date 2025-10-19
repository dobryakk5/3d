# План исправления проблемы с window_4 и door_2

## Описание проблемы

Проемы window_4 и door_2 находятся на расстоянии 1px друг от друга, но система все равно создает стену между ними, потому что логика проверки "находится ли junction между проемами" работает некорректно.

## Координаты объектов

- window_4: x=622, width=97 (правый край: 719)
- door_2: x=720, width=87 (левый край: 720)
- Junction 17: (718, 150) - находится на 1px слева от правого края window_4
- Junction 24: (806, 150) - находится внутри door_2
- Толщина стены: 22px (половина толщины: 11px)

## Правильная логика обработки

1. Сначала проверить, находится ли junction в пределах половины толщины стены от края проема
2. Если да - сдвинуть край проема до junction (расширить или сузить в зависимости от положения junction)
3. Потом проверить, есть ли рядом другой проем
4. Если рядом есть другой проем - не создавать стену
5. Если рядом нет другого проема - создать стену до следующего junction

## Необходимые изменения в функции construct_wall_segment_from_opening

### Изменение 1: Проверка расстояния до junction

Заменить текущую логику проверки "находится ли junction между проемами" на проверку расстояния от junction до края проема:

```python
# Вместо проверки "находится ли junction между проемами"
if direction == 'right':
    # Проверяем, находится ли junction в пределах половины толщины стены от края проема
    junction_near_edge = abs(nearest_junction.x - (bbox['x'] + bbox['width'])) <= wall_thickness / 2
elif direction == 'left':
    # Проверяем, находится ли junction в пределах половины толщины стены от края проема
    junction_near_edge = abs(nearest_junction.x - bbox['x']) <= wall_thickness / 2
# И аналогично для up/down
```

### Изменение 2: Сдвиг края проема до junction

Если junction находится близко к краю проема, сдвинуть край проема до junction:

```python
if junction_near_edge:
    # Сдвигаем край проема до junction
    print(f"    Сдвигаем край проема {opening_id} до junction {nearest_junction.id} в направлении {direction}")
    # Изменяем bbox проема, чтобы его край доходил до junction
    adjusted_bbox = extend_opening_to_junction(
        {'id': opening_id, 'bbox': bbox},
        direction,
        nearest_junction
    )
    # Обновляем bbox проема
    opening_with_junction.bbox = adjusted_bbox
    print(f"    Проем {opening_id} изменен: {bbox} -> {adjusted_bbox}")
```

### Изменение 3: Проверка наличия соседнего проема

После сдвига края проема проверить, есть ли соседний проем:

```python
# Проверяем, есть ли рядом другой проем
nearby_opening = find_nearby_opening_in_direction(
    {'id': opening_id, 'bbox': adjusted_bbox},
    direction,
    all_openings,
    wall_thickness
)

if nearby_opening:
    print(f"    Найден соседний проем {nearby_opening.get('id')}, стена не создается")
    continue  # Не создаем стену, если есть соседний проем
else:
    print(f"    Соседний проем не найден, создаем стену")
    # Создаем стену до следующего junction
```

## Ожидаемый результат

После этих изменений:
1. Край window_4 будет сдвинут до Junction 17 (x=718)
2. Край door_2 будет сдвинут до Junction 24 (x=806)
3. Система определит, что проемы находятся рядом, и не создаст стену между ними

## Файлы для изменения

- floor/visualize_polygons.py
  - Функция construct_wall_segment_from_opening (строки 455-624)
  - Изменить логику обработки junctions и соседних проемов