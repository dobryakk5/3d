# Решение проблемы T-junction при векторизации

## Проблема
На скриншоте видно, что красная стена (wall) проходит "насквозь" зелёный объект (door/window), вместо того чтобы примыкать к нему в точке T-junction.

## Почему это происходит?

### Текущий подход (НЕПРАВИЛЬНЫЙ):
```
1. Скелетизация стен → получаем тонкие линии
2. Трассировка контуров → получаем сегменты
3. Упрощение (approxPolyDP) → убираем лишние точки
```

**Проблема:** Алгоритм не знает где находятся T-junctions!
- Стена может "проскочить" через примыкание
- Или наоборот, разорваться где не нужно

### Правильный подход (С JUNCTIONS):

```
1. Извлечь junction points из heatmap модели
   ├─ wall_junction_3way (T) ← ВОТ ОНО!
   ├─ wall_junction_4way (X)
   ├─ wall_corner_90deg (L)
   └─ wall_endpoint

2. Построить граф стен:
   Vertices = junctions
   Edges = стены между junctions

3. Трассировать стены СТРОГО между junctions
   ├─ Начало сегмента = junction
   ├─ Конец сегмента = следующий junction
   └─ Путь = по скелету между ними

4. Результат: стены точно соединяются в junctions!
```

## Реализация

### Файл: `vectorize_with_graph.py`

Ключевые функции:

1. **extract_junctions_from_heatmap(heatmaps)**
   - Извлекает T/X/L junctions из каналов 0-3 heatmap
   - Возвращает координаты + тип каждого junction

2. **build_junction_graph(junctions, skeleton)**
   - Строит граф: вершины = junctions, рёбра = стены
   - Использует BFS по скелету для проверки связности

3. **extract_wall_segments_from_graph(graph)**
   - Каждое ребро графа = один сегмент стены
   - Гарантия: стены начинаются и заканчиваются в junctions

### Интеграция в `cubicasa_vectorize.py`:

```python
# Текущий код (строка 291-295):
junctions = extract_junctions(prediction, threshold=threshold_doors)
# ↓ junctions уже есть!

# ДОБАВИТЬ:
# Строим граф стен через junctions
from vectorize_with_graph import build_junction_graph, extract_wall_segments_from_graph

# Скелетизация
skeleton = skeletonize_walls(wall_mask_combined)

# Граф
graph = build_junction_graph(junctions_list, skeleton, max_distance=100)

# Новые сегменты - точно соединённые!
wall_segments = extract_wall_segments_from_graph(graph, junctions_list)
```

## Пример работы алгоритма

```
Исходная ситуация:
    │
    │ стена
    │
────┼──── T-junction (определён из heatmap!)
    │
    │ стена
    │

1. Модель предсказывает: junction в точке (x, y) типа "wall_junction_3way"

2. Граф:
   Vertex A (endpoint)
   │
   Edge1 (стена)
   │
   Vertex B (T-junction) ← координаты точно из heatmap!
   │
   Edge2 (стена)
   │
   Vertex C (endpoint)

3. Результат: стена ТОЧНО примыкает в T-junction!
```

## Преимущества

✅ **Точность:** Стены соединяются СТРОГО в junctions
✅ **Топология:** Правильная структура T/X/L соединений
✅ **Надёжность:** Используется информация из DL модели
✅ **Расширяемость:** Легко добавить другие типы junctions

## Следующие шаги

1. Добавить функции из `vectorize_with_graph.py` в `cubicasa_vectorize.py`
2. Заменить текущий `extract_wall_segments()` на graph-based подход
3. Протестировать на проблемных случаях (T-junctions)
4. Опционально: добавить постобработку для выпрямления стен

## Альтернатива (если нет модели)

Если нет доступа к heatmap модели:

1. **Определять junctions из геометрии:**
   - Найти точки пересечения скелетов
   - Проверить валентность (сколько рёбер сходится)
   - Классифицировать: 2=endpoint, 3=T, 4=X

2. **Snap стен к объектам:**
   - Если конец стены близко к door/window → примкнуть
   - Использовать tolerance ~15-20 пикселей

Но это менее надёжно, чем использование heatmap!
