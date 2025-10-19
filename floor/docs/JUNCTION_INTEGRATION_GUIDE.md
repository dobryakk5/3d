# Junction-Aware Vectorization - Integration Guide

## Проблема T-Junction - Решена!

### Старый подход (`cubicasa_vectorize.py`)
```
Порядок обработки:
1. Извлечь junctions из heatmap
2. Извлечь стены из контуров скелета
3. ❌ Стены и junctions НЕ СВЯЗАНЫ!

Результат: Стена проходит "насквозь" через T-junction вместо примыкания
```

### Новый подход (`cubicasa_vectorize_with_junctions.py`)
```
Порядок обработки:
1. Извлечь junctions из heatmap
2. Построить граф стен ЧЕРЕЗ junctions
3. ✅ Стены начинаются и заканчиваются ТОЧНО в junctions!

Результат: Стена примыкает к T-junction правильно
```

---

## Ключевые изменения

### 1. Новая функция: `extract_wall_segments_junction_aware()`

**Старая версия (cubicasa_vectorize.py:215):**
```python
def extract_wall_segments(wall_mask, min_length=50):
    """Extract wall line segments"""
    # Простое трассирование контуров
    skeleton = binary_erosion(wall_mask)
    contours = cv2.findContours(skeleton)

    # Упрощение
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon=3.0)
        # Каждый сегмент независим!
```

**Новая версия (cubicasa_vectorize_with_junctions.py:271):**
```python
def extract_wall_segments_junction_aware(wall_mask, junctions_dict, use_graph=True):
    """
    Extract wall segments using junction information.

    Если use_graph=True:
      1. Извлекает только wall junctions из heatmap
      2. Строит граф: вершины = junctions, рёбра = стены
      3. Проверяет связность по скелету (BFS)
      4. Возвращает сегменты, которые ТОЧНО соединены в junctions

    Если use_graph=False:
      - Fallback к старому методу (для сравнения)
    """
    # Только wall junctions
    junctions_list = []
    for jtype, points in junctions_dict.items():
        if 'wall' in jtype:  # T, X, L, endpoints
            junctions_list.append(points)

    # Граф
    graph = build_junction_graph(junctions_list, skeleton)

    # Сегменты из графа
    segments = extract_wall_segments_from_graph(graph, junctions_list)
    return segments
```

### 2. Новая функция: `build_junction_graph()`

```python
def build_junction_graph(junctions_list, skeleton, max_distance=80):
    """
    Строит граф стен через junctions.

    Алгоритм:
    1. Для каждой пары junctions:
       - Проверяем расстояние < max_distance
       - Проверяем связность по скелету (BFS)
    2. Если связаны → добавляем ребро в граф

    Результат: граф, где каждое ребро = стена между двумя junctions
    """
```

### 3. Использование в main()

**Старая версия:**
```python
# Сначала стены
wall_segments = extract_wall_segments(wall_mask)

# Потом junctions (независимо!)
junctions = extract_junctions(prediction)
```

**Новая версия:**
```python
# Сначала junctions!
junctions = extract_junctions(prediction)

# Потом стены ЧЕРЕЗ junctions
wall_segments = extract_wall_segments_junction_aware(
    wall_mask,
    junctions,  # ← Передаём junctions!
    use_graph=True
)
```

---

## Как запустить

### 1. Быстрый тест (новый алгоритм)
```bash
cd /Users/pavellebedev/Desktop/3d/floor
python3 cubicasa_vectorize_with_junctions.py
```

Результат:
- `floor_plan_with_junctions.svg` - SVG с правильными T-junctions
- `floor_plan_junctions.json` - Метаданные

### 2. Сравнение с контрольным

```bash
# Старый метод (без graph)
python3 cubicasa_vectorize.py
# → floor_plan.svg (T-junctions неправильные)

# Новый метод (с graph)
python3 cubicasa_vectorize_with_junctions.py
# → floor_plan_with_junctions.svg (T-junctions правильные!)
```

### 3. Параметры

В `main()` можно настроить:

```python
# Включить/выключить junction-aware extraction
use_junction_graph = True  # False = старый метод

# Максимальное расстояние для поиска соседних junctions
max_distance = 80  # px (увеличить если junctions далеко)

# Порог для обнаружения junctions
threshold_doors = 0.3  # Понизить = больше junctions
```

---

## Визуальное сравнение

### До (Старый метод - `cubicasa_vectorize.py`)
```
    │ Красная стена
    │
────┼──── Зелёный door/window
    │
    │ ❌ Стена проходит СКВОЗЬ, не прерывается
    │
```

### После (Новый метод - `cubicasa_vectorize_with_junctions.py`)
```
    │ Красная стена
    │
    ● ← T-junction (красная точка на SVG)
────┼──── Зелёный door/window
    ● ← T-junction
    │
    │ ✅ Стена примыкает ТОЧНО в T-junction!
```

---

## Технические детали

### Граф стен

**Вершины (Vertices):**
- Каждая вершина = junction point из heatmap
- Атрибуты: `{x, y, type, confidence}`
- Типы: `wall_junction_3way`, `wall_junction_4way`, `wall_corner_90deg`, `wall_endpoint`

**Рёбра (Edges):**
- Каждое ребро = стена между двумя junctions
- Атрибуты: `{weight: distance}`
- Создаётся, только если:
  1. Расстояние < `max_distance` (по умолчанию 80px)
  2. Есть непрерывный путь по скелету (BFS проверка)

### Проверка связности (BFS)

```python
def is_connected_by_skeleton(skeleton, point1, point2, max_distance):
    """
    BFS по пикселям скелета.

    Проверяет, что между point1 и point2:
    1. Есть непрерывная линия скелета
    2. Длина пути ≤ max_distance * 1.5

    Это гарантирует, что ребро графа соответствует реальной стене!
    """
```

---

## Производительность

### Старый метод
- Скорость: **Быстрее** (~1-2 сек для стен)
- Точность: ❌ T-junctions неправильные

### Новый метод
- Скорость: **Медленнее** (~5-10 сек для стен, из-за BFS)
- Точность: ✅ T-junctions правильные!

### Оптимизация (если нужно)

1. **Уменьшить `max_distance`:**
   ```python
   max_distance = 60  # Меньше пар для проверки
   ```

2. **Использовать только важные junctions:**
   ```python
   # Фильтровать по confidence
   if junc['confidence'] > 0.5:  # Только уверенные
       junctions_list.append(junc)
   ```

3. **Кэшировать skeleton:**
   ```python
   # Если wall_mask не меняется, skeleton можно переиспользовать
   ```

---

## Отладка

### Если получается мало стен

```python
# Увеличить max_distance
max_distance = 100  # или 120

# Понизить threshold
threshold_doors = 0.2  # Больше junctions
```

### Если получается слишком много стен

```python
# Уменьшить max_distance
max_distance = 60  # Только близкие junctions

# Повысить threshold
threshold_doors = 0.4  # Только уверенные junctions
```

### Посмотреть граф

```python
# После build_junction_graph():
print(f"Nodes: {list(graph.nodes(data=True))}")
print(f"Edges: {list(graph.edges(data=True))}")
```

---

## Следующие шаги

### 1. Выпрямление стен (опционально)

После получения правильных T-junctions, можно применить:
```bash
python3 straighten_walls_smart.py floor_plan_with_junctions.svg floor_plan_final.svg
```

Это сделает стены строго горизонтальными/вертикальными, **сохраняя T-junctions**.

### 2. Интеграция в основной pipeline

Заменить в `cubicasa_vectorize.py`:
```python
# БЫЛО:
wall_segments = extract_wall_segments(wall_mask)

# СТАЛО:
wall_segments = extract_wall_segments_junction_aware(
    wall_mask, junctions, use_graph=True
)
```

### 3. Добавление в экспорт

В `export_objects.py` добавить поле `junction_type`:
```python
wall_obj = {
    'type': 'wall',
    'start': segment['start'],
    'end': segment['end'],
    'start_junction': segment['start_type'],  # NEW!
    'end_junction': segment['end_type']        # NEW!
}
```

---

## Резюме

| Характеристика | Старый метод | Новый метод |
|---|---|---|
| **T-junctions** | ❌ Неправильные | ✅ Правильные |
| **Скорость** | ~2 сек | ~7 сек |
| **Точность топологии** | Низкая | Высокая |
| **Использование DL** | Только стены | Стены + junctions |
| **Файл** | `cubicasa_vectorize.py` | `cubicasa_vectorize_with_junctions.py` |

**Рекомендация:** Использовать новый метод для финальной векторизации!
