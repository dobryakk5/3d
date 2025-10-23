# План модификации логики выбора junction для outbound

## Текущая проблема
В файле `export_objects.py` функция `main()` использует вторую ближайшую junction к углу фундамента для создания street объекта. Нужно изменить логику, чтобы выбиралась junction, с которой начинается outbound.

## Определение outbound
Outbound junction - это ПЕРВЫЙ junction, который используется для построения outline (контура здания).

## Предлагаемое решение

### 1. Новая функция `find_outbound_junction`

```python
def find_outbound_junction(junctions, building_outline, foundation):
    """
    Находит ПЕРВЫЙ junction, который используется для построения outline
    
    Args:
        junctions: Список всех junctions
        building_outline: Контур здания с вершинами
        foundation: Фундамент здания
        
    Returns:
        dict: Найденная outbound junction или None
    """
    if not junctions or not building_outline:
        return None
    
    # 1. Получаем ПЕРВУЮ junction из building outline
    if building_outline and building_outline.get('vertices'):
        # Берем первую вершину building outline
        first_vertex = building_outline['vertices'][0]
        
        # Если у вершины есть информация о junction, возвращаем её
        if 'junction_id' in first_vertex:
            # Находим junction по ID
            for junction in junctions:
                if junction.get('id') == first_vertex['junction_id']:
                    print(f"   Found first outline junction: ID={junction.get('id')}, coords=({junction['x']:.1f}, {junction['y']:.1f})")
                    return junction
        
        # Если у первой вершины нет junction_id, ищем ближайшую junction к ней
        first_point = {'x': first_vertex['x'], 'y': first_vertex['y']}
        nearest_junction = find_nearest_junction_to_point(first_point, junctions)
        if nearest_junction:
            print(f"   Found nearest junction to first outline vertex: ID={nearest_junction.get('id')}, coords=({nearest_junction['x']:.1f}, {nearest_junction['y']:.1f})")
            return nearest_junction
    
    # 2. Если не нашли в building outline, используем первую junction из списка
    if junctions:
        first_junction = junctions[0]
        print(f"   Using first junction from list: ID={first_junction.get('id')}, coords=({first_junction['x']:.1f}, {first_junction['y']:.1f})")
        return first_junction
    
    return None
```

### 2. Модификация функции `main()`

В функции `main()` нужно заменить строки 2696-2735:

```python
# Create street object
print("\n   Creating street object...")
try:
    if json_data["foundation"] and json_data["junctions"]:
        # 1. Находим outbound junction (ПЕРВАЯ junction из building outline)
        outbound_junction = find_outbound_junction(
            json_data["junctions"], 
            json_data["building_outline"],
            json_data["foundation"]
        )
        
        if outbound_junction:
            print(f"   Outbound junction found: ({outbound_junction['x']:.1f}, {outbound_junction['y']:.1f})")
            
            # 2. Находим ближайший край фундамента к outbound junction
            foundation_edge, distance, projection, t = find_closest_foundation_edge(
                outbound_junction, json_data["foundation"])
            if foundation_edge:
                print(f"   Closest foundation edge found, distance: {distance:.1f}px")
                
                # 3. Создаем объект street с тремя точками
                street_object = create_street_object(outbound_junction, foundation_edge, wall_thickness_pixels)
                if street_object:
                    json_data["street"] = street_object
                    print(f"   Street object created with wall thickness: {wall_thickness_pixels:.1f}px")
                    print(f"   Outbound junction: ({street_object['junction']['x']:.1f}, {street_object['junction']['y']:.1f})")
                    print(f"   Street point: ({street_object['street_point']['x']:.1f}, {street_object['street_point']['y']:.1f})")
                    print(f"   Home point: ({street_object['home_point']['x']:.1f}, {street_object['home_point']['y']:.1f})")
                else:
                    print("   Failed to create street object")
            else:
                print("   No foundation edge found")
        else:
            print("   No outbound junction found")
    else:
        print("   No foundation or junctions available for street creation")
except Exception as e:
    print(f"   Error creating street object: {e}")
```

## Преимущества нового подхода

1. **Логическая обоснованность**: Выбирается ПЕРВАЯ junction из контура здания
2. **Последовательность**: Outbound junction соответствует начальной точке построения outline
3. **Простота**: Не требует сложных вычислений расстояний

## Тестирование

После внедрения изменений необходимо проверить:
1. Корректность определения первой junction из building outline
2. Правильность создания street объекта
3. Визуальное соответствие в SVG файле

## Обратная совместимость

Если building_outline не содержит информации о junctions, система вернется к использованию первой junction из списка, обеспечивая работоспособность кода.