# План улучшенной визуализации junctions

## Проблема

Текущий `junction_type_analyzer.py` неправильно определяет тип J15 как "unknown", в то время как `improved_junction_type_analyzer.py` правильно определяет его как T-junction с 3 ребрами (валентностью).

## Решение

Необходимо создать новый скрипт визуализации, который будет использовать логику из `improved_junction_type_analyzer.py` для определения типов всех junctions.

## Требования к новому скрипту

1. **Использовать `improved_junction_type_analyzer.py`** в качестве основы для определения типов junctions
2. **Создать функцию для анализа всех junctions** на план этажа
3. **Сгенерировать SVG visualization** с правильными типами junctions
4. **Добавить цветовую кодировку** для разных типов junctions:
   - L-junction: Красный
   - T-junction: Зеленый
   - X-junction: Синий
   - straight: Желтый
   - unknown: Серый

## План реализации

### Шаг 1: Создание функции анализа всех junctions

```python
def analyze_all_junctions_improved(data, wall_thickness=20.0):
    """
    Анализирует все junctions с использованием улучшенной логики
    
    Args:
        data: Данные плана с junctions и wall_polygons
        wall_thickness: Толщина стены для определения значимых расширений
    
    Returns:
        Список junctions с определенными типами
    """
    junctions = data.get('junctions', [])
    wall_polygons = data.get('wall_polygons', [])
    
    analyzed_junctions = []
    
    for idx, junction in enumerate(junctions):
        # Находим полигон, содержащий junction
        containing_polygon = None
        for wall in wall_polygons:
            vertices = wall.get('vertices', [])
            if vertices and is_point_in_polygon(junction['x'], junction['y'], vertices):
                containing_polygon = wall
                break
        
        if containing_polygon:
            # Определяем тип с учетом толщины стены
            junction_type = determine_junction_type_with_thickness(
                junction, containing_polygon['vertices'], wall_thickness
            )
            
            analyzed_junctions.append({
                'junction': junction,
                'type': junction_type,
                'polygon': containing_polygon
            })
        else:
            # Junction не внутри полигона
            analyzed_junctions.append({
                'junction': junction,
                'type': 'unknown',
                'polygon': None
            })
    
    return analyzed_junctions
```

### Шаг 2: Создание функции визуализации

```python
def create_junctions_svg(analyzed_junctions, input_svg, output_svg):
    """
    Создает SVG с визуализацией junctions
    
    Args:
        analyzed_junctions: Список проанализированных junctions
        input_svg: Путь к исходному SVG файлу
        output_svg: Путь для сохранения результата
    """
    # Цвета для разных типов junctions
    type_colors = {
        'L': '#ff0000',      # Красный
        'T': '#00ff00',      # Зеленый
        'X': '#0000ff',      # Синий
        'straight': '#ffff00',  # Желтый
        'unknown': '#888888'   # Серый
    }
    
    # Загружаем SVG
    tree = ET.parse(input_svg)
    root = tree.getroot()
    
    # Добавляем стили
    style = ET.SubElement(root, 'style')
    style.text = """
    .junction-label {
        font-family: Arial, sans-serif;
        font-size: 12px;
        font-weight: bold;
        text-anchor: middle;
        dominant-baseline: middle;
    }
    .junction-point {
        stroke-width: 2;
        fill-opacity: 0.7;
    }
    """
    
    # Создаем группу для junctions
    junctions_group = ET.SubElement(root, 'g')
    junctions_group.set('id', 'junctions')
    
    # Добавляем каждый junction
    for idx, analyzed in enumerate(analyzed_junctions):
        junction = analyzed['junction']
        jx, jy = junction['x'], junction['y']
        jtype = analyzed['type']
        
        # Определяем цвет
        color = type_colors.get(jtype, '#888888')
        
        # Добавляем точку
        circle = ET.SubElement(junctions_group, 'circle')
        circle.set('cx', str(jx))
        circle.set('cy', str(jy))
        circle.set('r', '5')
        circle.set('fill', color)
        circle.set('class', 'junction-point')
        circle.set('stroke', '#000000')
        
        # Добавляем метку с типом
        text = ET.SubElement(junctions_group, 'text')
        text.set('x', str(jx))
        text.set('y', str(jy - 10))
        text.set('class', 'junction-label')
        text.set('fill', color)
        text.text = jtype
        
        # Добавляем метку с ID
        text_id = ET.SubElement(junctions_group, 'text')
        text_id.set('x', str(jx))
        text_id.set('y', str(jy + 15))
        text_id.set('class', 'junction-label')
        text_id.set('fill', '#000000')
        text_id.set('font-size', '8px')
        text_id.text = f"J{idx+1}"
    
    # Сохраняем результат
    tree.write(output_svg, encoding='utf-8', xml_declaration=True)
```

### Шаг 3: Основная функция

```python
def main():
    # Загружаем данные
    with open('plan_floor1_objects.json', 'r') as f:
        data = json.load(f)
    
    # Анализируем junctions
    analyzed_junctions = analyze_all_junctions_improved(data, wall_thickness=20.0)
    
    # Создаем визуализацию
    create_junctions_svg(
        analyzed_junctions,
        'wall_polygons_opening_based.svg',
        'wall_polygons_with_improved_junction_types.svg'
    )
    
    # Выводим статистику
    type_counts = {}
    for analyzed in analyzed_junctions:
        jtype = analyzed['type']
        type_counts[jtype] = type_counts.get(jtype, 0) + 1
    
    print("Статистика по типам junctions:")
    for jtype, count in type_counts.items():
        print(f"  {jtype}: {count}")
    
    print(f"\nВсего junctions: {len(analyzed_junctions)}")
    print("Результат сохранен в: wall_polygons_with_improved_junction_types.svg")
```

## Ожидаемый результат

После выполнения скрипта мы получим SVG файл `wall_polygons_with_improved_junction_types.svg`, в котором:
- J15 будет правильно определен как T-junction (зеленый цвет)
- Все остальные junctions также будут правильно классифицированы
- Будет присутствовать легенда с объяснением цветовой схемы

## Проверка результата

Для проверки можно запустить скрипт `view_svg.py` для открытия файла в браузере:

```bash
cd floor && python view_svg.py
```

Или открыть файл `wall_polygons_with_improved_junction_types.svg` вручную в браузере.