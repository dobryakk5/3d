# Implementation Guide for L-Junction Extensions

## Overview
This guide provides the exact code changes needed to implement the L-junction extension functionality in `visualize_polygons.py`.

## Step-by-Step Implementation

### 1. Add New Functions (after line 812)

Add these functions after the `analyze_junction_types_with_thickness` function:

```python
def find_l_junctions(junctions: List[JunctionPoint]) -> List[JunctionPoint]:
    """
    Extracts all L-junctions from the list of processed junctions
    
    Args:
        junctions: List of all junction points with detected types
        
    Returns:
        List of L-junction points
    """
    l_junctions = [j for j in junctions if j.detected_type == 'L']
    print(f"  ✓ Найдено {len(l_junctions)} L-junctions")
    return l_junctions

def find_wall_segments_at_l_junction(l_junction: JunctionPoint, 
                                   wall_segments: List[WallSegmentFromOpening],
                                   junction_wall_segments: List[WallSegmentFromJunction],
                                   tolerance: float) -> Tuple[List, List]:
    """
    Finds wall segments connected to a specific L-junction
    
    Args:
        l_junction: The L-junction point
        wall_segments: List of wall segments from openings
        junction_wall_segments: List of wall segments from junctions
        tolerance: Distance tolerance for connection checking
        
    Returns:
        Tuple of (opening_segments, junction_segments) connected to the L-junction
    """
    opening_segments = []
    junction_segments = []
    jx, jy = l_junction.x, l_junction.y
    
    # Check opening-based wall segments
    for segment in wall_segments:
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        if start_dist <= tolerance or end_dist <= tolerance:
            opening_segments.append(segment)
    
    # Check junction-based wall segments
    for segment in junction_wall_segments:
        start_dist = math.sqrt((segment.start_junction.x - jx)**2 + (segment.start_junction.y - jy)**2)
        end_dist = math.sqrt((segment.end_junction.x - jx)**2 + (segment.end_junction.y - jy)**2)
        
        if start_dist <= tolerance or end_dist <= tolerance:
            junction_segments.append(segment)
    
    return opening_segments, junction_segments

def extend_segment_to_polygon_edge(segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                 l_junction: JunctionPoint,
                                 wall_polygons: List[Dict],
                                 wall_thickness: float) -> Dict[str, float]:
    """
    Extends a wall segment from an L-junction to the edge of the containing wall polygon
    
    Args:
        segment: The wall segment to extend
        l_junction: The L-junction point
        wall_polygons: List of wall polygons
        wall_thickness: Wall thickness for calculations
        
    Returns:
        Extended bbox dictionary
    """
    # Find the polygon containing the L-junction
    containing_polygon = None
    for wall in wall_polygons:
        vertices = wall.get('vertices', [])
        if vertices and is_point_in_polygon(l_junction.x, l_junction.y, vertices):
            containing_polygon = wall
            break
    
    if not containing_polygon:
        return segment.bbox  # Return original if no polygon found
    
    # Get the direction of extension based on junction directions
    directions = l_junction.directions
    if not directions:
        return segment.bbox
    
    # Determine which direction to extend (choose the first available)
    extend_direction = directions[0]
    
    # Get the polygon edge intersection in that direction
    analysis = analyze_polygon_extensions_with_thickness(
        {'x': l_junction.x, 'y': l_junction.y},
        containing_polygon['vertices'],
        wall_thickness
    )
    
    intersections = analysis['intersections']
    
    # Create extended bbox based on direction
    if extend_direction in intersections and intersections[extend_direction] is not None:
        if segment.orientation == 'horizontal':
            if extend_direction == 'left':
                # Extend to the left
                new_x = intersections[extend_direction]
                new_width = segment.bbox['x'] + segment.bbox['width'] - new_x
                return {
                    'x': new_x,
                    'y': segment.bbox['y'],
                    'width': new_width,
                    'height': segment.bbox['height'],
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
            elif extend_direction == 'right':
                # Extend to the right
                new_width = intersections[extend_direction] - segment.bbox['x']
                return {
                    'x': segment.bbox['x'],
                    'y': segment.bbox['y'],
                    'width': new_width,
                    'height': segment.bbox['height'],
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
        else:  # vertical
            if extend_direction == 'up':
                # Extend upward
                new_y = intersections[extend_direction]
                new_height = segment.bbox['y'] + segment.bbox['height'] - new_y
                return {
                    'x': segment.bbox['x'],
                    'y': new_y,
                    'width': segment.bbox['width'],
                    'height': new_height,
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
            elif extend_direction == 'down':
                # Extend downward
                new_height = intersections[extend_direction] - segment.bbox['y']
                return {
                    'x': segment.bbox['x'],
                    'y': segment.bbox['y'],
                    'width': segment.bbox['width'],
                    'height': new_height,
                    'orientation': segment.orientation,
                    'extended': True,
                    'extension_direction': extend_direction
                }
    
    return segment.bbox  # Return original if extension not possible

def extend_segment_to_perpendicular_x(segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                    perpendicular_segment: Union[WallSegmentFromOpening, WallSegmentFromJunction],
                                    l_junction: JunctionPoint) -> Dict[str, float]:
    """
    Extends a wall segment to reach the X coordinate of a perpendicular segment
    
    Args:
        segment: The wall segment to extend
        perpendicular_segment: The perpendicular wall segment
        l_junction: The L-junction point
        
    Returns:
        Extended bbox dictionary
    """
    # Determine which segment is horizontal and which is vertical
    if segment.orientation == 'horizontal' and perpendicular_segment.orientation == 'vertical':
        # Extend horizontal segment to reach vertical segment's X
        target_x = perpendicular_segment.bbox['x']
        
        # Determine if we need to extend left or right
        if target_x < segment.bbox['x']:
            # Extend to the left
            new_x = target_x
            new_width = segment.bbox['x'] + segment.bbox['width'] - target_x
            extension_direction = 'left'
        else:
            # Extend to the right
            new_width = target_x + perpendicular_segment.bbox['width'] - segment.bbox['x']
            new_x = segment.bbox['x']
            extension_direction = 'right'
        
        return {
            'x': new_x,
            'y': segment.bbox['y'],
            'width': new_width,
            'height': segment.bbox['height'],
            'orientation': segment.orientation,
            'extended': True,
            'extension_direction': extension_direction,
            'extension_type': 'perpendicular_alignment'
        }
    
    elif segment.orientation == 'vertical' and perpendicular_segment.orientation == 'horizontal':
        # Extend vertical segment to reach horizontal segment's Y
        target_y = perpendicular_segment.bbox['y']
        
        # Determine if we need to extend up or down
        if target_y < segment.bbox['y']:
            # Extend upward
            new_y = target_y
            new_height = segment.bbox['y'] + segment.bbox['height'] - target_y
            extension_direction = 'up'
        else:
            # Extend downward
            new_height = target_y + perpendicular_segment.bbox['height'] - segment.bbox['y']
            new_y = segment.bbox['y']
            extension_direction = 'down'
        
        return {
            'x': segment.bbox['x'],
            'y': new_y,
            'width': segment.bbox['width'],
            'height': new_height,
            'orientation': segment.orientation,
            'extended': True,
            'extension_direction': extension_direction,
            'extension_type': 'perpendicular_alignment'
        }
    
    return segment.bbox  # Return original if segments are not perpendicular

def process_l_junction_extensions(junctions: List[JunctionPoint],
                                wall_segments: List[WallSegmentFromOpening],
                                junction_wall_segments: List[WallSegmentFromJunction],
                                wall_polygons: List[Dict],
                                wall_thickness: float) -> List[Dict[str, float]]:
    """
    Processes all L-junctions and creates extended wall segments
    
    Args:
        junctions: List of all junction points
        wall_segments: List of wall segments from openings
        junction_wall_segments: List of wall segments from junctions
        wall_polygons: List of wall polygons
        wall_thickness: Wall thickness for calculations
        
    Returns:
        List of extended bbox dictionaries for visualization
    """
    extended_segments = []
    
    # Find all L-junctions
    l_junctions = find_l_junctions(junctions)
    
    if not l_junctions:
        print("  ✓ L-junctions не найдены")
        return extended_segments
    
    # Process each L-junction
    for l_junction in l_junctions:
        print(f"  Обработка L-junction {l_junction.id} ({l_junction.x}, {l_junction.y})")
        
        # Find connected wall segments
        opening_segments, junction_segments = find_wall_segments_at_l_junction(
            l_junction, wall_segments, junction_wall_segments, wall_thickness / 2.0
        )
        
        all_connected_segments = opening_segments + junction_segments
        
        if len(all_connected_segments) < 2:
            print(f"    ✗ Недостаточно сегментов для L-junction {l_junction.id}")
            continue
        
        # Select the first segment for extension to polygon edge
        selected_segment = all_connected_segments[0]
        print(f"    ✓ Выбран сегмент для расширения до края полигона: {selected_segment.segment_id}")
        
        # Extend to polygon edge
        extended_to_edge = extend_segment_to_polygon_edge(
            selected_segment, l_junction, wall_polygons, wall_thickness
        )
        
        if extended_to_edge.get('extended', False):
            extended_segments.append(extended_to_edge)
            print(f"    ✓ Сегмент расширен до края полигона в направлении {extended_to_edge.get('extension_direction')}")
        
        # If we have at least 2 segments, extend one to align with the perpendicular
        if len(all_connected_segments) >= 2:
            perpendicular_segment = all_connected_segments[1]
            print(f"    ✓ Выбран перпендикулярный сегмент: {perpendicular_segment.segment_id}")
            
            # Extend to align with perpendicular segment
            extended_to_perpendicular = extend_segment_to_perpendicular_x(
                selected_segment, perpendicular_segment, l_junction
            )
            
            if extended_to_perpendicular.get('extended', False):
                extended_segments.append(extended_to_perpendicular)
                print(f"    ✓ Сегмент расширен для выравнивания с перпендикулярным сегментом")
    
    print(f"  ✓ Создано {len(extended_segments)} расширенных сегментов для L-junctions")
    return extended_segments
```

### 2. Add Visualization Function (after line 1223)

Add this function after the `draw_junction_based_wall_bboxes` function:

```python
def draw_extended_segments(dwg: svgwrite.Drawing,
                          extended_segments: List[Dict[str, float]],
                          inverse_scale: float,
                          padding: float) -> None:
    """
    Draws extended wall segments with special styling
    
    Args:
        dwg: SVG drawing object
        extended_segments: List of extended bbox dictionaries
        inverse_scale: Scale factor for coordinate transformation
        padding: Padding for SVG coordinates
    """
    extended_group = dwg.add(dwg.g(id='extended_segments'))
    
    # Style for extended segments
    extended_style = {
        'stroke': '#FF69B4',  # Hot pink
        'stroke_width': 3,
        'fill': 'none',
        'stroke_dasharray': '10,5',  # Dashed line
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round',
        'opacity': 0.8
    }
    
    print(f"  ✓ Отрисовка {len(extended_segments)} расширенных сегментов")
    
    for idx, segment in enumerate(extended_segments):
        # Transform coordinates
        x, y = transform_coordinates(segment['x'], segment['y'], inverse_scale, padding)
        width = segment['width'] * inverse_scale
        height = segment['height'] * inverse_scale
        
        # Create rectangle
        rect = dwg.rect(insert=(x, y), size=(width, height), **extended_style)
        extended_group.add(rect)
        
        # Add label with extension info
        extension_type = segment.get('extension_type', 'polygon_edge')
        direction = segment.get('extension_direction', 'unknown')
        orientation_label = 'h' if segment['orientation'] == 'horizontal' else 'v'
        
        text = dwg.text(
            f"E{idx+1}_{direction[0].upper()}{orientation_label}_{extension_type}",
            insert=(x + width/2, y + height/2),
            text_anchor='middle',
            fill='#FF69B4',
            font_size='8px',
            font_weight='bold'
        )
        extended_group.add(text)
```

### 3. Update Legend Function (modify line 1379)

Update the `add_legend` function to include extended segments:

```python
def add_legend(dwg: svgwrite.Drawing, width: int, height: int, styles: Dict) -> None:
    """Добавляет легенду с описанием цветов и типов junctions"""
    legend_group = dwg.add(dwg.g(id='legend'))
    
    # Позиция легенды
    legend_x = 20
    legend_y = height - 345  # Increased for extended segments
    item_height = 25
    
    # Заголовок легенды
    title = dwg.text(
        "Легенда:",
        insert=(legend_x, legend_y),
        fill='black',
        font_size='16px',
        font_weight='bold'
    )
    legend_group.add(title)
    
    # Элементы легенды
    legend_items = [
        ("Стены (opening-based)", styles['wall']),
        ("Стены (junction-based)", {'stroke': '#FF6347', 'fill': 'none'}),
        ("Стеновые полигоны (JSON)", {'stroke': '#808080', 'fill': 'none', 'stroke_dasharray': '5,5'}),
        ("Junction L-типа", get_junction_style('L')),
        ("Junction T-типа", get_junction_style('T')),
        ("Junction X-типа", get_junction_style('X')),
        ("Прямое соединение", get_junction_style('straight')),
        ("Неизвестный тип", get_junction_style('unknown')),
        ("Связи с junctions", {'stroke': '#FF00FF', 'fill': 'none', 'stroke_dasharray': '2,2'}),
        ("Расширенные сегменты", {'stroke': '#FF69B4', 'fill': 'none', 'stroke_dasharray': '10,5'}),
        ("Колонны", styles['pillar']),
        ("Окна", styles['window']),
        ("Двери", styles['door'])
    ]
    
    for i, (label, style) in enumerate(legend_items):
        y_pos = legend_y + (i + 1) * item_height
        
        # Прямоугольник с цветом
        rect = dwg.rect(
            insert=(legend_x, y_pos - 10),
            size=(20, 15),
            **{k: v for k, v in style.items() if k in ['fill', 'stroke', 'stroke_dasharray']}
        )
        legend_group.add(rect)
        
        # Текст
        text = dwg.text(
            label,
            insert=(legend_x + 30, y_pos + 3),
            fill='black',
            font_size='14px'
        )
        legend_group.add(text)
    
    print("  ✓ Легенда добавлена")
```

### 4. Update Main Function (modify after line 1523)

Update the `visualize_polygons_opening_based_with_junction_types` function to process L-junctions:

```python
    # Process L-junctions and create extensions
    print(f"\n{'='*60}")
    print("ОБРАБОТКА L-JUNCTIONS И СОЗДАНИЕ РАСШИРЕНИЙ")
    print(f"{'='*60}")
    
    extended_segments = process_l_junction_extensions(
        junctions_with_types, wall_segments, junction_wall_segments, 
        data.get('wall_polygons', []), wall_thickness
    )
    
    print(f"\n{'='*60}")
    print(f"ИТОГО: {len(wall_segments)} сегментов стен из проемов + {len(junction_wall_segments)} сегментов стен из junctions + {len(extended_segments)} расширенных сегментов")
    print(f"{'='*60}")
    
    # Convert WallSegmentFromOpening to bbox format for visualization
    processed_segments = [segment.bbox for segment in wall_segments]
    # Add junction-based segments
    processed_segments.extend([segment.bbox for segment in junction_wall_segments])
    # Add extended segments
    processed_segments.extend(extended_segments)
```

### 5. Update Visualization Order (modify after line 1564)

Update the visualization order to include extended segments:

```python
    # 6. Отрисовываем расширенные сегменты
    draw_extended_segments(dwg, extended_segments, inverse_scale, padding)
    
    # 7. Наконец отрисовываем сегменты стен из проемов (верхний слой)
    draw_opening_based_wall_bboxes(dwg, wall_segments, inverse_scale, padding, styles)
```

### 6. Update Statistics (modify after line 1591)

Update the statistics output to include extended segments:

```python
    print(f"  Сегменты стен из junctions: {len(junction_wall_segments)}")
    print(f"  Расширенные сегменты: {len(extended_segments)}")
    print(f"  Всего сегментов стен: {len(wall_segments) + len(junction_wall_segments) + len(extended_segments)}")
```

## Required Import

Add this import at the top of the file (after line 17):

```python
from typing import Union
```

## Testing

After implementing these changes:

1. Run the script: `python visualize_polygons.py`
2. Check the output for L-junction processing messages
3. Open the generated SVG file
4. Look for hot pink dashed lines indicating extended segments
5. Verify that the legend includes "Расширенные сегменты"

## Expected Results

The script should:
1. Identify all L-junctions in the floor plan
2. For each L-junction, select one connected wall segment
3. Extend the selected segment to the edge of the containing wall polygon
4. If there's a perpendicular segment, extend the selected segment to align with it
5. Visualize all extensions with hot pink dashed lines and labels
6. Update the legend to include the extended segments
7. Update the statistics to show the count of extended segments