#!/usr/bin/env python3
"""
Create correct vector overlay on raster image.
- No legend
- Correct scale (coordinates in JSON are already in downscaled space)
- Vector objects overlay on full-resolution raster
"""

import json
import svgwrite
from PIL import Image
import sys

def load_json(path):
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def create_overlay_svg(data, raster_path, output_path):
    """Create SVG overlay with raster image as background"""

    # Load raster image to get dimensions
    try:
        img = Image.open(raster_path)
        img_width, img_height = img.size
        print(f"  Raster dimensions: {img_width}x{img_height}")
    except Exception as e:
        print(f"✗ Error loading raster image: {e}")
        return None

    # Get scale factor from JSON
    scale = data['metadata']['scale_factor']
    print(f"  Scale factor: {scale:.4f}")
    print(f"  JSON coordinates are in downscaled space")

    # Calculate inverse scale to map JSON coords to raster
    inverse_scale = 1.0 / scale
    print(f"  Inverse scale: {inverse_scale:.4f}")

    # Create SVG with raster dimensions
    dwg = svgwrite.Drawing(output_path, size=(img_width, img_height), profile='tiny')

    # Add raster as background (using relative path)
    dwg.add(dwg.image(
        href=raster_path,
        insert=(0, 0),
        size=(img_width, img_height)
    ))

    # Define colors (bright for visibility on raster)
    colors = {
        'wall_hatching': 'red',      # Red for hatching walls
        'wall_dl': 'lime',            # Bright green for DL walls
        'window': 'cyan',             # Cyan for windows
        'door': 'yellow',             # Yellow for doors
        'pillar': 'magenta',          # Magenta for pillars
    }

    # Draw walls
    wall_group = dwg.add(dwg.g(id='walls', opacity=0.8))
    for wall in data['walls']:
        # JSON coords are in downscaled space, multiply by inverse_scale
        x1 = wall['start']['x'] * inverse_scale
        y1 = wall['start']['y'] * inverse_scale
        x2 = wall['end']['x'] * inverse_scale
        y2 = wall['end']['y'] * inverse_scale

        color = colors['wall_hatching'] if wall['source'] == 'hatching' else colors['wall_dl']
        wall_group.add(dwg.line(
            start=(x1, y1),
            end=(x2, y2),
            stroke=color,
            stroke_width=6,
            stroke_linecap='round'
        ))

    # Draw pillars
    pillar_group = dwg.add(dwg.g(id='pillars', opacity=0.6))
    for pillar in data['pillars']:
        bbox = pillar['bbox']
        x = bbox['x'] * inverse_scale
        y = bbox['y'] * inverse_scale
        w = bbox['width'] * inverse_scale
        h = bbox['height'] * inverse_scale

        pillar_group.add(dwg.rect(
            insert=(x, y),
            size=(w, h),
            fill=colors['pillar'],
            stroke=colors['pillar'],
            stroke_width=3
        ))

    # Draw windows
    window_group = dwg.add(dwg.g(id='windows', opacity=0.8))
    for opening in data['openings']:
        if opening['type'] == 'window':
            bbox = opening['bbox']
            x = bbox['x'] * inverse_scale
            y = bbox['y'] * inverse_scale
            w = bbox['width'] * inverse_scale
            h = bbox['height'] * inverse_scale

            window_group.add(dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill='none',
                stroke=colors['window'],
                stroke_width=4
            ))

    # Draw doors
    door_group = dwg.add(dwg.g(id='doors', opacity=0.8))
    for opening in data['openings']:
        if opening['type'] == 'door':
            bbox = opening['bbox']
            x = bbox['x'] * inverse_scale
            y = bbox['y'] * inverse_scale
            w = bbox['width'] * inverse_scale
            h = bbox['height'] * inverse_scale

            door_group.add(dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill='none',
                stroke=colors['door'],
                stroke_width=4
            ))

    dwg.save()
    print(f"✓ Overlay SVG saved: {output_path}")
    print(f"  SVG dimensions: {img_width}x{img_height}")
    return img_width, img_height

def create_pure_vector_svg(data, output_path):
    """Create pure vector SVG (no raster background)"""

    scale = data['metadata']['scale_factor']
    inverse_scale = 1.0 / scale

    # Calculate bounds from all objects
    all_coords = []

    for wall in data['walls']:
        all_coords.extend([
            wall['start']['x'], wall['start']['y'],
            wall['end']['x'], wall['end']['y']
        ])

    for opening in data['openings']:
        bbox = opening['bbox']
        all_coords.extend([
            bbox['x'], bbox['y'],
            bbox['x'] + bbox['width'], bbox['y'] + bbox['height']
        ])

    for pillar in data['pillars']:
        bbox = pillar['bbox']
        all_coords.extend([
            bbox['x'], bbox['y'],
            bbox['x'] + bbox['width'], bbox['y'] + bbox['height']
        ])

    if not all_coords:
        print("✗ No objects to draw")
        return None

    # Get bounds in downscaled space
    min_x = min([all_coords[i] for i in range(0, len(all_coords), 2)])
    max_x = max([all_coords[i] for i in range(0, len(all_coords), 2)])
    min_y = min([all_coords[i] for i in range(1, len(all_coords), 2)])
    max_y = max([all_coords[i] for i in range(1, len(all_coords), 2)])

    # Scale to raster space
    min_x *= inverse_scale
    max_x *= inverse_scale
    min_y *= inverse_scale
    max_y *= inverse_scale

    # Add margin
    margin = 50
    width = int(max_x - min_x + 2 * margin)
    height = int(max_y - min_y + 2 * margin)
    offset_x = -min_x + margin
    offset_y = -min_y + margin

    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')

    # White background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

    # Define colors
    colors = {
        'wall_hatching': '#2C3E50',
        'wall_dl': '#34495E',
        'window': '#3498DB',
        'door': '#E67E22',
        'pillar': '#E74C3C',
    }

    # Draw walls
    wall_group = dwg.add(dwg.g(id='walls'))
    for wall in data['walls']:
        x1 = (wall['start']['x'] * inverse_scale + offset_x)
        y1 = (wall['start']['y'] * inverse_scale + offset_y)
        x2 = (wall['end']['x'] * inverse_scale + offset_x)
        y2 = (wall['end']['y'] * inverse_scale + offset_y)

        color = colors['wall_hatching'] if wall['source'] == 'hatching' else colors['wall_dl']
        wall_group.add(dwg.line(
            start=(x1, y1),
            end=(x2, y2),
            stroke=color,
            stroke_width=8,
            stroke_linecap='round'
        ))

    # Draw pillars
    pillar_group = dwg.add(dwg.g(id='pillars'))
    for pillar in data['pillars']:
        bbox = pillar['bbox']
        x = bbox['x'] * inverse_scale + offset_x
        y = bbox['y'] * inverse_scale + offset_y
        w = bbox['width'] * inverse_scale
        h = bbox['height'] * inverse_scale

        pillar_group.add(dwg.rect(
            insert=(x, y),
            size=(w, h),
            fill=colors['pillar'],
            fill_opacity=0.6,
            stroke='darkred',
            stroke_width=2
        ))

    # Draw windows
    window_group = dwg.add(dwg.g(id='windows'))
    for opening in data['openings']:
        if opening['type'] == 'window':
            bbox = opening['bbox']
            x = bbox['x'] * inverse_scale + offset_x
            y = bbox['y'] * inverse_scale + offset_y
            w = bbox['width'] * inverse_scale
            h = bbox['height'] * inverse_scale

            window_group.add(dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill=colors['window'],
                fill_opacity=0.7,
                stroke='darkblue',
                stroke_width=3
            ))

    # Draw doors
    door_group = dwg.add(dwg.g(id='doors'))
    for opening in data['openings']:
        if opening['type'] == 'door':
            bbox = opening['bbox']
            x = bbox['x'] * inverse_scale + offset_x
            y = bbox['y'] * inverse_scale + offset_y
            w = bbox['width'] * inverse_scale
            h = bbox['height'] * inverse_scale

            door_group.add(dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill=colors['door'],
                fill_opacity=0.7,
                stroke='darkorange',
                stroke_width=3
            ))

    dwg.save()
    print(f"✓ Pure vector SVG saved: {output_path}")
    print(f"  Dimensions: {width}x{height}")
    return width, height

def main():
    json_path = 'plan_floor1_objects_fixed.json'
    raster_path = 'plan_floor1.jpg'
    overlay_output = 'plan_floor1_overlay_correct.svg'
    vector_output = 'plan_floor1_vector_correct.svg'

    print("="*70)
    print("CREATING CORRECT VECTOR OVERLAY")
    print("="*70)

    # Load JSON
    print(f"\n[1/3] Loading fixed JSON: {json_path}")
    try:
        data = load_json(json_path)
        stats = data['statistics']
        print(f"  ✓ Walls: {stats['walls']}, Windows: {stats['windows']}, Doors: {stats['doors']}, Pillars: {stats['pillars']}")
    except Exception as e:
        print(f"✗ Error loading JSON: {e}")
        return 1

    # Create overlay
    print(f"\n[2/3] Creating overlay SVG...")
    try:
        create_overlay_svg(data, raster_path, overlay_output)
    except Exception as e:
        print(f"✗ Error creating overlay: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create pure vector
    print(f"\n[3/3] Creating pure vector SVG...")
    try:
        create_pure_vector_svg(data, vector_output)
    except Exception as e:
        print(f"✗ Error creating vector: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  1. Overlay: {overlay_output}")
    print(f"  2. Vector:  {vector_output}")
    print(f"\nOpen {overlay_output} to check alignment!")

    return 0

if __name__ == '__main__':
    sys.exit(main())
