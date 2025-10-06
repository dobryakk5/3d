#!/usr/bin/env python3
"""
Visualize floor plan detection results in stages:
1. Original image
2. Junctions (45 points)
3. Windows (7)
4. Doors (7)
5. Walls (87 segments)
6. Combined result
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def draw_junctions(img, junctions_dict):
    """Draw junction points with color coding by type"""
    img_vis = img.copy()

    # Color mapping for different junction types
    type_colors = {
        'wall_junction_3way': (0, 0, 255),      # Red
        'door_left_corner': (0, 255, 0),        # Green
        'door_right_corner': (0, 200, 100),     # Light green
        'window_left_corner': (255, 100, 0),    # Blue
        'window_right_corner': (255, 200, 0),   # Light blue
    }

    # Default color for unknown types
    default_color = (128, 128, 128)  # Gray

    for jtype, junctions_list in junctions_dict.items():
        color = type_colors.get(jtype, default_color)

        for j in junctions_list:
            x, y = int(j['x']), int(j['y'])
            cv2.circle(img_vis, (x, y), 8, color, -1)
            cv2.circle(img_vis, (x, y), 8, (0, 0, 0), 2)

    return img_vis

def draw_windows(img, windows):
    """Draw window bounding boxes"""
    img_vis = img.copy()

    for w in windows:
        x, y, width, height = w['bbox']
        x, y, width, height = int(x), int(y), int(width), int(height)

        # Draw filled semi-transparent blue rectangle
        overlay = img_vis.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 200, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0, img_vis)

        # Draw border
        cv2.rectangle(img_vis, (x, y), (x + width, y + height), (255, 150, 0), 3)

    return img_vis

def draw_doors(img, doors):
    """Draw door bounding boxes"""
    img_vis = img.copy()

    for d in doors:
        x, y, width, height = d['bbox']
        x, y, width, height = int(x), int(y), int(width), int(height)

        # Draw filled semi-transparent green rectangle
        overlay = img_vis.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 100), -1)
        cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0, img_vis)

        # Draw border
        cv2.rectangle(img_vis, (x, y), (x + width, y + height), (0, 200, 0), 3)

    return img_vis

def draw_walls(img, wall_segments):
    """Draw wall segments"""
    img_vis = img.copy()

    for seg in wall_segments:
        x1, y1 = int(seg['start'][0]), int(seg['start'][1])
        x2, y2 = int(seg['end'][0]), int(seg['end'][1])

        cv2.line(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 4)

    return img_vis

def draw_combined(img, junctions_dict, windows, doors, wall_segments):
    """Draw all elements together"""
    img_vis = img.copy()

    # 1. Walls (red lines)
    for seg in wall_segments:
        x1, y1 = int(seg['start'][0]), int(seg['start'][1])
        x2, y2 = int(seg['end'][0]), int(seg['end'][1])
        cv2.line(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 2. Windows (blue rectangles)
    for w in windows:
        x, y, width, height = w['bbox']
        x, y, width, height = int(x), int(y), int(width), int(height)
        overlay = img_vis.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 200, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0, img_vis)
        cv2.rectangle(img_vis, (x, y), (x + width, y + height), (255, 150, 0), 2)

    # 3. Doors (green rectangles)
    for d in doors:
        x, y, width, height = d['bbox']
        x, y, width, height = int(x), int(y), int(width), int(height)
        overlay = img_vis.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 100), -1)
        cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0, img_vis)
        cv2.rectangle(img_vis, (x, y), (x + width, y + height), (0, 200, 0), 2)

    # 4. Junctions (colored circles by type)
    type_colors = {
        'wall_junction_3way': (0, 0, 255),      # Red
        'door_left_corner': (0, 255, 0),        # Green
        'door_right_corner': (0, 200, 100),     # Light green
        'window_left_corner': (255, 100, 0),    # Blue
        'window_right_corner': (255, 200, 0),   # Light blue
    }
    default_color = (128, 128, 128)

    for jtype, junctions_list in junctions_dict.items():
        color = type_colors.get(jtype, default_color)
        for j in junctions_list:
            x, y = int(j['x']), int(j['y'])
            cv2.circle(img_vis, (x, y), 6, color, -1)
            cv2.circle(img_vis, (x, y), 6, (0, 0, 0), 1)

    return img_vis

def main():
    # Load data
    print("Loading data...")
    img = cv2.imread('floor_plan.jpg')

    with open('floor_plan_detections.json', 'r') as f:
        data = json.load(f)

    junctions = data['junctions']
    windows = data['windows']
    doors = data['doors']
    wall_segments = data['walls']

    # Count junctions
    total_junctions = sum(len(v) for v in junctions.values())
    print(f"  {total_junctions} junctions")
    print(f"  {len(windows)} windows")
    print(f"  {len(doors)} doors")
    print(f"  {len(wall_segments)} wall segments")

    # Create visualizations
    print("\nCreating stage visualizations...")

    img_original = img.copy()
    img_junctions = draw_junctions(img, junctions)
    img_windows = draw_windows(img, windows)
    img_doors = draw_doors(img, doors)
    img_walls = draw_walls(img, wall_segments)
    img_combined = draw_combined(img, junctions, windows, doors, wall_segments)

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Floor Plan Detection Pipeline - Stage by Stage', fontsize=20, fontweight='bold')

    # Convert BGR to RGB for matplotlib
    axes[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original Image', fontsize=16)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(img_junctions, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'2. Junctions ({total_junctions} points)', fontsize=16)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(img_windows, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'3. Windows ({len(windows)} detected)', fontsize=16)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(img_doors, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'4. Doors ({len(doors)} detected)', fontsize=16)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(img_walls, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'5. Walls ({len(wall_segments)} segments)', fontsize=16)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('6. Combined Result', fontsize=16)
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save
    output_path = 'detection_stages.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.close()

    # Also save individual images
    cv2.imwrite('stage_1_original.png', img_original)
    cv2.imwrite('stage_2_junctions.png', img_junctions)
    cv2.imwrite('stage_3_windows.png', img_windows)
    cv2.imwrite('stage_4_doors.png', img_doors)
    cv2.imwrite('stage_5_walls.png', img_walls)
    cv2.imwrite('stage_6_combined.png', img_combined)

    print("\n✓ Individual stage images saved:")
    print("  stage_1_original.png")
    print("  stage_2_junctions.png")
    print("  stage_3_windows.png")
    print("  stage_4_doors.png")
    print("  stage_5_walls.png")
    print("  stage_6_combined.png")

if __name__ == '__main__':
    main()
