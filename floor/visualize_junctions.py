#!/usr/bin/env python3
"""
Visualize junctions analysis similar to plan_floor1_JUNCTIONS.png
Shows: Original, Heatmap Sum, Junction Points by Type, All Junctions,
       Junctions on Walls, Wall Graph from Junctions, Wall Graph Clean
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import sys
sys.path.append('floortrans')
from floortrans.models import get_model

# Junction type names (0-20)
JUNCTION_TYPES = [
    'wall_junction_3way',      # 0
    'door_left_corner',         # 1
    'door_right_corner',        # 2
    'window_left_corner',       # 3
    'window_right_corner',      # 4
    'unknown_5',                # 5
    'unknown_6',                # 6
    'unknown_7',                # 7
    'unknown_8',                # 8
    'unknown_9',                # 9
    'unknown_10',               # 10
    'unknown_11',               # 11
    'unknown_12',               # 12
    'unknown_13',               # 13
    'unknown_14',               # 14
    'unknown_15',               # 15
    'unknown_16',               # 16
    'unknown_17',               # 17
    'unknown_18',               # 18
    'unknown_19',               # 19
    'unknown_20',               # 20
]

def multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1]):
    """Run inference at multiple scales"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)

    all_preds = []
    for scale in scales:
        if scale != 1.0:
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            scaled = img_tensor

        with torch.no_grad():
            pred = model(scaled)

        if scale != 1.0:
            pred = F.interpolate(pred, size=(img_tensor.shape[2], img_tensor.shape[3]),
                               mode='bilinear', align_corners=False)
        all_preds.append(pred)

    return torch.mean(torch.stack(all_preds), dim=0)

def extract_junctions(heatmaps, threshold=0.3):
    """Extract junction points from heatmaps"""
    junctions_by_type = {}

    for channel in range(21):
        heatmap = heatmaps[channel]
        peaks = (heatmap > threshold).astype(np.uint8)

        labeled, num = ndimage.label(peaks)
        points = []

        for i in range(1, num + 1):
            region = (labeled == i)
            y_coords, x_coords = np.where(region)

            if len(x_coords) == 0:
                continue

            # Use center of mass
            y = int(np.mean(y_coords))
            x = int(np.mean(x_coords))
            confidence = float(heatmap[y, x])

            points.append({'x': x, 'y': y, 'confidence': confidence})

        if points:
            junctions_by_type[JUNCTION_TYPES[channel]] = points

    return junctions_by_type

def build_wall_graph(junctions_dict):
    """Build wall connections from junction points"""
    # Collect wall-related junctions
    wall_points = []

    # Wall junctions are typically unknown types 13-18
    wall_types = ['unknown_13', 'unknown_14', 'unknown_15', 'unknown_16', 'unknown_18']

    for jtype in wall_types:
        if jtype in junctions_dict:
            for j in junctions_dict[jtype]:
                wall_points.append((j['x'], j['y']))

    # Add explicit wall junctions
    if 'wall_junction_3way' in junctions_dict:
        for j in junctions_dict['wall_junction_3way']:
            wall_points.append((j['x'], j['y']))

    # Build connections (simple: connect nearby points)
    connections = []
    max_distance = 150  # Max distance to connect

    for i, p1 in enumerate(wall_points):
        for j, p2 in enumerate(wall_points):
            if i >= j:
                continue

            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

            if dist < max_distance:
                # Check if roughly horizontal or vertical
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])

                # Allow if mostly horizontal or mostly vertical
                if dx > 3 * dy or dy > 3 * dx:
                    connections.append((p1, p2))

    return wall_points, connections

def main():
    print("="*80)
    print("JUNCTION ANALYSIS VISUALIZATION")
    print("="*80)

    image_path = 'plan_floor1.jpg'

    # Load model
    print("\n[1/4] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load image
    print("\n[2/4] Preprocessing image...")
    img_orig = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img_orig.size

    max_size = 2048
    w, h = orig_width, orig_height
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        img = img_orig

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    img_np = np.array(img)
    img_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Normalize
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    print("\n[3/4] Running DL inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Extract heatmaps (channels 0-20)
    heatmaps = prediction[0, :21].cpu().data.numpy()

    # Extract junctions
    print("\n[4/4] Extracting junctions...")
    junctions_dict = extract_junctions(heatmaps, threshold=0.3)

    total_junctions = sum(len(v) for v in junctions_dict.values())
    print(f"   Found {total_junctions} junction points")
    print(f"   Junction types: {len(junctions_dict)}")

    # Build wall graph
    wall_points, wall_connections = build_wall_graph(junctions_dict)
    print(f"   Wall points: {len(wall_points)}")
    print(f"   Wall connections: {len(wall_connections)}")

    print("\n[5/5] Creating visualization...")

    # Create figure
    fig = plt.figure(figsize=(24, 16))

    # Colors for junction types
    type_colors = plt.cm.tab20(np.linspace(0, 1, 21))

    # 1. Original Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # 2. Heatmap Sum (channels 0-9)
    ax2 = plt.subplot(2, 3, 2)
    heatmap_sum = np.sum(heatmaps[:10], axis=0)
    heatmap_vis = ax2.imshow(heatmap_sum, cmap='hot', vmin=0, vmax=1.0)

    # Overlay junction points
    for jtype, points in junctions_dict.items():
        for p in points:
            ax2.plot(p['x'], p['y'], 'o', color='red', markersize=4)

    ax2.set_title(f'Heatmap Sum (channels 0-9)\n{total_junctions} points detected',
                  fontsize=16, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(heatmap_vis, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Junction Points by Type
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    for i, (jtype, points) in enumerate(junctions_dict.items()):
        color = type_colors[i % 21]
        for p in points:
            ax3.plot(p['x'], p['y'], 'o', color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=1)

    ax3.set_title(f'Junction Points by Type\n{total_junctions} total points',
                  fontsize=16, fontweight='bold')
    ax3.axis('off')

    # Add legend
    legend_items = []
    for i, (jtype, points) in enumerate(junctions_dict.items()):
        color = type_colors[i % 21]
        legend_items.append(patches.Patch(color=color,
                           label=f'{jtype} ({len(points)})'))
    ax3.legend(handles=legend_items, loc='upper right', fontsize=8,
              framealpha=0.9, ncol=2)

    # 4. All Junction Points
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    for jtype, points in junctions_dict.items():
        for p in points:
            ax4.plot(p['x'], p['y'], 'o', color='red', markersize=6,
                    markeredgecolor='black', markeredgewidth=1)

    ax4.set_title(f'All Junction Points\n{total_junctions} points',
                  fontsize=16, fontweight='bold')
    ax4.axis('off')

    # 5. Junctions on Walls
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw wall points
    for x, y in wall_points:
        ax5.plot(x, y, 'o', color='yellow', markersize=10,
                markeredgecolor='black', markeredgewidth=2)

    ax5.set_title(f'Junctions on Walls\n{len(wall_points)} wall points',
                  fontsize=16, fontweight='bold')
    ax5.axis('off')

    # 6. Wall Graph from Junctions
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), alpha=0.3)

    # Draw connections
    for (x1, y1), (x2, y2) in wall_connections:
        ax6.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

    # Draw points
    for x, y in wall_points:
        ax6.plot(x, y, 'o', color='yellow', markersize=8,
                markeredgecolor='black', markeredgewidth=1)

    ax6.set_title(f'Wall Graph from Junctions\n{len(wall_connections)} connections',
                  fontsize=16, fontweight='bold', color='green')
    ax6.axis('off')

    # Add statistics text
    stats_text = f"""JUNCTION STATISTICS:

Total junction points: {total_junctions}
Junction types: {len(junctions_dict)}
Wall connections: {len(wall_connections)}

Top junction types:

"""
    # Sort by count
    sorted_types = sorted(junctions_dict.items(), key=lambda x: len(x[1]), reverse=True)
    for jtype, points in sorted_types[:5]:
        stats_text += f"  {jtype}: {len(points)}\n"

    ax6.text(1.15, 0.5, stats_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    plt.suptitle('Junction Analysis Pipeline', fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    output_path = 'plan_floor1_JUNCTIONS_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nJunction Statistics:")
    print(f"  Total points:  {total_junctions}")
    print(f"  Types:         {len(junctions_dict)}")
    print(f"  Wall points:   {len(wall_points)}")
    print(f"  Connections:   {len(wall_connections)}")
    print(f"\nOutput: {output_path}")

if __name__ == '__main__':
    main()
