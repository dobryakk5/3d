#!/usr/bin/env python
"""
Extract wall junctions, corners and key points from heatmaps
The model predicts 21 heatmap channels with different junction types
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
import json
import cv2

from floortrans.models import get_model

# Heatmap channel meanings (from CubiCasa5K paper)
HEATMAP_TYPES = {
    0: 'wall_junction_3way',      # T-образное пересечение
    1: 'wall_junction_4way',      # Крест (4 стены)
    2: 'wall_corner_90deg',       # Прямой угол стены
    3: 'wall_endpoint',           # Конец стены
    4: 'door_left_corner',        # Левый угол двери
    5: 'door_right_corner',       # Правый угол двери
    6: 'window_left_corner',      # Левый угол окна
    7: 'window_right_corner',     # Правый угол окна
    # ... и так далее для всех 21 типов
}

def extract_heatmaps(model, image_tensor):
    """Extract heatmap predictions from model"""
    with torch.no_grad():
        prediction = model(image_tensor)

    # Heatmaps are channels 0-20
    heatmaps = prediction[0, :21].cpu().numpy()

    return heatmaps

def find_junction_points(heatmap, threshold=0.3, min_distance=5):
    """
    Найти точки пересечений на heatmap

    Args:
        heatmap: 2D array with probabilities
        threshold: minimum probability to consider
        min_distance: minimum distance between points (pixels)

    Returns:
        List of (x, y, confidence) tuples
    """
    # Пороговая обработка
    mask = heatmap > threshold

    # Найти локальные максимумы
    from scipy.ndimage import maximum_filter
    max_filtered = maximum_filter(heatmap, size=min_distance)
    maxima = (heatmap == max_filtered) & mask

    # Извлечь координаты
    y_coords, x_coords = np.where(maxima)
    confidences = heatmap[y_coords, x_coords]

    # Собрать в список точек
    points = []
    for x, y, conf in zip(x_coords, y_coords, confidences):
        points.append({
            'x': int(x),
            'y': int(y),
            'confidence': float(conf)
        })

    return points

def extract_all_junctions(heatmaps, threshold=0.3):
    """Извлечь все типы пересечений из всех каналов"""
    all_junctions = {}

    for channel_idx in range(heatmaps.shape[0]):
        heatmap = heatmaps[channel_idx]

        # Найти точки на этом канале
        points = find_junction_points(heatmap, threshold=threshold)

        if len(points) > 0:
            junction_type = HEATMAP_TYPES.get(channel_idx, f'unknown_{channel_idx}')
            all_junctions[junction_type] = points

    return all_junctions

def connect_junctions_to_walls(junctions, wall_mask, max_distance=30):
    """
    Связать точки пересечений в линии стен

    Алгоритм:
    1. Взять все junction points
    2. Соединить близкие точки линиями
    3. Линии, которые проходят через wall_mask = стены
    """
    # Собрать все точки вместе
    all_points = []
    for junction_type, points in junctions.items():
        for point in points:
            all_points.append({
                'x': point['x'],
                'y': point['y'],
                'type': junction_type,
                'confidence': point['confidence']
            })

    if len(all_points) < 2:
        return []

    # Построить граф соседних точек
    from scipy.spatial.distance import cdist

    coords = np.array([[p['x'], p['y']] for p in all_points])
    distances = cdist(coords, coords)

    # Соединить точки которые близко друг к другу
    edges = []
    for i in range(len(all_points)):
        for j in range(i+1, len(all_points)):
            dist = distances[i, j]

            if dist < max_distance:
                # Проверить проходит ли линия через стену
                p1 = all_points[i]
                p2 = all_points[j]

                # Нарисовать линию и проверить пересечение с wall_mask
                line_mask = np.zeros_like(wall_mask, dtype=np.uint8)
                cv2.line(line_mask, (p1['x'], p1['y']), (p2['x'], p2['y']), 255, 1)

                # Если линия проходит через стену - это ребро графа стен
                overlap = np.sum((line_mask > 0) & (wall_mask > 0))
                if overlap > dist * 0.5:  # Хотя бы 50% линии на стене
                    edges.append({
                        'from': i,
                        'to': j,
                        'distance': float(dist),
                        'points': [
                            {'x': p1['x'], 'y': p1['y']},
                            {'x': p2['x'], 'y': p2['y']}
                        ]
                    })

    return edges, all_points

def main():
    print("=" * 80)
    print("EXTRACTING WALL JUNCTIONS AND CORNERS FROM HEATMAPS")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    max_size = 2048
    n_classes = 44

    print(f"\n[1/6] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("   ✓ Model loaded")

    print(f"\n[2/6] Loading image...")
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size

    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    scale_factor = orig_size[0] / img.width
    print(f"   Image: {img.size}")

    img_np = np.array(img)
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    print(f"\n[3/6] Running model inference...")
    with torch.no_grad():
        prediction = model(img_tensor)

    # Извлечь разные части предсказания
    heatmaps = prediction[0, :21].cpu().numpy()  # Углы и пересечения
    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().numpy()
    wall_mask = (rooms_pred == 2).astype(np.uint8) * 255

    print(f"   Heatmaps shape: {heatmaps.shape}")
    print(f"   Rooms shape: {rooms_pred.shape}")

    print(f"\n[4/6] Extracting junction points...")
    junctions = extract_all_junctions(heatmaps, threshold=0.3)

    total_junctions = sum(len(points) for points in junctions.values())
    print(f"   Found {total_junctions} junction points across {len(junctions)} types:")
    for junction_type, points in sorted(junctions.items()):
        if len(points) > 0:
            print(f"     - {junction_type}: {len(points)} points")

    print(f"\n[5/6] Connecting junctions to form walls...")
    edges, all_points = connect_junctions_to_walls(junctions, wall_mask, max_distance=50)
    print(f"   Found {len(edges)} wall segments from junctions")

    print(f"\n[6/6] Creating visualization...")

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

    # Row 1: Heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    # Sum all heatmaps для visualization
    heatmap_sum = np.sum(heatmaps[:10], axis=0)  # Первые 10 каналов
    ax2.imshow(img_np)
    im = ax2.imshow(heatmap_sum, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    ax2.set_title('Heatmap Sum (channels 0-9)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    # Показать отдельные каналы
    fig_channels = plt.figure(figsize=(16, 12))
    for i in range(min(9, heatmaps.shape[0])):
        ax_ch = fig_channels.add_subplot(3, 3, i+1)
        ax_ch.imshow(heatmaps[i], cmap='hot', vmin=0, vmax=1)
        ax_ch.set_title(f'Ch{i}: {HEATMAP_TYPES.get(i, "unknown")}', fontsize=8)
        ax_ch.axis('off')
    fig_channels.tight_layout()
    fig_channels.savefig('heatmap_channels.png', dpi=150, bbox_inches='tight')
    plt.close(fig_channels)

    ax3.text(0.5, 0.5, f'Individual heatmaps\nsaved to:\nheatmap_channels.png\n\n{len(junctions)} types found',
             ha='center', va='center', fontsize=12, transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis('off')

    # Row 2: Junction points
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img_np)

    # Разные цвета для разных типов
    colors = plt.cm.tab10(np.linspace(0, 1, len(junctions)))

    for (junction_type, points), color in zip(junctions.items(), colors):
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        ax4.scatter(xs, ys, c=[color], s=50, alpha=0.8,
                   label=f'{junction_type} ({len(points)})', edgecolors='white', linewidths=1)

    ax4.set_title(f'Junction Points by Type\n{total_junctions} total points',
                  fontsize=14, fontweight='bold')
    ax4.axis('off')
    if len(junctions) > 0:
        ax4.legend(fontsize=8, loc='upper right')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img_np)

    # Показать все точки одним цветом
    for points in junctions.values():
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        ax5.scatter(xs, ys, c='red', s=30, alpha=0.6)

    ax5.set_title(f'All Junction Points\n{total_junctions} points',
                  fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(img_np)
    ax6.imshow(wall_mask, cmap='Reds', alpha=0.4)

    # Показать точки на стенах
    for points in junctions.values():
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        ax6.scatter(xs, ys, c='yellow', s=30, alpha=0.8, edgecolors='black', linewidths=1)

    ax6.set_title('Junctions on Walls', fontsize=14, fontweight='bold')
    ax6.axis('off')

    # Row 3: Connected walls
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(img_np)

    # Рисуем соединения
    for edge in edges:
        p1, p2 = edge['points']
        ax7.plot([p1['x'], p2['x']], [p1['y'], p2['y']],
                'r-', linewidth=2, alpha=0.6)

    # Точки поверх линий
    if len(all_points) > 0:
        xs = [p['x'] for p in all_points]
        ys = [p['y'] for p in all_points]
        ax7.scatter(xs, ys, c='yellow', s=40, edgecolors='black', linewidths=1, zorder=10)

    ax7.set_title(f'Wall Graph from Junctions\n{len(edges)} connections',
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[2, 1])
    # Только граф стен
    graph_img = np.ones((*img_np.shape[:2], 3), dtype=np.uint8) * 255

    for edge in edges:
        p1, p2 = edge['points']
        cv2.line(graph_img, (p1['x'], p1['y']), (p2['x'], p2['y']), (0, 0, 255), 2)

    for point in all_points:
        cv2.circle(graph_img, (point['x'], point['y']), 5, (255, 0, 0), -1)

    ax8.imshow(graph_img)
    ax8.set_title('Wall Graph (Clean)', fontsize=14, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 2])
    # Statistics
    stats_text = f"""
JUNCTION STATISTICS:

Total junction points: {total_junctions}
Junction types: {len(junctions)}
Wall connections: {len(edges)}

Top junction types:
"""

    sorted_junctions = sorted(junctions.items(), key=lambda x: len(x[1]), reverse=True)
    for junction_type, points in sorted_junctions[:5]:
        stats_text += f"\n  {junction_type}: {len(points)}"

    ax9.text(0.1, 0.9, stats_text, fontsize=11, family='monospace',
             verticalalignment='top', transform=ax9.transAxes)
    ax9.axis('off')

    output_path = 'plan_floor1_JUNCTIONS.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

    # Export JSON
    results = {
        'method': 'Heatmap junction extraction',
        'statistics': {
            'total_junctions': total_junctions,
            'junction_types': len(junctions),
            'wall_connections': len(edges)
        },
        'junctions_by_type': {k: len(v) for k, v in junctions.items()},
        'junctions': junctions,
        'edges': edges,
        'all_points': all_points
    }

    json_path = 'junctions_detection.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {json_path}")

    print(f"\n{'=' * 80}")
    print("JUNCTION EXTRACTION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\n  🔴 Junction points: {total_junctions}")
    print(f"  🔗 Wall connections: {len(edges)}")
    print(f"  📊 Junction types: {len(junctions)}")
    print(f"\n  💾 Visualization: {output_path}")
    print(f"  💾 Individual heatmaps: heatmap_channels.png")
    print(f"  💾 Data: {json_path}")
    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
