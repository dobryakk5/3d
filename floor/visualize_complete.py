#!/usr/bin/env python3
"""
Complete detection visualization with detection method details:
1. Original image
2. Junctions (labeled by type: window/door/wall)
3. Windows (with detection method info)
4. Doors (with detection method info)
5. Walls (DL vs Hatching texture)
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

# Import detection functions
exec(open('cubicasa_vectorize.py').read().split('if __name__')[0])

def categorize_junctions(junctions_dict):
    """Collect all junctions without categorization"""
    all_junctions = []
    for jtype, points in junctions_dict.items():
        for p in points:
            all_junctions.append({**p, 'type': jtype})
    return all_junctions

def analyze_detection_method(detection, wall_segments, all_junctions):
    """Determine how a detection was validated"""
    methods = ['DL']  # All start with DL

    x, y, w, h = detection['bbox']
    center_x, center_y = x + w/2, y + h/2

    # Check if on wall
    on_wall = False
    for seg in wall_segments:
        x1, y1 = seg['start']
        x2, y2 = seg['end']

        # Check distance to wall line
        dx = x2 - x1
        dy = y2 - y1
        line_len_sq = dx*dx + dy*dy

        if line_len_sq == 0:
            continue

        t = max(0, min(1, ((center_x - x1) * dx + (center_y - y1) * dy) / line_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.sqrt((center_x - proj_x)**2 + (center_y - proj_y)**2)

        if dist < 30:
            on_wall = True
            break

    if on_wall:
        methods.append('Wall-aligned')

    # Aspect ratio check removed - accept all detections

    # Check for nearby junctions
    has_junctions = False
    for j in all_junctions:
        jx, jy = j['x'], j['y']
        if abs(jx - center_x) < w + 50 and abs(jy - center_y) < h + 50:
            has_junctions = True
            break

    if has_junctions:
        methods.append('Junctions')

    return methods

def main(perimeter_algorithm='alpha_shape', output_suffix=''):
    """
    Main visualization function

    Args:
        perimeter_algorithm: 'alpha_shape', 'gift_wrapping', or 'boundary_trace'
        output_suffix: suffix for output filename (e.g., '_v1', '_v2')
    """
    print("="*80)
    print(f"COMPLETE DETECTION VISUALIZATION - {perimeter_algorithm}")
    print("="*80)

    image_path = 'plan_floor1.jpg'

    # Load model
    print("\n[1/5] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Preprocess
    print("\n[2/5] Preprocessing...")
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

    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Inference
    print("\n[3/5] Running DL inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Extract icons
    icons_logits = prediction[0, 33:44]
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    door_prob = icons_pred[2]
    window_prob = icons_pred[1]

    threshold = 0.3
    door_mask = (door_prob > threshold).astype(np.uint8)
    window_mask = (window_prob > threshold).astype(np.uint8)

    _, doors = refine_dl_detections(door_mask, min_size=50, max_size=6000)
    _, windows = refine_dl_detections(window_mask, min_size=50, max_size=6000)

    # Detect walls
    print("\n[4/5] Detecting walls...")
    wall_mask_hatching = detect_hatching(img_np, kernel_size=25, density_threshold=0.2)
    wall_segments_hatching = extract_wall_segments(wall_mask_hatching, min_length=50)

    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()
    wall_mask_dl = (rooms_pred == 2).astype(np.uint8) * 255
    wall_segments_dl = extract_wall_segments(wall_mask_dl, min_length=50)

    # Tag walls with source
    for seg in wall_segments_hatching:
        seg['source'] = 'hatching'
    for seg in wall_segments_dl:
        seg['source'] = 'DL'

    wall_segments = wall_segments_hatching + wall_segments_dl

    # Extract junctions (without type categorization)
    heatmaps = prediction[0, :21].cpu().data.numpy()
    junctions_dict = extract_junctions(prediction, threshold=threshold)
    all_junctions = categorize_junctions(junctions_dict)

    # Filter windows/doors
    windows = remove_nested_boxes(windows, iou_threshold=0.15)
    doors = remove_nested_boxes(doors, iou_threshold=0.5)
    windows = filter_windows_on_walls(windows, wall_segments)

    # Filter doors: reject if window_prob > door_prob at center
    def filter_door_vs_window(doors, door_prob, window_prob):
        """Reject doors where model predicts higher window probability"""
        valid_doors = []
        for door in doors:
            x, y, w, h = door['bbox']
            cx, cy = int(x + w/2), int(y + h/2)

            if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
                d_conf = door_prob[cy, cx]
                w_conf = window_prob[cy, cx]

                # Keep only if door confidence is higher than window
                if d_conf > w_conf:
                    valid_doors.append(door)
            else:
                # Keep if out of bounds (shouldn't happen)
                valid_doors.append(door)

        return valid_doors

    doors_before = len(doors)
    doors = filter_door_vs_window(doors, door_prob, window_prob)
    print(f"   After door/window disambiguation: {len(doors)}/{doors_before} doors kept")

    # Analyze detection methods
    print("\n[5/5] Analyzing detection methods...")
    for w in windows:
        w['methods'] = analyze_detection_method(w, wall_segments, all_junctions)
    for d in doors:
        d['methods'] = analyze_detection_method(d, wall_segments, all_junctions)

    print(f"   Windows: {len(windows)}, Doors: {len(doors)}")
    print(f"   Walls: {len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL")
    print(f"   Junctions: {len(all_junctions)}")

    # Print door detection details with DL predictions
    print("\nDoor detection details:")
    for i, d in enumerate(doors, 1):
        x, y, w, h = d['bbox']
        methods = ' + '.join(d['methods'])
        aspect = max(w, h) / max(min(w, h), 1)

        # Check DL prediction at door location
        cx, cy = int(x + w/2), int(y + h/2)
        if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
            door_conf = door_prob[cy, cx]
            window_conf = window_prob[cy, cx]
            print(f"   D{i}: bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f}) aspect={aspect:.1f} door_prob={door_conf:.3f} window_prob={window_conf:.3f} methods=[{methods}]")
        else:
            print(f"   D{i}: bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f}) aspect={aspect:.1f} methods=[{methods}]")

    # Build closed contour and filter noise
    print("\nBuilding house contour...")

    # Debug: check connectivity
    print(f"   Total wall segments before filtering: {len(wall_segments)}")

    # Filter isolated wall segments - keep ALL connected components (not just largest)
    def filter_connected_walls(wall_segments, min_component_size=3):
        """Keep only wall segments that belong to large connected components"""
        if not wall_segments:
            return []

        # Build adjacency graph
        from collections import defaultdict
        graph = defaultdict(set)

        def points_close(p1, p2, threshold=15):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < threshold

        # Connect segments that share endpoints
        for i, seg1 in enumerate(wall_segments):
            for j, seg2 in enumerate(wall_segments):
                if i >= j:
                    continue

                # Check if segments are connected
                if (points_close(seg1['start'], seg2['start']) or
                    points_close(seg1['start'], seg2['end']) or
                    points_close(seg1['end'], seg2['start']) or
                    points_close(seg1['end'], seg2['end'])):
                    graph[i].add(j)
                    graph[j].add(i)

        # Find connected components using DFS
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(len(wall_segments)):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(component)

        # Keep ALL components that meet size threshold (not just largest!)
        valid_segments = []
        if components:
            print(f"   Found {len(components)} connected components:")
            for idx, comp in enumerate(components):
                print(f"     Component {idx+1}: {len(comp)} segments")
                if len(comp) >= min_component_size:
                    valid_segments.extend([wall_segments[i] for i in comp])

        return valid_segments if valid_segments else wall_segments

    wall_segments_before = len(wall_segments)
    wall_segments = filter_connected_walls(wall_segments, min_component_size=5)
    print(f"   Filtered isolated walls: {len(wall_segments)}/{wall_segments_before} segments kept")

    # Collect all wall points
    all_wall_points = []
    for seg in wall_segments:
        all_wall_points.append(seg['start'])
        all_wall_points.append(seg['end'])

    if all_wall_points:
        all_wall_points = np.array(all_wall_points)

        # Find bounding box of main structure
        x_coords = all_wall_points[:, 0]
        y_coords = all_wall_points[:, 1]

        # Use percentiles to ignore outliers (moderate filtering)
        x_min = np.percentile(x_coords, 5)
        x_max = np.percentile(x_coords, 95)
        y_min = np.percentile(y_coords, 5)
        y_max = np.percentile(y_coords, 95)

        # Add margin
        margin = 50
        x_min = max(0, int(x_min - margin))
        x_max = min(img_display.shape[1], int(x_max + margin))
        y_min = max(0, int(y_min - margin))
        y_max = min(img_display.shape[0], int(y_max + margin))

        crop_bounds = (x_min, x_max, y_min, y_max)

        # Filter wall segments - keep only those FULLY inside main structure
        main_wall_segments = []
        for seg in wall_segments:
            x1, y1 = seg['start']
            x2, y2 = seg['end']

            # Both endpoints must be inside bounds
            start_inside = (x_min <= x1 <= x_max and y_min <= y1 <= y_max)
            end_inside = (x_min <= x2 <= x_max and y_min <= y2 <= y_max)

            if start_inside and end_inside:
                main_wall_segments.append(seg)

        # Filter junctions - keep only those inside main structure
        main_junctions = []
        for j in all_junctions:
            if x_min <= j['x'] <= x_max and y_min <= j['y'] <= y_max:
                main_junctions.append(j)

        print(f"   Filtered: {len(main_wall_segments)}/{len(wall_segments)} walls inside main structure")
        print(f"   Crop bounds: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    else:
        crop_bounds = None
        main_wall_segments = wall_segments
        main_junctions = all_junctions

    # Create visualization
    print("\nCreating visualization...")

    fig = plt.figure(figsize=(48, 32))

    # Compute hatching density map for visualization
    def compute_hatching_density(image, kernel_size=25):
        """Compute hatching density map (filter out noise like text, stairs)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        def create_line_kernel(length, angle_deg):
            angle_rad = np.deg2rad(angle_deg)
            kernel = np.zeros((length, length), dtype=np.uint8)
            center = length // 2
            for i in range(length):
                offset = i - center
                x = int(center + offset * np.cos(angle_rad))
                y = int(center + offset * np.sin(angle_rad))
                if 0 <= x < length and 0 <= y < length:
                    kernel[y, x] = 1
            return kernel

        # Only diagonal angles for wall hatching (exclude horizontal/vertical for stairs/text)
        angles = [45, -45, 135, -135]
        all_lines = np.zeros_like(binary, dtype=np.float32)

        for angle in angles:
            kernel = create_line_kernel(kernel_size, angle)
            detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            all_lines += detected.astype(np.float32)

        # Use larger averaging kernel to smooth out small text patterns
        kernel_avg = np.ones((30, 30), dtype=np.float32) / 900
        density = cv2.filter2D(all_lines, -1, kernel_avg)
        density_normalized = density / density.max() if density.max() > 0 else density

        # Apply threshold to remove weak responses (text, noise)
        density_cleaned = np.where(density_normalized > 0.15, density_normalized, 0)
        density_cleaned = density_cleaned / density_cleaned.max() if density_cleaned.max() > 0 else density_cleaned

        return density_cleaned

    hatching_density = compute_hatching_density(img_np, kernel_size=25)

    # 1. Original
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    ax1.set_title('1. Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Junctions (all types)
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw all junctions in same color
    for j in all_junctions:
        ax2.plot(j['x'], j['y'], 'o', color='red',
                markersize=10, markeredgecolor='black', markeredgewidth=1.5)

    ax2.set_title(f"2. Junctions\n{len(all_junctions)} detected", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Windows
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']

        # Draw box
        rect = patches.Rectangle((x, y), width, height, linewidth=3,
                                edgecolor='cyan', facecolor='cyan', alpha=0.3)
        ax3.add_patch(rect)

        # Label
        methods_str = ' + '.join(w['methods'])
        label = f"W{i}"
        ax3.text(x + width/2, y - 5, label, fontsize=10, fontweight='bold',
                color='blue', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='blue', linewidth=0.5))

    ax3.set_title(f'3. Windows ({len(windows)} detected)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Add detection methods legend
    methods_text = "Detection methods:\n"
    for i, w in enumerate(windows, 1):
        methods_text += f"W{i}: {' + '.join(w['methods'])}\n"

    ax3.text(1.02, 0.5, methods_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
            family='monospace')

    # 4. Doors
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    for i, d in enumerate(doors, 1):
        x, y, width, height = d['bbox']

        # Draw box
        rect = patches.Rectangle((x, y), width, height, linewidth=3,
                                edgecolor='green', facecolor='green', alpha=0.3)
        ax4.add_patch(rect)

        # Label
        label = f"D{i}"
        ax4.text(x + width/2, y - 5, label, fontsize=10, fontweight='bold',
                color='darkgreen', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='green', linewidth=0.5))

    ax4.set_title(f'4. Doors ({len(doors)} detected)', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Add detection methods
    methods_text = "Detection methods:\n"
    for i, d in enumerate(doors, 1):
        methods_text += f"D{i}: {' + '.join(d['methods'])}\n"

    ax4.text(1.02, 0.5, methods_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            family='monospace')

    # 5. Walls (separated by source)
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), alpha=0.4)

    # Draw hatching walls in red
    for seg in wall_segments_hatching:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        ax5.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7, label='_nolegend_')

    # Draw DL walls in blue
    for seg in wall_segments_dl:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        ax5.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7, label='_nolegend_')

    title = f"5. Walls by Source\n"
    title += f"Hatching: {len(wall_segments_hatching)}  DL: {len(wall_segments_dl)}"
    ax5.set_title(title, fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Legend
    legend_patches = [
        patches.Patch(color='red', label=f'Hatching texture ({len(wall_segments_hatching)})'),
        patches.Patch(color='blue', label=f'DL segmentation ({len(wall_segments_dl)})')
    ]
    ax5.legend(handles=legend_patches, loc='upper right', fontsize=10)

    # 6. Combined result
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw walls
    for seg in wall_segments:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        color = 'red' if seg['source'] == 'hatching' else 'blue'
        ax6.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.5)

    # Draw windows
    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']
        rect = patches.Rectangle((x, y), width, height, linewidth=2,
                                edgecolor='cyan', facecolor='none')
        ax6.add_patch(rect)
        ax6.text(x + width/2, y + height/2, f"W{i}", fontsize=8,
                color='blue', ha='center', va='center', fontweight='bold')

    # Draw doors
    for i, d in enumerate(doors, 1):
        x, y, width, height = d['bbox']
        rect = patches.Rectangle((x, y), width, height, linewidth=2,
                                edgecolor='green', facecolor='none')
        ax6.add_patch(rect)
        ax6.text(x + width/2, y + height/2, f"D{i}", fontsize=8,
                color='darkgreen', ha='center', va='center', fontweight='bold')

    # Draw junctions
    for j in all_junctions:
        ax6.plot(j['x'], j['y'], 'o', color='red',
                markersize=4, markeredgecolor='black', markeredgewidth=0.5)

    ax6.set_title('6. Complete Result', fontsize=14, fontweight='bold', color='darkgreen')
    ax6.axis('off')

    # 7. FINAL - Full view with exterior perimeter
    ax7 = plt.subplot(2, 4, 7)

    # Show full image (no crop)
    ax7.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw all filtered walls
    for seg in wall_segments:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        color = 'red' if seg['source'] == 'hatching' else 'blue'
        ax7.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.3)

    # Draw windows
    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']
        rect = patches.Rectangle((x, y), width, height,
                                linewidth=2, edgecolor='cyan', facecolor='none')
        ax7.add_patch(rect)
        ax7.text(x + width/2, y + height/2, f"W{i}",
                fontsize=8, color='blue', ha='center', va='center',
                fontweight='bold')

    # Draw doors
    for i, d in enumerate(doors, 1):
        x, y, width, height = d['bbox']
        rect = patches.Rectangle((x, y), width, height,
                                linewidth=2, edgecolor='green', facecolor='none')
        ax7.add_patch(rect)
        ax7.text(x + width/2, y + height/2, f"D{i}",
                fontsize=8, color='darkgreen', ha='center', va='center',
                fontweight='bold')

    # Draw junctions
    for j in all_junctions:
        ax7.plot(j['x'], j['y'], 'o',
                color='red', markersize=5,
                markeredgecolor='black', markeredgewidth=0.5)

    # Build exterior perimeter from walls using selected algorithm
    if wall_segments:
        try:
            if perimeter_algorithm == 'alpha_shape':
                # ALGORITHM 1: Alpha-shape (concave hull)
                print(f"   Using algorithm: Alpha-shape (concave hull)")

                # Collect all wall endpoints
                all_wall_points = []
                for seg in wall_segments:
                    all_wall_points.append(seg['start'])
                    all_wall_points.append(seg['end'])

                all_wall_points = np.array(all_wall_points)

                # Try alpha-shape with scipy
                from scipy.spatial import Delaunay
                tri = Delaunay(all_wall_points)

                # Build edge set with length threshold
                alpha = 150  # Threshold distance
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        p1_idx = simplex[i]
                        p2_idx = simplex[(i+1)%3]
                        p1 = all_wall_points[p1_idx]
                        p2 = all_wall_points[p2_idx]
                        length = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

                        if length < alpha:
                            edge = tuple(sorted([p1_idx, p2_idx]))
                            edges.add(edge)

                # Find boundary edges (appear only once in triangle set)
                from collections import defaultdict
                edge_count = defaultdict(int)
                for simplex in tri.simplices:
                    for i in range(3):
                        p1_idx = simplex[i]
                        p2_idx = simplex[(i+1)%3]
                        edge = tuple(sorted([p1_idx, p2_idx]))
                        if edge in edges:
                            edge_count[edge] += 1

                # Boundary edges appear in only one triangle
                boundary_edges = [e for e, count in edge_count.items() if count == 1]

                # Build adjacency graph
                graph = defaultdict(list)
                for e in boundary_edges:
                    graph[e[0]].append(e[1])
                    graph[e[1]].append(e[0])

                # Trace boundary
                if boundary_edges:
                    start_idx = min(graph.keys(), key=lambda i: (all_wall_points[i][0], all_wall_points[i][1]))
                    perimeter_indices = [start_idx]
                    current = start_idx
                    visited = {start_idx}

                    while len(visited) < len(graph):
                        neighbors = [n for n in graph[current] if n not in visited]
                        if not neighbors:
                            break
                        next_idx = neighbors[0]
                        perimeter_indices.append(next_idx)
                        visited.add(next_idx)
                        current = next_idx

                    perimeter_points = all_wall_points[perimeter_indices]
                else:
                    # Fallback to convex hull
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(all_wall_points)
                    perimeter_points = all_wall_points[hull.vertices]

            elif perimeter_algorithm == 'gift_wrapping':
                # ALGORITHM 2: Gift Wrapping from walls + openings + hatching-detected pillars
                print(f"   Using algorithm: Gift Wrapping (walls + openings + pillar detection)")

                # Collect boundary points
                all_boundary_points = []

                # 1. Add wall segment endpoints (already filtered from isolated components)
                for seg in wall_segments:
                    all_boundary_points.append(seg['start'])
                    all_boundary_points.append(seg['end'])

                # 2. Add window corners
                for w in windows:
                    x, y, width, height = w['bbox']
                    all_boundary_points.extend([
                        (x, y), (x + width, y),
                        (x, y + height), (x + width, y + height)
                    ])

                # 3. Add door corners
                for d in doors:
                    x, y, width, height = d['bbox']
                    all_boundary_points.extend([
                        (x, y), (x + width, y),
                        (x, y + height), (x + width, y + height)
                    ])

                # 4. Detect support pillars from hatching density (small isolated high-density regions)
                # Find contours in hatching density map with LOWER threshold
                hatching_binary = (hatching_density > 0.2).astype(np.uint8) * 255  # Lowered from 0.3
                contours, _ = cv2.findContours(hatching_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                pillar_count = 0
                pillar_debug = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Small isolated regions (15x15 to 200x200 pixels) = potential pillars
                    if 225 < area < 40000:  # Lowered min from 400
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Check aspect ratio (pillars are roughly square)
                        aspect = max(w, h) / max(min(w, h), 1)
                        if aspect < 3.0:  # Relaxed from 2.5
                            # Add corners of pillar
                            all_boundary_points.extend([
                                (x, y), (x + w, y),
                                (x, y + h), (x + w, y + h)
                            ])
                            pillar_count += 1
                            pillar_debug.append(f"({x},{y},{w}x{h},area={area:.0f},aspect={aspect:.1f})")
                        else:
                            # Debug: why rejected
                            if area < 5000:  # Log small rejected regions
                                print(f"      Rejected pillar (aspect={aspect:.1f}): {x},{y} size={w}x{h}")
                    elif 100 < area < 225:
                        # Debug very small regions
                        x, y, w, h = cv2.boundingRect(cnt)
                        print(f"      Too small region: {x},{y} size={w}x{h} area={area:.0f}")

                all_boundary_points = np.array(all_boundary_points)
                wall_endpoints = len(wall_segments) * 2
                print(f"   Using {len(all_boundary_points)} points ({wall_endpoints} wall endpoints + {len(windows)*4} window corners + {len(doors)*4} door corners + {pillar_count} pillars)")
                if pillar_count > 0:
                    print(f"   Detected {pillar_count} support pillars from hatching texture:")
                    for p in pillar_debug:
                        print(f"      Pillar: {p}")

                def pt_key(pt):
                    return (round(pt[0]), round(pt[1]))

                def points_close(p1, p2, threshold=15):
                    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < threshold

                # Apply Gift Wrapping to ALL boundary points (creates single unified perimeter)
                def gift_wrapping(points):
                    n = len(points)
                    if n < 3:
                        return points

                    hull = []
                    start = np.argmin(points[:, 0])  # Leftmost point
                    current = start

                    while True:
                        hull.append(current)
                        next_point = (current + 1) % n

                        for i in range(n):
                            if i == current:
                                continue
                            # Cross product to find leftmost turn
                            v1 = points[i] - points[current]
                            v2 = points[next_point] - points[current]
                            cross = v1[0] * v2[1] - v1[1] * v2[0]
                            if cross > 0:
                                next_point = i

                        current = next_point
                        if current == start:
                            break

                    return points[hull]

                # Create unified perimeter
                perimeter_points = gift_wrapping(all_boundary_points)
                print(f"   Built unified perimeter with {len(perimeter_points)} vertices")

            else:  # 'boundary_trace'
                # ALGORITHM 3: Boundary tracing along wall segments
                print(f"   Using algorithm: Boundary tracing (right-hand rule)")

                # This is the existing implementation
                # Build graph of connected wall segments
                from collections import defaultdict
                graph = defaultdict(list)

                # Round coordinates to handle floating point issues
                def pt_key(pt):
                    return (round(pt[0]), round(pt[1]))

                # Build adjacency graph from filtered walls
                for seg in wall_segments:
                    start_key = pt_key(seg['start'])
                    end_key = pt_key(seg['end'])
                    graph[start_key].append(end_key)
                    graph[end_key].append(start_key)

                # Find exterior perimeter by walking along outer boundary
                # Start from leftmost-bottommost point (for right-hand rule going DOWN first)
                all_points = list(graph.keys())
                start_pt = min(all_points, key=lambda p: (p[0], -p[1]))  # Left, then bottom

                # Trace perimeter using right-hand rule (exterior on right)
                # Starting from left-bottom and going clockwise
                perimeter = [start_pt]
                visited_edges = set()
                current = start_pt
                prev_direction = np.pi / 2  # Start going down (positive Y)

                max_iterations = len(wall_segments) * 3
                iterations = 0

                while iterations < max_iterations:
                    iterations += 1
                    neighbors = graph[current]

                    if len(neighbors) == 0:
                        break

                    # Choose next point: prefer rightmost turn (exterior boundary)
                    next_pt = None
                    if prev_direction is not None:
                        # Calculate angles to all neighbors
                        angles = []
                        for neighbor in neighbors:
                            edge = tuple(sorted([current, neighbor]))
                            if edge in visited_edges:
                                continue

                            # Direction to neighbor
                            dx = neighbor[0] - current[0]
                            dy = neighbor[1] - current[1]
                            angle = np.arctan2(dy, dx)

                            # Relative angle from previous direction
                            relative_angle = (angle - prev_direction + 2*np.pi) % (2*np.pi)
                            angles.append((relative_angle, neighbor, edge))

                        if angles:
                            # Pick rightmost turn (smallest positive angle)
                            angles.sort()
                            _, next_pt, next_edge = angles[0]
                            visited_edges.add(next_edge)

                    if next_pt is None:
                        # First step or no unvisited neighbors with angle preference
                        for neighbor in neighbors:
                            edge = tuple(sorted([current, neighbor]))
                            if edge not in visited_edges:
                                next_pt = neighbor
                                visited_edges.add(edge)
                                break

                    if next_pt is None:
                        break  # No more unvisited edges

                    if next_pt == start_pt and len(perimeter) > 2:
                        break  # Completed loop

                    # Update direction for next iteration
                    prev_direction = np.arctan2(next_pt[1] - current[1], next_pt[0] - current[0])

                    perimeter.append(next_pt)
                    current = next_pt

                perimeter_points = np.array(perimeter)

            # Common code for all algorithms: draw the perimeter
            perimeter_x = np.append(perimeter_points[:, 0], perimeter_points[0, 0])
            perimeter_y = np.append(perimeter_points[:, 1], perimeter_points[0, 1])

            # Estimate wall thickness from wall segments
            wall_thickness = 10  # Default in pixels

            # Better estimate: measure distance between parallel wall segments
            for i, seg1 in enumerate(wall_segments[:20]):  # Sample first 20
                p1_start = np.array(seg1['start'])
                p1_end = np.array(seg1['end'])
                vec1 = p1_end - p1_start
                len1 = np.linalg.norm(vec1)

                if len1 < 10:
                    continue

                vec1_norm = vec1 / len1

                # Find parallel segments
                for seg2 in wall_segments[i+1:]:
                    p2_start = np.array(seg2['start'])
                    p2_end = np.array(seg2['end'])
                    vec2 = p2_end - p2_start
                    len2 = np.linalg.norm(vec2)

                    if len2 < 10:
                        continue

                    vec2_norm = vec2 / len2

                    # Check if parallel (dot product ~ ±1)
                    dot = abs(np.dot(vec1_norm, vec2_norm))
                    if dot > 0.95:  # Parallel
                        # Measure perpendicular distance
                        to_seg2 = p2_start - p1_start
                        dist_to_seg2 = np.linalg.norm(to_seg2)

                        if dist_to_seg2 > 0:
                            to_seg2_norm = to_seg2 / dist_to_seg2
                            # Cross product for 2D vectors
                            perp_dist = abs(vec1_norm[0] * to_seg2_norm[1] - vec1_norm[1] * to_seg2_norm[0]) * dist_to_seg2

                            if 5 < perp_dist < 50:  # Reasonable wall thickness
                                wall_thickness = max(wall_thickness, perp_dist)
                                break

            wall_thickness = min(wall_thickness, 20)  # Cap at 20 pixels

            # Make line 3x thinner
            display_thickness = wall_thickness / 3

            # Draw perimeter line
            ax7.plot(perimeter_x, perimeter_y, color='darkblue',
                    linewidth=display_thickness, alpha=0.8,
                    linestyle='-', solid_capstyle='round', solid_joinstyle='round',
                    label=f'Unified Exterior Perimeter', zorder=5)

            ax7.legend(loc='upper right', fontsize=10)

        except Exception as e:
            print(f"   Warning: Could not create perimeter: {e}")

    ax7.set_title(f'7. FINAL - Exterior Perimeter\n({len(wall_segments)} walls, full view)',
                 fontsize=14, fontweight='bold', color='blue')
    ax7.axis('off')

    # 8. Hatching Texture Detection
    ax8 = plt.subplot(2, 4, 8)
    # Show hatching density as heatmap
    im = ax8.imshow(hatching_density, cmap='hot', vmin=0, vmax=1)
    ax8.set_title('8. Hatching Density Map\n(Wall Texture Detection)', fontsize=14, fontweight='bold')
    ax8.axis('off')

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Density')

    # Overall title
    plt.suptitle('Floor Plan Detection Analysis - Method Breakdown',
                fontsize=18, fontweight='bold', y=0.99)

    plt.tight_layout(pad=0.05, h_pad=0.05, w_pad=0.05)

    # Save with reduced DPI to avoid exceeding max size
    output_path = f'plan_floor1_COMPLETE_analysis{output_suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Windows:   {len(windows)}")
    print(f"  Doors:     {len(doors)}")
    print(f"  Walls:     {len(wall_segments)} ({len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL)")
    print(f"  Junctions: {len(all_junctions)}")
    print(f"\nOutput: {output_path}")

if __name__ == '__main__':
    import sys

    # v11: Debug pillar detection with relaxed thresholds
    main(perimeter_algorithm='gift_wrapping', output_suffix='_v11_debug_pillars')
