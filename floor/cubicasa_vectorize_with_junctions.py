#!/usr/bin/env python
"""
Enhanced CubiCasa vectorization with junction-aware wall extraction.

Key improvement: Walls are traced THROUGH junctions, ensuring proper T-junction connectivity.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy import ndimage
from scipy.ndimage import maximum_filter
import cv2
import svgwrite
import json
import networkx as nx
import math

from floortrans.models import get_model


def multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1]):
    """Run inference at multiple scales and average"""
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    predictions = []

    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=True)
        else:
            scaled = img_tensor

        with torch.no_grad():
            pred = model(scaled)

        if scale != 1.0:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        predictions.append(pred)

    return torch.stack(predictions).mean(dim=0)


def refine_dl_detections(mask, min_size=50, max_size=6000, debug=False):
    """Clean up DL predictions"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    labeled, num = ndimage.label(mask_closed)
    refined = np.zeros_like(mask)

    detections = []
    filtered_sizes = []

    for i in range(1, num + 1):
        region = labeled == i
        size = region.sum()

        if min_size <= size <= max_size:
            refined[region] = True
            rows, cols = np.where(region)
            x_min, x_max = int(cols.min()), int(cols.max())
            y_min, y_max = int(rows.min()), int(rows.max())

            detections.append({
                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                'center': (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
                'area': int(size)
            })
        else:
            if debug:
                filtered_sizes.append(size)

    if debug and filtered_sizes:
        print(f"   Filtered {len(filtered_sizes)} regions: sizes = {sorted(filtered_sizes)}")

    return refined, detections


def remove_nested_boxes(detections, iou_threshold=0.5):
    """Remove overlapping boxes using IoU"""
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d['area'], reverse=True)

    keep = []
    for det in sorted_dets:
        x1, y1, w1, h1 = det['bbox']
        is_overlapping = False

        for kept_det in keep:
            x2, y2, w2, h2 = kept_det['bbox']

            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0

                if iou > iou_threshold:
                    is_overlapping = True
                    break

        if not is_overlapping:
            keep.append(det)

    return keep


def detect_hatching(image, kernel_size=25, density_threshold=0.2):
    """Detect wall hatching patterns"""
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

    angles = [45, -45, 135, -135]
    all_lines = np.zeros_like(binary, dtype=np.float32)

    for angle in angles:
        kernel = create_line_kernel(kernel_size, angle)
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        all_lines += detected.astype(np.float32)

    kernel_avg = np.ones((20, 20), dtype=np.float32) / 400
    density = cv2.filter2D(all_lines, -1, kernel_avg)
    density_normalized = density / density.max() if density.max() > 0 else density

    wall_mask = (density_normalized > density_threshold).astype(np.uint8) * 255
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return wall_mask


def skeletonize_walls(wall_mask):
    """Skeletonize wall mask to get thin lines"""
    from scipy.ndimage import binary_erosion
    skeleton = wall_mask.copy()
    for _ in range(3):
        skeleton = binary_erosion(skeleton).astype(np.uint8) * 255
    return skeleton


def find_junction_points(heatmap, threshold=0.3, min_distance=5):
    """Find junction points from heatmap"""
    mask = heatmap > threshold
    max_filtered = maximum_filter(heatmap, size=min_distance)
    maxima = (heatmap == max_filtered) & mask
    y_coords, x_coords = np.where(maxima)
    confidences = heatmap[y_coords, x_coords]

    points = []
    for x, y, conf in zip(x_coords, y_coords, confidences):
        points.append({'x': int(x), 'y': int(y), 'confidence': float(conf)})

    return points


def extract_junctions(prediction, threshold=0.3):
    """Extract junction points from heatmaps"""
    heatmaps = prediction[0, :21].cpu().numpy()

    HEATMAP_TYPES = {
        0: 'wall_junction_3way',
        1: 'wall_junction_4way',
        2: 'wall_corner_90deg',
        3: 'wall_endpoint',
        4: 'door_left_corner',
        5: 'door_right_corner',
        6: 'window_left_corner',
        7: 'window_right_corner',
    }

    all_junctions = {}
    for ch_idx in range(21):
        points = find_junction_points(heatmaps[ch_idx], threshold)
        if points:
            type_name = HEATMAP_TYPES.get(ch_idx, f'unknown_{ch_idx}')
            all_junctions[type_name] = points

    return all_junctions


def find_nearest_skeleton_point(skeleton, point, radius=10):
    """Find nearest skeleton point to given location"""
    x, y = point
    h, w = skeleton.shape

    # Check point itself
    if 0 <= x < w and 0 <= y < h and skeleton[y, x] > 0:
        return (x, y)

    # Search in radius
    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                    return (nx, ny)

    return None


def is_connected_by_skeleton(skeleton, point1, point2, max_distance):
    """Check if two points are connected by skeleton using BFS"""
    x1, y1 = point1
    x2, y2 = point2

    h, w = skeleton.shape

    # Find nearest skeleton points
    start = find_nearest_skeleton_point(skeleton, (x1, y1), radius=10)
    if start is None:
        return False

    end = find_nearest_skeleton_point(skeleton, (x2, y2), radius=10)
    if end is None:
        return False

    # BFS from start to end
    visited = set()
    queue = [start]
    visited.add(start)

    max_steps = int(max_distance * 1.5)
    steps = 0

    while queue and steps < max_steps:
        steps += 1
        current = queue.pop(0)
        cx, cy = current

        # Reached goal?
        if abs(cx - end[0]) <= 3 and abs(cy - end[1]) <= 3:
            return True

        # Check 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < w and 0 <= ny < h):
                    continue

                if (nx, ny) in visited:
                    continue

                if skeleton[ny, nx] > 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False


def build_junction_graph(junctions_list, skeleton, max_distance=80):
    """Build graph: vertices = junctions, edges = walls"""
    G = nx.Graph()

    # Add vertices
    for i, junc in enumerate(junctions_list):
        G.add_node(i, **junc)

    print(f"   Building graph from {len(junctions_list)} junctions...")

    # For each junction, find neighbors
    edge_count = 0
    for i, junc1 in enumerate(junctions_list):
        x1, y1 = junc1['x'], junc1['y']

        for j, junc2 in enumerate(junctions_list):
            if i >= j:  # Avoid duplicates and self-loops
                continue

            x2, y2 = junc2['x'], junc2['y']

            # Distance
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if dist > max_distance:
                continue

            # Check if connected by skeleton
            if is_connected_by_skeleton(skeleton, (x1, y1), (x2, y2), max_distance):
                G.add_edge(i, j, weight=dist)
                edge_count += 1

    print(f"   Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    return G


def extract_wall_segments_from_graph(graph, junctions_list):
    """Extract wall segments from junction graph"""
    segments = []

    for edge in graph.edges():
        i, j = edge
        junc1 = junctions_list[i]
        junc2 = junctions_list[j]

        segment = {
            'start': (junc1['x'], junc1['y']),
            'end': (junc2['x'], junc2['y']),
            'start_type': junc1.get('type', 'unknown'),
            'end_type': junc2.get('type', 'unknown'),
            'length': float(math.sqrt(
                (junc2['x'] - junc1['x'])**2 + (junc2['y'] - junc1['y'])**2
            ))
        }
        segments.append(segment)

    return segments


def extract_wall_segments_junction_aware(wall_mask, junctions_dict, use_graph=True):
    """
    Extract wall segments using junction information.

    If use_graph=True, builds junction graph and traces walls between junctions.
    If use_graph=False, falls back to traditional contour-based extraction.
    """
    # Convert junctions dict to flat list with types
    junctions_list = []
    for jtype, points in junctions_dict.items():
        if 'wall' in jtype:  # Only wall junctions
            for pt in points:
                pt_with_type = pt.copy()
                pt_with_type['type'] = jtype
                junctions_list.append(pt_with_type)

    print(f"   Found {len(junctions_list)} wall junction points")

    if not use_graph or len(junctions_list) < 2:
        print("   Falling back to contour-based extraction")
        # Traditional approach
        from scipy.ndimage import binary_erosion
        skeleton = wall_mask.copy()
        for _ in range(3):
            skeleton = binary_erosion(skeleton).astype(np.uint8) * 255

        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        segments = []
        for contour in contours:
            if len(contour) >= 2:
                contour = contour.squeeze()
                if len(contour.shape) == 1 or len(contour) < 2:
                    continue

                epsilon = 3.0
                approx = cv2.approxPolyDP(contour, epsilon, False)

                if len(approx) >= 2:
                    for i in range(len(approx) - 1):
                        pt1 = (int(approx[i][0][0]), int(approx[i][0][1]))
                        pt2 = (int(approx[i+1][0][0]), int(approx[i+1][0][1]))
                        length = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
                        if length >= 50:
                            segments.append({
                                'start': pt1,
                                'end': pt2,
                                'length': float(length)
                            })
        return segments

    # Junction-aware approach
    print("   Using junction-aware graph extraction")

    # Skeletonize
    skeleton = skeletonize_walls(wall_mask)

    # Build junction graph
    graph = build_junction_graph(junctions_list, skeleton, max_distance=80)

    # Extract segments from graph
    segments = extract_wall_segments_from_graph(graph, junctions_list)

    return segments


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def is_on_wall(window, wall_segments, max_distance=30):
    """Check if window is aligned with any wall"""
    x, y, w, h = window['bbox']

    points = [
        (x + w/2, y + h/2),
        (x + w/2, y),
        (x + w/2, y + h),
        (x, y + h/2),
        (x + w, y + h/2),
    ]

    for segment in wall_segments:
        x1, y1 = segment['start']
        x2, y2 = segment['end']

        for px, py in points:
            dist = point_to_line_distance(px, py, x1, y1, x2, y2)
            if dist < max_distance:
                return True

    return False


def filter_windows_on_walls(windows, wall_segments):
    """Keep only windows that lie on wall lines"""
    valid = []
    for w in windows:
        if is_on_wall(w, wall_segments):
            valid.append(w)
    return valid


def create_svg(output_path, image_shape, walls, doors, windows, junctions, scale_factor):
    """Create SVG file with all detected elements"""
    height, width = image_shape[:2]

    orig_width = int(width * scale_factor)
    orig_height = int(height * scale_factor)

    dwg = svgwrite.Drawing(output_path, size=(orig_width, orig_height), profile='tiny')

    # Draw walls
    wall_group = dwg.add(dwg.g(id='walls', stroke='black', stroke_width=int(6*scale_factor), fill='none'))
    for wall in walls:
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        wall_group.add(dwg.line(
            start=(float(x1 * scale_factor), float(y1 * scale_factor)),
            end=(float(x2 * scale_factor), float(y2 * scale_factor)),
            stroke='black',
            stroke_width=int(6 * scale_factor)
        ))

    # Draw doors
    door_group = dwg.add(dwg.g(id='doors'))
    for idx, door in enumerate(doors, 1):
        x, y, w, h = door['bbox']
        door_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill='blue',
            fill_opacity=0.5,
            stroke='darkblue',
            stroke_width=int(3 * scale_factor)
        ))
        text = dwg.text(
            f'D{idx}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='white',
            font_size=int(min(w, h) * 0.5 * scale_factor)
        )
        text['font-weight'] = 'bold'
        door_group.add(text)

    # Draw windows
    window_group = dwg.add(dwg.g(id='windows'))
    for idx, window in enumerate(windows, 1):
        x, y, w, h = window['bbox']
        window_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill='cyan',
            fill_opacity=0.5,
            stroke='blue',
            stroke_width=int(3 * scale_factor)
        ))
        text = dwg.text(
            f'W{idx}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='black',
            font_size=int(min(w, h) * 0.5 * scale_factor)
        )
        text['font-weight'] = 'bold'
        window_group.add(text)

    # Draw junction points
    junction_group = dwg.add(dwg.g(id='junctions'))

    junction_colors = {
        'wall_junction_3way': 'red',
        'wall_junction_4way': 'darkred',
        'wall_corner_90deg': 'orange',
        'wall_endpoint': 'yellow',
        'door_left_corner': 'blue',
        'door_right_corner': 'darkblue',
        'window_left_corner': 'cyan',
        'window_right_corner': 'darkcyan'
    }

    for jtype, points in junctions.items():
        color = junction_colors.get(jtype, 'gray')
        radius = 5 if 'wall' in jtype else 3
        for pt in points:
            junction_group.add(dwg.circle(
                center=(float(pt['x'] * scale_factor), float(pt['y'] * scale_factor)),
                r=int(radius * scale_factor),
                fill=color
            ))

    dwg.save()
    return orig_width, orig_height


def main():
    print("="*80)
    print("CUBICASA FLOOR PLAN VECTORIZATION (JUNCTION-AWARE)")
    print("="*80)

    # Configuration
    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    output_svg = 'floor_plan_with_junctions.svg'
    max_size = 2048
    threshold_doors = 0.3
    threshold_windows = 0.2
    use_junction_graph = True  # NEW: Enable junction-aware extraction

    # Load original
    img_orig = Image.open(image_path)
    orig_width, orig_height = img_orig.size
    print(f"\nOriginal image: {orig_width}x{orig_height}")

    # Load model
    print("\n[1/6] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Preprocess
    print("\n[2/6] Preprocessing...")
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    processed_width, processed_height = img.size
    scale_to_original = orig_width / processed_width
    print(f"   Processed: {processed_width}x{processed_height}")
    print(f"   Scale factor: {scale_to_original:.2f}")

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    img_np = np.array(img)

    # Normalize
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    print("\n[3/6] Running multi-scale DL inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Extract icons
    icons_logits = prediction[0, 33:44]
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    door_prob = icons_pred[2]
    window_prob = icons_pred[1]

    door_mask = door_prob > threshold_doors
    window_mask = window_prob > threshold_windows

    raw_door_regions = ndimage.label(door_mask)[1]
    raw_window_regions = ndimage.label(window_mask)[1]
    print(f"   Raw regions: {raw_door_regions} doors, {raw_window_regions} windows")

    _, doors = refine_dl_detections(door_mask, min_size=50, max_size=6000)
    _, windows = refine_dl_detections(window_mask, min_size=50, max_size=6000)

    # Extract junctions FIRST (before walls!)
    print("\n[4/6] Extracting junction points...")
    junctions = extract_junctions(prediction, threshold=threshold_doors)
    total_junctions = sum(len(pts) for pts in junctions.values())
    print(f"   Detected: {total_junctions} junction points")

    # Extract walls using junction information
    print("\n[5/6] Detecting walls (junction-aware)...")

    # Walls from hatching
    wall_mask_hatching = detect_hatching(img_np, kernel_size=25, density_threshold=0.2)
    wall_segments_hatching = extract_wall_segments_junction_aware(
        wall_mask_hatching, junctions, use_graph=use_junction_graph
    )

    # Walls from DL room segmentation
    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()
    wall_mask_dl = (rooms_pred == 2).astype(np.uint8) * 255
    wall_segments_dl = extract_wall_segments_junction_aware(
        wall_mask_dl, junctions, use_graph=use_junction_graph
    )

    # Combine both wall sources
    wall_segments = wall_segments_hatching + wall_segments_dl
    print(f"   Total: {len(wall_segments)} wall segments ({len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL)")

    # Remove overlapping detections
    windows = remove_nested_boxes(windows, iou_threshold=0.15)
    doors = remove_nested_boxes(doors, iou_threshold=0.5)

    # Filter windows
    windows_before = len(windows)
    windows = filter_windows_on_walls(windows, wall_segments)
    print(f"   After wall alignment filter: {len(windows)}/{windows_before} windows kept")

    print(f"   Final: {len(doors)} doors, {len(windows)} windows")

    # Create SVG
    print("\n[6/6] Creating SVG...")
    svg_w, svg_h = create_svg(
        output_svg,
        img_np.shape,
        wall_segments,
        doors,
        windows,
        junctions,
        scale_to_original
    )

    # Save metadata
    metadata = {
        'source_image': image_path,
        'output_svg': output_svg,
        'original_size': f'{orig_width}x{orig_height}',
        'processed_size': f'{processed_width}x{processed_height}',
        'svg_size': f'{svg_w}x{svg_h}',
        'scale_factor': float(scale_to_original),
        'junction_aware': use_junction_graph,
        'statistics': {
            'walls': len(wall_segments),
            'doors': len(doors),
            'windows': len(windows),
            'junctions': total_junctions
        }
    }

    with open('floor_plan_junctions.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("VECTORIZATION COMPLETE (JUNCTION-AWARE)")
    print("="*80)
    print(f"\nResults:")
    print(f"  Walls:     {len(wall_segments)} segments")
    print(f"  Doors:     {len(doors)}")
    print(f"  Windows:   {len(windows)}")
    print(f"  Junctions: {total_junctions} points")
    print(f"\nOutput:")
    print(f"  SVG:  {output_svg} ({svg_w}x{svg_h})")
    print(f"  JSON: floor_plan_junctions.json")
    print(f"\nMode: {'Junction-aware graph extraction' if use_junction_graph else 'Traditional contour extraction'}")
    print(f"Total elements: {len(wall_segments) + len(doors) + len(windows) + total_junctions}")


if __name__ == '__main__':
    main()
