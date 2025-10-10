#!/usr/bin/env python
"""
Complete CubiCasa vectorization to SVG:
1. Multi-scale DL inference for doors and windows
2. Hatching detection for walls
3. Heatmap junctions for corners
4. Export to SVG
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
    # Apply morphological closing to connect nearby regions (e.g., split windows)
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
    """Remove overlapping boxes using IoU (Intersection over Union)"""
    if len(detections) <= 1:
        return detections

    # Sort by area (largest first)
    sorted_dets = sorted(detections, key=lambda d: d['area'], reverse=True)

    keep = []
    for det in sorted_dets:
        x1, y1, w1, h1 = det['bbox']
        is_overlapping = False

        for kept_det in keep:
            x2, y2, w2, h2 = kept_det['bbox']

            # Calculate IoU (Intersection over Union)
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

                # If high overlap, skip smaller box
                if iou > iou_threshold:
                    is_overlapping = True
                    break

        if not is_overlapping:
            keep.append(det)

    return keep


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def is_on_wall(window, wall_segments, max_distance=30):
    """Check if window is aligned with any wall"""
    x, y, w, h = window['bbox']

    # Window center and edges
    points = [
        (x + w/2, y + h/2),  # center
        (x + w/2, y),        # top center
        (x + w/2, y + h),    # bottom center
        (x, y + h/2),        # left center
        (x + w, y + h/2),    # right center
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
        # Must be on wall
        if not is_on_wall(w, wall_segments):
            continue

        valid.append(w)

    return valid


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


def extract_wall_segments(wall_mask, min_length=50):
    """Extract wall line segments"""
    # Alternative thinning without ximgproc
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
                    if length >= min_length:
                        segments.append({
                            'start': pt1,
                            'end': pt2,
                            'length': float(length)
                        })

    return segments


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


def create_svg(output_path, image_shape, walls, doors, windows, junctions, scale_factor):
    """Create SVG file with all detected elements"""
    height, width = image_shape[:2]

    # Scale back to original resolution
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
        # Label
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
        # Label
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
    print("CUBICASA FLOOR PLAN VECTORIZATION")
    print("="*80)

    # Configuration
    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    output_svg = 'floor_plan.svg'
    max_size = 2048
    threshold_doors = 0.3
    threshold_windows = 0.2  # Lower threshold for windows to catch all 7

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

    door_prob = icons_pred[2]  # Door = class 2
    window_prob = icons_pred[1]  # Window = class 1

    door_mask = door_prob > threshold_doors
    window_mask = window_prob > threshold_windows

    # Count raw regions before filtering
    raw_door_regions = ndimage.label(door_mask)[1]
    raw_window_regions = ndimage.label(window_mask)[1]
    print(f"   Raw regions: {raw_door_regions} doors, {raw_window_regions} windows")

    _, doors = refine_dl_detections(door_mask, min_size=50, max_size=6000)  # Higher max for large windows
    _, windows = refine_dl_detections(window_mask, min_size=50, max_size=6000)  # Higher max for large windows

    # Extract walls from BOTH sources
    print("\n[4/6] Detecting walls...")

    # Walls from hatching
    wall_mask_hatching = detect_hatching(img_np, kernel_size=25, density_threshold=0.2)
    wall_segments_hatching = extract_wall_segments(wall_mask_hatching, min_length=50)

    # Walls from DL room segmentation
    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()
    wall_mask_dl = (rooms_pred == 2).astype(np.uint8) * 255  # Class 2 = wall
    wall_segments_dl = extract_wall_segments(wall_mask_dl, min_length=50)

    # Combine both wall sources
    wall_segments = wall_segments_hatching + wall_segments_dl
    print(f"   Detected: {len(wall_segments)} wall segments ({len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL)")

    # Remove overlapping detections
    windows = remove_nested_boxes(windows, iou_threshold=0.15)
    doors = remove_nested_boxes(doors, iou_threshold=0.5)

    # Filter windows: keep only those on wall lines
    windows_before = len(windows)
    windows = filter_windows_on_walls(windows, wall_segments)
    print(f"   After wall alignment filter: {len(windows)}/{windows_before} windows kept")

    print(f"   Final: {len(doors)} doors, {len(windows)} windows")

    # Extract junctions
    print("\n[5/6] Extracting junction points...")
    junctions = extract_junctions(prediction, threshold=threshold_doors)
    total_junctions = sum(len(pts) for pts in junctions.values())
    print(f"   Detected: {total_junctions} junction points")

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
        'statistics': {
            'walls': len(wall_segments),
            'doors': len(doors),
            'windows': len(windows),
            'junctions': total_junctions
        }
    }

    with open('floor_plan.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save detailed detection data for visualization
    detections_data = {
        'junctions': junctions,
        'windows': windows,
        'doors': doors,
        'walls': wall_segments
    }

    with open('floor_plan_detections.json', 'w') as f:
        json.dump(detections_data, f, indent=2)

    print("\n" + "="*80)
    print("VECTORIZATION COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Walls:     {len(wall_segments)} segments")
    print(f"  Doors:     {len(doors)}")
    print(f"  Windows:   {len(windows)}")
    print(f"  Junctions: {total_junctions} points")
    print(f"\nOutput:")
    print(f"  SVG:  {output_svg} ({svg_w}x{svg_h})")
    print(f"  JSON: floor_plan.json")
    print(f"\nTotal elements: {len(wall_segments) + len(doors) + len(windows) + total_junctions}")


if __name__ == '__main__':
    main()
