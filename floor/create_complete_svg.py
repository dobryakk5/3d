"""
Complete floorplan vectorization to SVG
Combines: DL walls/doors/windows + heatmap junctions + hatching detection
"""
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import maximum_filter
import svgwrite
import json

from floortrans.loaders import FloorplanSVG
from floortrans.models import get_model


def load_model(weight_path, model_name='hg_furukawa_original', num_classes=[21, 12, 11]):
    """Load pre-trained model"""
    model = get_model(model_name, 51)
    n_classes = sum(num_classes)
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def detect_hatching(image, kernel_size=15, density_threshold=0.3):
    """Detect hatching patterns (diagonal lines) to identify walls"""
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

    # Detect lines at multiple angles
    angles = [45, -45, 135, -135]
    all_lines = np.zeros_like(binary, dtype=np.float32)

    for angle in angles:
        kernel = create_line_kernel(kernel_size, angle)
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        all_lines += detected.astype(np.float32)

    # Calculate hatching density
    kernel_avg = np.ones((20, 20), dtype=np.float32) / 400
    density = cv2.filter2D(all_lines, -1, kernel_avg)
    density_normalized = density / density.max() if density.max() > 0 else density

    # Threshold to get wall mask
    wall_mask = (density_normalized > density_threshold).astype(np.uint8) * 255

    # Clean up
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return wall_mask


def find_junction_points(heatmap, threshold=0.3, min_distance=5):
    """Find junction points from heatmap using local maxima"""
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
    """Extract all junction points from 21 heatmap channels"""
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


def refine_dl_detections(mask, min_size=30, max_size=3000):
    """Clean up DL predictions"""
    labeled, num = ndimage.label(mask)
    refined = np.zeros_like(mask)

    detections = []
    for i in range(1, num + 1):
        region = labeled == i
        size = region.sum()
        if min_size <= size <= max_size:
            refined[region] = True
            y_coords, x_coords = np.where(region)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            detections.append({
                'bbox': (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                'center': (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            })

    return refined, detections


def extract_wall_segments(wall_mask, min_length=20):
    """Extract wall segments as line segments"""
    # Skeletonize walls
    skeleton = cv2.ximgproc.thinning(wall_mask)

    # Find contours
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        if len(contour) >= 2:
            contour = contour.squeeze()
            if len(contour.shape) == 1:
                continue
            if len(contour) < 2:
                continue

            # Approximate contour to line segments
            epsilon = 2.0
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


def create_svg(image_shape, walls, doors, windows, junctions, output_path):
    """Create SVG file with all detected elements"""
    height, width = image_shape[:2]

    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')

    # Add original image as background (optional)
    # dwg.add(dwg.image(href='plan_floor1.jpg', insert=(0, 0), size=(width, height), opacity=0.3))

    # Draw walls
    wall_group = dwg.add(dwg.g(id='walls', stroke='black', stroke_width=3, fill='none'))
    for wall in walls:
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        wall_group.add(dwg.line(
            start=(float(x1), float(y1)),
            end=(float(x2), float(y2)),
            stroke='black',
            stroke_width=4
        ))

    # Draw doors
    door_group = dwg.add(dwg.g(id='doors', fill='blue', opacity=0.6))
    for door in doors:
        x, y, w, h = door['bbox']
        door_group.add(dwg.rect(
            insert=(x, y),
            size=(w, h),
            fill='blue',
            opacity=0.6,
            stroke='darkblue',
            stroke_width=2
        ))
        # Add arc for door swing
        cx, cy = door['center']
        door_group.add(dwg.circle(
            center=(cx, cy),
            r=3,
            fill='darkblue'
        ))

    # Draw windows
    window_group = dwg.add(dwg.g(id='windows', fill='lightblue', opacity=0.6))
    for window in windows:
        x, y, w, h = window['bbox']
        window_group.add(dwg.rect(
            insert=(x, y),
            size=(w, h),
            fill='lightblue',
            opacity=0.6,
            stroke='blue',
            stroke_width=2
        ))

    # Draw junctions/corners
    junction_group = dwg.add(dwg.g(id='junctions'))

    # Wall junctions (red)
    if 'wall_junction_3way' in junctions:
        for pt in junctions['wall_junction_3way']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=5, fill='red'))

    if 'wall_junction_4way' in junctions:
        for pt in junctions['wall_junction_4way']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=5, fill='darkred'))

    if 'wall_corner_90deg' in junctions:
        for pt in junctions['wall_corner_90deg']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=4, fill='orange'))

    if 'wall_endpoint' in junctions:
        for pt in junctions['wall_endpoint']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=3, fill='yellow'))

    # Door corners (blue)
    if 'door_left_corner' in junctions:
        for pt in junctions['door_left_corner']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=3, fill='blue'))

    if 'door_right_corner' in junctions:
        for pt in junctions['door_right_corner']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=3, fill='darkblue'))

    # Window corners (cyan)
    if 'window_left_corner' in junctions:
        for pt in junctions['window_left_corner']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=3, fill='cyan'))

    if 'window_right_corner' in junctions:
        for pt in junctions['window_right_corner']:
            junction_group.add(dwg.circle(center=(pt['x'], pt['y']), r=3, fill='darkcyan'))

    dwg.save()
    print(f"SVG saved to {output_path}")


def main():
    # Configuration
    image_path = 'plan_floor1.jpg'
    weight_path = 'model_best_val_loss_var.pkl'
    output_svg = 'floor_plan.svg'
    max_size = 2048  # Increased for better detection
    threshold = 0.3

    print("Loading image...")
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # Resize if too large
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"Resized from {w}x{h} to {new_w}x{new_h}")
        w, h = new_w, new_h

    img_array = np.array(img)

    # Load model and run inference
    print("Loading model...")
    model = load_model(weight_path)

    print("Running inference...")
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        prediction = model(img_tensor)

    # Extract predictions
    print("Extracting predictions...")
    heatmaps = prediction[0, :21]
    rooms_logits = prediction[0, 21:33]
    icons_logits = prediction[0, 33:44]

    # Get room and icon predictions
    rooms_pred = torch.softmax(rooms_logits, 0)
    icons_pred = torch.softmax(icons_logits, 0)

    # Extract walls from room segmentation (class 2 = wall)
    wall_prob = rooms_pred[2].cpu().numpy()
    dl_wall_mask = (wall_prob > threshold).astype(np.uint8) * 255

    # Extract doors (icon class 4 = door)
    door_prob = icons_pred[4].cpu().numpy()
    door_mask = door_prob > threshold
    _, doors = refine_dl_detections(door_mask, min_size=30, max_size=3000)

    # Extract windows (icon class 5 = window)
    window_prob = icons_pred[5].cpu().numpy()
    window_mask = window_prob > threshold
    _, windows = refine_dl_detections(window_mask, min_size=30, max_size=3000)

    # Detect hatching for wall refinement
    print("Detecting hatching patterns...")
    hatching_mask = detect_hatching(img_array, kernel_size=15, density_threshold=0.25)

    # Combine DL walls and hatching walls
    combined_walls = cv2.bitwise_or(dl_wall_mask, hatching_mask)
    combined_walls = cv2.morphologyEx(combined_walls, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Extract wall segments
    print("Extracting wall segments...")
    wall_segments = extract_wall_segments(combined_walls, min_length=20)

    # Extract junctions from heatmaps
    print("Extracting junction points...")
    junctions = extract_junctions(prediction, threshold=0.3)

    # Create SVG
    print("Creating SVG...")
    create_svg(img_array.shape, wall_segments, doors, windows, junctions, output_svg)

    # Save summary
    summary = {
        'image': image_path,
        'size': f'{w}x{h}',
        'statistics': {
            'walls': len(wall_segments),
            'doors': len(doors),
            'windows': len(windows),
            'junctions': sum(len(pts) for pts in junctions.values())
        },
        'wall_segments': wall_segments,
        'doors': doors,
        'windows': windows,
        'junctions': junctions
    }

    with open('floor_plan_data.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("COMPLETE VECTORIZATION RESULTS")
    print("="*80)
    print(f"Walls:     {len(wall_segments)} segments")
    print(f"Doors:     {len(doors)} detected")
    print(f"Windows:   {len(windows)} detected")
    print(f"Junctions: {sum(len(pts) for pts in junctions.values())} points")
    print(f"\nSVG saved to: {output_svg}")
    print(f"Data saved to: floor_plan_data.json")

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(combined_walls, cmap='gray')
    axes[0, 1].set_title(f'Combined Walls ({len(wall_segments)} segments)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_array)
    for door in doors:
        x, y, w, h = door['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=2)
        axes[0, 2].add_patch(rect)
    axes[0, 2].set_title(f'Doors ({len(doors)} detected)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(img_array)
    for window in windows:
        x, y, w, h = window['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='cyan', linewidth=2)
        axes[1, 0].add_patch(rect)
    axes[1, 0].set_title(f'Windows ({len(windows)} detected)')
    axes[1, 0].axis('off')

    # Junctions visualization
    axes[1, 1].imshow(img_array)
    colors = {
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
        color = colors.get(jtype, 'gray')
        for pt in points:
            axes[1, 1].plot(pt['x'], pt['y'], 'o', color=color, markersize=5)
    axes[1, 1].set_title(f'Junctions ({sum(len(pts) for pts in junctions.values())} points)')
    axes[1, 1].axis('off')

    # Complete overlay
    axes[1, 2].imshow(img_array, alpha=0.7)
    for seg in wall_segments:
        x_coords = [seg['start'][0], seg['end'][0]]
        y_coords = [seg['start'][1], seg['end'][1]]
        axes[1, 2].plot(x_coords, y_coords, 'k-', linewidth=2, alpha=0.8)
    for door in doors:
        x, y, w, h = door['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor='blue', alpha=0.3, edgecolor='blue', linewidth=2)
        axes[1, 2].add_patch(rect)
    for window in windows:
        x, y, w, h = window['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor='cyan', alpha=0.3, edgecolor='cyan', linewidth=2)
        axes[1, 2].add_patch(rect)
    axes[1, 2].set_title('Complete Vectorization')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('complete_vectorization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Visualization saved to: complete_vectorization.png")


if __name__ == '__main__':
    main()
