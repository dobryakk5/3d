#!/usr/bin/env python
"""
Wall detection and extraction from floorplan
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy import ndimage
import cv2
import json

from floortrans.models import get_model

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]

def multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1]):
    """Run inference at multiple scales"""
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

def extract_walls_opencv(image_np):
    """Extract walls using traditional CV"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Binary threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to get thick lines (walls)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rect)

    # Find thick edges (walls are typically thicker than regular lines)
    kernel = np.ones((2, 2), np.uint8)
    walls = cv2.dilate(binary, kernel, iterations=1)
    walls = cv2.erode(walls, kernel, iterations=1)

    return walls > 0

def extract_wall_segments(wall_mask, scale_factor=1.0):
    """Extract individual wall segments and their properties"""
    # Label connected components
    labeled, num_walls = ndimage.label(wall_mask)

    wall_segments = []

    for i in range(1, num_walls + 1):
        segment = labeled == i
        area = segment.sum()

        # Filter out very small segments (noise)
        if area < 50:
            continue

        # Get bounding box
        rows, cols = np.where(segment)
        if len(rows) == 0:
            continue

        x_min, x_max = int(cols.min()), int(cols.max())
        y_min, y_max = int(rows.min()), int(rows.max())

        width = x_max - x_min
        height = y_max - y_min

        # Determine if horizontal or vertical
        if width > height * 1.5:
            orientation = 'horizontal'
            length = width
            thickness = height
        elif height > width * 1.5:
            orientation = 'vertical'
            length = height
            thickness = width
        else:
            orientation = 'diagonal'
            length = max(width, height)
            thickness = min(width, height)

        # Get center
        center_y, center_x = ndimage.center_of_mass(segment)

        wall_info = {
            'id': i,
            'type': 'wall',
            'orientation': orientation,
            'bbox': {
                'x_min': int(x_min * scale_factor),
                'y_min': int(y_min * scale_factor),
                'x_max': int(x_max * scale_factor),
                'y_max': int(y_max * scale_factor),
                'width': int(width * scale_factor),
                'height': int(height * scale_factor)
            },
            'center': {
                'x': int(center_x * scale_factor),
                'y': int(center_y * scale_factor)
            },
            'dimensions': {
                'length': int(length * scale_factor),
                'thickness': int(thickness * scale_factor),
                'area': int(area * scale_factor * scale_factor)
            }
        }

        wall_segments.append(wall_info)

    return wall_segments

def extract_wall_contours(wall_mask, scale_factor=1.0):
    """Extract wall contours as polygons"""
    # Convert to uint8 for OpenCV
    wall_uint8 = (wall_mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(wall_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []

    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue

        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Scale coordinates
        scaled_points = []
        for point in approx:
            x, y = point[0]
            scaled_points.append([
                int(x * scale_factor),
                int(y * scale_factor)
            ])

        if len(scaled_points) >= 3:
            polygons.append({
                'id': i + 1,
                'points': scaled_points,
                'num_vertices': len(scaled_points)
            })

    return polygons

def calculate_wall_statistics(wall_mask, scale_factor=1.0):
    """Calculate overall wall statistics"""
    total_wall_pixels = wall_mask.sum()
    total_wall_area = int(total_wall_pixels * scale_factor * scale_factor)

    # Estimate total wall length (using skeleton)
    from skimage.morphology import skeletonize
    skeleton = skeletonize(wall_mask)
    wall_length_pixels = skeleton.sum()
    wall_length = int(wall_length_pixels * scale_factor)

    # Average thickness
    avg_thickness = total_wall_pixels / wall_length_pixels if wall_length_pixels > 0 else 0
    avg_thickness_scaled = int(avg_thickness * scale_factor)

    return {
        'total_area_pixels': total_wall_area,
        'total_length_pixels': wall_length,
        'average_thickness_pixels': avg_thickness_scaled,
        'coverage_percentage': float(total_wall_pixels) / wall_mask.size * 100
    }

def main():
    print("=" * 80)
    print("WALL DETECTION AND ANALYSIS")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    max_size = 2048
    n_classes = 44

    # Load original
    img_orig = Image.open(image_path)
    orig_width, orig_height = img_orig.size

    print(f"\n[1/5] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("   ✓ Model loaded")

    print(f"\n[2/5] Preprocessing image...")
    img = Image.open(image_path).convert('RGB')

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    processed_width, processed_height = img.size
    scale_to_original = orig_width / processed_width

    print(f"   Original: {orig_width}x{orig_height}")
    print(f"   Processed: {processed_width}x{processed_height}")
    print(f"   Scale factor: {scale_to_original:.2f}")

    img_np = np.array(img)
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    print(f"\n[3/5] Running Deep Learning inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Get room segmentation (Wall class is index 2)
    rooms_logits = prediction[0, 21:33]
    rooms_pred = F.softmax(rooms_logits, 0).cpu().data.numpy()

    # Extract wall class probability
    wall_prob = rooms_pred[2]  # Wall is class 2

    # Get predicted classes
    rooms_class = np.argmax(rooms_pred, axis=0)

    # Wall mask from DL
    dl_walls = (rooms_class == 2)

    print(f"   DL wall pixels: {dl_walls.sum()}")
    print(f"   DL wall probability max: {wall_prob.max():.3f}")

    print(f"\n[4/5] Running OpenCV wall extraction...")
    cv_walls = extract_walls_opencv(img_np)

    print(f"   CV wall pixels: {cv_walls.sum()}")

    # Combine methods (use DL as primary, CV for refinement)
    walls_combined = np.logical_or(dl_walls, cv_walls)

    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    walls_combined_uint8 = (walls_combined * 255).astype(np.uint8)
    walls_combined_uint8 = cv2.morphologyEx(walls_combined_uint8, cv2.MORPH_CLOSE, kernel)
    walls_final = walls_combined_uint8 > 0

    print(f"   Combined wall pixels: {walls_final.sum()}")

    print(f"\n[5/5] Analyzing wall structure...")

    # Extract wall segments
    wall_segments = extract_wall_segments(walls_final, scale_to_original)
    print(f"   Found {len(wall_segments)} wall segments")

    # Extract contours
    wall_polygons = extract_wall_contours(walls_final, scale_to_original)
    print(f"   Extracted {len(wall_polygons)} wall contours")

    # Calculate statistics
    wall_stats = calculate_wall_statistics(walls_final, scale_to_original)
    print(f"   Total wall area: {wall_stats['total_area_pixels']} pixels²")
    print(f"   Estimated wall length: {wall_stats['total_length_pixels']} pixels")
    print(f"   Average wall thickness: {wall_stats['average_thickness_pixels']} pixels")
    print(f"   Wall coverage: {wall_stats['coverage_percentage']:.2f}%")

    print(f"\n[6/6] Creating visualization...")

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

    img_show = (img_normalized + 1) / 2

    # Row 1: Input and segmentation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_show)
    ax1.set_title('Original Floorplan', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(rooms_class, cmap='tab20', vmin=0, vmax=11)
    ax2.set_title('Room Segmentation\n(Wall = Orange)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=np.arange(12), fraction=0.046)
    cbar2.ax.set_yticklabels(room_classes, fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_show)
    im3 = ax3.imshow(wall_prob, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    ax3.set_title('Wall Probability Map (DL)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: Different methods
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img_show)
    ax4.imshow(dl_walls, cmap='Reds', alpha=0.6)
    ax4.set_title(f'Deep Learning Walls\n({dl_walls.sum()} pixels)', fontsize=14, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img_show)
    ax5.imshow(cv_walls, cmap='Blues', alpha=0.6)
    ax5.set_title(f'OpenCV Walls\n({cv_walls.sum()} pixels)', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(img_show)
    ax6.imshow(walls_final, cmap='Greens', alpha=0.7)
    ax6.set_title(f'Combined Walls\n({walls_final.sum()} pixels)',
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax6.axis('off')

    # Row 3: Wall analysis
    ax7 = fig.add_subplot(gs[2, 0])
    # Show only walls (white on black)
    ax7.imshow(walls_final, cmap='gray')
    ax7.set_title('Wall Mask (Binary)', fontsize=14, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[2, 1])
    # Colorize wall segments by orientation
    segment_viz = np.zeros((*walls_final.shape, 3))
    for seg in wall_segments:
        x_min = int(seg['bbox']['x_min'] / scale_to_original)
        y_min = int(seg['bbox']['y_min'] / scale_to_original)
        x_max = int(seg['bbox']['x_max'] / scale_to_original)
        y_max = int(seg['bbox']['y_max'] / scale_to_original)

        mask = walls_final[y_min:y_max, x_min:x_max]

        if seg['orientation'] == 'horizontal':
            color = [1, 0, 0]  # Red
        elif seg['orientation'] == 'vertical':
            color = [0, 0, 1]  # Blue
        else:
            color = [0, 1, 0]  # Green

        for c in range(3):
            segment_viz[y_min:y_max, x_min:x_max, c] = np.where(
                mask, color[c], segment_viz[y_min:y_max, x_min:x_max, c]
            )

    ax8.imshow(segment_viz)
    ax8.set_title(f'Wall Segments by Orientation\nRed=H, Blue=V, Green=D ({len(wall_segments)} segments)',
                  fontsize=12, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 2])
    # Draw wall contours
    ax9.imshow(img_show)
    contour_img = np.zeros((*walls_final.shape, 4))

    wall_uint8 = (walls_final * 255).astype(np.uint8)
    contours, _ = cv2.findContours(wall_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(contour_img, [contour], -1, (1, 0, 0, 0.8), 2)

    ax9.imshow(contour_img)
    ax9.set_title(f'Wall Contours\n({len(wall_polygons)} polygons)', fontsize=14, fontweight='bold')
    ax9.axis('off')

    output_path = 'plan_floor1_WALLS.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close('all')

    # Export data
    print(f"\n[7/7] Exporting wall data...")

    results = {
        'image': {
            'filename': image_path,
            'original_size': {'width': orig_width, 'height': orig_height},
            'processed_size': {'width': processed_width, 'height': processed_height},
            'scale_factor': float(scale_to_original)
        },
        'statistics': wall_stats,
        'summary': {
            'total_segments': len(wall_segments),
            'horizontal_walls': sum(1 for s in wall_segments if s['orientation'] == 'horizontal'),
            'vertical_walls': sum(1 for s in wall_segments if s['orientation'] == 'vertical'),
            'diagonal_walls': sum(1 for s in wall_segments if s['orientation'] == 'diagonal'),
            'total_contours': len(wall_polygons)
        },
        'wall_segments': wall_segments,
        'wall_polygons': wall_polygons
    }

    # JSON
    json_path = 'walls_detection.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {json_path}")

    # TXT report
    txt_path = 'walls_detection.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WALL DETECTION AND ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Image: {image_path}\n")
        f.write(f"Original size: {orig_width} x {orig_height} pixels\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total wall area: {wall_stats['total_area_pixels']:,} pixels²\n")
        f.write(f"  Estimated total length: {wall_stats['total_length_pixels']:,} pixels\n")
        f.write(f"  Average thickness: {wall_stats['average_thickness_pixels']} pixels\n")
        f.write(f"  Wall coverage: {wall_stats['coverage_percentage']:.2f}%\n\n")

        f.write("WALL SEGMENTS:\n")
        f.write(f"  Total segments: {len(wall_segments)}\n")
        f.write(f"  Horizontal: {sum(1 for s in wall_segments if s['orientation'] == 'horizontal')}\n")
        f.write(f"  Vertical: {sum(1 for s in wall_segments if s['orientation'] == 'vertical')}\n")
        f.write(f"  Diagonal: {sum(1 for s in wall_segments if s['orientation'] == 'diagonal')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DETAILED SEGMENT LIST:\n")
        f.write("=" * 80 + "\n\n")

        for seg in sorted(wall_segments, key=lambda x: x['dimensions']['area'], reverse=True)[:20]:
            f.write(f"Segment #{seg['id']} ({seg['orientation'].upper()}):\n")
            f.write(f"  Position: ({seg['center']['x']}, {seg['center']['y']})\n")
            f.write(f"  Length: {seg['dimensions']['length']} pixels\n")
            f.write(f"  Thickness: {seg['dimensions']['thickness']} pixels\n")
            f.write(f"  Area: {seg['dimensions']['area']:,} pixels²\n\n")

    print(f"   ✓ Saved: {txt_path}")

    print(f"\n{'=' * 80}")
    print(f"{'WALL DETECTION SUMMARY':^80}")
    print(f"{'=' * 80}")
    print(f"\n  📏 Total wall segments: {len(wall_segments)}")
    print(f"     - Horizontal: {sum(1 for s in wall_segments if s['orientation'] == 'horizontal')}")
    print(f"     - Vertical: {sum(1 for s in wall_segments if s['orientation'] == 'vertical')}")
    print(f"     - Diagonal: {sum(1 for s in wall_segments if s['orientation'] == 'diagonal')}")
    print(f"\n  📐 Total area: {wall_stats['total_area_pixels']:,} pixels²")
    print(f"  📏 Total length: {wall_stats['total_length_pixels']:,} pixels")
    print(f"  📊 Average thickness: {wall_stats['average_thickness_pixels']} pixels")
    print(f"  🎯 Coverage: {wall_stats['coverage_percentage']:.2f}%")
    print(f"\n{'=' * 80}")
    print(f"\n✅ Results saved to:")
    print(f"   - {output_path}")
    print(f"   - {json_path}")
    print(f"   - {txt_path}")
    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
