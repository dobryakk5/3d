#!/usr/bin/env python
"""
Detection using ONLY Deep Learning (without CV noise)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy import ndimage
import json
import csv

from floortrans.models import get_model

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

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

def refine_dl_detections(mask, min_size=30, max_size=3000):
    """Clean up DL predictions by removing noise"""
    labeled, num = ndimage.label(mask)
    refined = np.zeros_like(mask)

    for i in range(1, num + 1):
        region = labeled == i
        size = region.sum()

        if min_size <= size <= max_size:
            refined[region] = True

    return refined

def extract_detection_info(mask, element_type, scale_factor=1.0):
    """Extract bounding boxes and metadata"""
    labeled, num_regions = ndimage.label(mask)
    detections = []

    for i in range(1, num_regions + 1):
        region = labeled == i
        rows, cols = np.where(region)

        if len(rows) == 0:
            continue

        x_min, x_max = int(cols.min()), int(cols.max())
        y_min, y_max = int(rows.min()), int(rows.max())
        center_y, center_x = ndimage.center_of_mass(region)
        area = region.sum()

        detection = {
            'id': i,
            'type': element_type,
            'bbox': {
                'x_min': int(x_min * scale_factor),
                'y_min': int(y_min * scale_factor),
                'x_max': int(x_max * scale_factor),
                'y_max': int(y_max * scale_factor),
                'width': int((x_max - x_min) * scale_factor),
                'height': int((y_max - y_min) * scale_factor)
            },
            'center': {
                'x': int(center_x * scale_factor),
                'y': int(center_y * scale_factor)
            },
            'area_pixels': int(area),
            'area_scaled': int(area * scale_factor * scale_factor),
            'confidence': 'high'  # All from DL model
        }

        detections.append(detection)

    return detections

def main():
    print("=" * 80)
    print("DEEP LEARNING ONLY DETECTION (Clean Results)")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    max_size = 2048
    n_classes = 44

    # Load original for scale
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

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    img_np = np.array(img)
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    print(f"\n[3/5] Running Deep Learning inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Get room and icon predictions
    rooms_logits = prediction[0, 21:33]
    icons_logits = prediction[0, 33:44]

    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    # Extract door and window predictions
    door_prob = icons_pred[2]  # Door channel
    window_prob = icons_pred[1]  # Window channel

    # Test different thresholds
    print(f"\n   Testing thresholds...")
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        doors = door_prob > threshold
        windows = window_prob > threshold
        door_regions = ndimage.label(doors)[1]
        window_regions = ndimage.label(windows)[1]
        print(f"   Threshold {threshold}: {door_regions} doors, {window_regions} windows")

    # Use optimal threshold
    threshold = 0.3
    dl_doors = door_prob > threshold
    dl_windows = window_prob > threshold

    print(f"\n   Selected threshold: {threshold}")
    print(f"   Raw: {dl_doors.sum()} door pixels, {dl_windows.sum()} window pixels")

    print(f"\n[4/5] Refining detections (removing noise)...")
    doors_final = refine_dl_detections(dl_doors, min_size=30, max_size=3000)
    windows_final = refine_dl_detections(dl_windows, min_size=30, max_size=3000)

    door_labels, num_doors = ndimage.label(doors_final)
    window_labels, num_windows = ndimage.label(windows_final)

    print(f"   Refined: {doors_final.sum()} door pixels ({num_doors} regions)")
    print(f"   Refined: {windows_final.sum()} window pixels ({num_windows} regions)")

    # Extract info
    door_detections = extract_detection_info(doors_final, 'door', scale_to_original)
    window_detections = extract_detection_info(windows_final, 'window', scale_to_original)

    print(f"\n[5/5] Creating visualization and exports...")

    # Visualization
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15)

    img_show = (img_normalized + 1) / 2

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_show)
    ax1.set_title('Original Floorplan', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(rooms_pred, cmap='tab20', vmin=0, vmax=11)
    ax2.set_title('Room Segmentation', fontsize=14, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=np.arange(12), fraction=0.046)
    cbar2.ax.set_yticklabels(room_classes, fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    icons_display = np.argmax(icons_pred, axis=0)
    im3 = ax3.imshow(icons_display, cmap='tab20', vmin=0, vmax=10)
    ax3.set_title('All Icons (Raw DL Output)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=np.arange(11), fraction=0.046)
    cbar3.ax.set_yticklabels(icon_classes, fontsize=9)

    # Row 2: Probability maps and final detection
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img_show)
    im4 = ax4.imshow(door_prob, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    ax4.set_title(f'Door Probability Map\n(threshold={threshold})', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img_show)
    im5 = ax5.imshow(window_prob, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    ax5.set_title(f'Window Probability Map\n(threshold={threshold})', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Final labeled result
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(img_show)

    overlay = np.zeros((*doors_final.shape, 4))

    cmap_doors = plt.cm.Reds
    cmap_windows = plt.cm.Blues

    for i in range(1, num_doors + 1):
        mask = door_labels == i
        color = cmap_doors(0.6 + 0.2 * (i % 3) / 3)
        overlay[mask] = color

        y, x = ndimage.center_of_mass(mask)
        if not np.isnan(x) and not np.isnan(y):
            ax6.text(x, y, f'D{i}', fontsize=9, fontweight='bold',
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))

    for i in range(1, num_windows + 1):
        mask = window_labels == i
        color = cmap_windows(0.6 + 0.2 * (i % 3) / 3)
        overlay[mask] = color

        y, x = ndimage.center_of_mass(mask)
        if not np.isnan(x) and not np.isnan(y):
            ax6.text(x, y, f'W{i}', fontsize=9, fontweight='bold',
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.8))

    ax6.imshow(overlay)
    ax6.set_title(f'FINAL: {num_doors} Doors + {num_windows} Windows\n(Deep Learning Only)',
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax6.axis('off')

    output_path = 'plan_floor1_DL_ONLY.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close('all')

    # Export data
    results = {
        'image': {
            'filename': image_path,
            'original_size': {'width': orig_width, 'height': orig_height},
            'processed_size': {'width': processed_width, 'height': processed_height},
            'scale_factor': float(scale_to_original)
        },
        'detection_method': 'Deep Learning Only (CubiCasa5K model)',
        'threshold': threshold,
        'summary': {
            'total_doors': len(door_detections),
            'total_windows': len(window_detections),
            'total_elements': len(door_detections) + len(window_detections)
        },
        'detections': {
            'doors': door_detections,
            'windows': window_detections
        }
    }

    # JSON
    json_path = 'detections_DL_only.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {json_path}")

    # CSV
    csv_path = 'detections_DL_only.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Type', 'Center_X', 'Center_Y', 'BBox_X_Min', 'BBox_Y_Min',
                        'BBox_X_Max', 'BBox_Y_Max', 'Width', 'Height', 'Area_Pixels'])

        for det in door_detections:
            writer.writerow([
                f"D{det['id']}", det['type'],
                det['center']['x'], det['center']['y'],
                det['bbox']['x_min'], det['bbox']['y_min'],
                det['bbox']['x_max'], det['bbox']['y_max'],
                det['bbox']['width'], det['bbox']['height'],
                det['area_scaled']
            ])

        for det in window_detections:
            writer.writerow([
                f"W{det['id']}", det['type'],
                det['center']['x'], det['center']['y'],
                det['bbox']['x_min'], det['bbox']['y_min'],
                det['bbox']['x_max'], det['bbox']['y_max'],
                det['bbox']['width'], det['bbox']['height'],
                det['area_scaled']
            ])

    print(f"   ✓ Saved: {csv_path}")

    # TXT
    txt_path = 'detections_DL_only.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FLOORPLAN DETECTION RESULTS (Deep Learning Only)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Image: {image_path}\n")
        f.write(f"Original size: {orig_width} x {orig_height} pixels\n")
        f.write(f"Detection method: Deep Learning (CubiCasa5K model)\n")
        f.write(f"Threshold: {threshold}\n\n")

        f.write(f"SUMMARY:\n")
        f.write(f"  Total doors:   {len(door_detections)}\n")
        f.write(f"  Total windows: {len(window_detections)}\n")
        f.write(f"  Total elements: {len(door_detections) + len(window_detections)}\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"DOORS ({len(door_detections)} detected):\n")
        f.write("=" * 80 + "\n\n")

        for det in door_detections:
            f.write(f"Door D{det['id']}:\n")
            f.write(f"  Position: ({det['center']['x']}, {det['center']['y']})\n")
            f.write(f"  Size: {det['bbox']['width']} x {det['bbox']['height']} pixels\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"WINDOWS ({len(window_detections)} detected):\n")
        f.write("=" * 80 + "\n\n")

        for det in window_detections:
            f.write(f"Window W{det['id']}:\n")
            f.write(f"  Position: ({det['center']['x']}, {det['center']['y']})\n")
            f.write(f"  Size: {det['bbox']['width']} x {det['bbox']['height']} pixels\n\n")

    print(f"   ✓ Saved: {txt_path}")

    print(f"\n{'=' * 80}")
    print(f"{'CLEAN DETECTION SUMMARY (DL ONLY)':^80}")
    print(f"{'=' * 80}")
    print(f"\n  🚪 Doors:   {len(door_detections)}")
    print(f"  🪟 Windows: {len(window_detections)}")
    print(f"  📦 Total:   {len(door_detections) + len(window_detections)}")
    print(f"\n{'=' * 80}")
    print(f"\n✅ Results saved to:")
    print(f"   - {output_path}")
    print(f"   - {json_path}")
    print(f"   - {csv_path}")
    print(f"   - {txt_path}")
    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
