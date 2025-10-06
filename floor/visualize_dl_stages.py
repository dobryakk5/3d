#!/usr/bin/env python3
"""
Create DL detection stages visualization similar to plan_floor1_DL_ONLY.png
Shows: Original, Room Segmentation, Icons, Door/Window Probability Maps, Final Result
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

# Import model
import sys
sys.path.append('floortrans')
from floortrans.loaders import FloorplanSVG
from floortrans.models import get_model

def multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1]):
    """Run inference at multiple scales and average results"""
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

    avg_pred = torch.mean(torch.stack(all_preds), dim=0)
    return avg_pred

def refine_dl_detections(mask, min_size=50, max_size=6000):
    """Refine binary mask to get clean regions"""
    # Apply morphological closing to connect nearby regions (e.g., split windows)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    labeled, num_features = ndimage.label(mask_closed)

    refined_mask = np.zeros_like(mask)
    detections = []

    for region_id in range(1, num_features + 1):
        region_mask = (labeled == region_id)
        size = np.sum(region_mask)

        if size < min_size or size > max_size:
            continue

        refined_mask[region_mask] = 255

        coords = np.argwhere(region_mask)
        if len(coords) == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        detections.append({
            'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
            'area': int(size)
        })

    return refined_mask, detections

def main():
    print("="*80)
    print("DL DETECTION STAGES VISUALIZATION")
    print("="*80)

    image_path = 'plan_floor1.jpg'

    # Load and preprocess
    print("\n[1/5] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("\n[2/5] Preprocessing image...")
    img_orig = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img_orig.size

    # Resize to max 2048
    max_size = 2048
    w, h = orig_width, orig_height
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        img = img_orig

    processed_width, processed_height = img.size

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    img_np = np.array(img)

    # For visualization
    img_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Normalize for model
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    print("\n[3/5] Running DL inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Extract room segmentation
    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()

    # Extract icons
    icons_logits = prediction[0, 33:44]
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    door_prob = icons_pred[2]
    window_prob = icons_pred[1]

    print("\n[4/5] Processing detections...")
    threshold = 0.3

    door_mask = (door_prob > threshold).astype(np.uint8)
    window_mask = (window_prob > threshold).astype(np.uint8)

    _, doors = refine_dl_detections(door_mask, min_size=50, max_size=6000)
    _, windows = refine_dl_detections(window_mask, min_size=50, max_size=6000)

    print(f"   Detected: {len(doors)} doors, {len(windows)} windows")

    print("\n[5/5] Creating visualization...")

    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.1)

    # Room segmentation colormap
    room_colors = np.array([
        [200, 200, 200],  # 0: Undefined
        [100, 150, 200],  # 1: Garage
        [200, 200, 150],  # 2: Storage
        [150, 100, 100],  # 3: Railing
        [255, 150, 200],  # 4: Entry
        [200, 150, 255],  # 5: Bath
        [180, 150, 220],  # 6: Bed Room
        [255, 100, 100],  # 7: Living Room
        [150, 255, 150],  # 8: Kitchen
        [220, 220, 220],  # 9: Wall
        [255, 200, 100],  # 10: Outdoor
        [100, 150, 255],  # 11: Background
    ], dtype=np.uint8)

    # Icon colormap
    icon_colors = {
        0: [200, 200, 200],  # No icon
        1: [255, 165, 0],    # Window - orange
        2: [0, 255, 0],      # Door - green
        3: [255, 255, 0],    # Closet - yellow
        4: [255, 0, 0],      # Electrical - red
        5: [128, 0, 128],    # Toilet - purple
        6: [255, 192, 203],  # Sink - pink
        7: [165, 42, 42],    # Sauna - brown
        8: [255, 105, 180],  # Fire place - hot pink
        9: [64, 224, 208],   # Bathtub - turquoise
        10: [139, 69, 19],   # Chimney - brown
    }

    # 1. Original Floorplan
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Floorplan', fontsize=18, fontweight='bold')
    ax1.axis('off')

    # 2. Room Segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    room_rgb = room_colors[rooms_pred % len(room_colors)]
    ax2.imshow(room_rgb)
    ax2.set_title('Room Segmentation', fontsize=18, fontweight='bold')
    ax2.axis('off')

    # Add legend for room segmentation
    room_labels = ['Undefined', 'Garage', 'Storage', 'Railing', 'Entry', 'Bath',
                   'Bed Room', 'Living Room', 'Kitchen', 'Wall', 'Outdoor', 'Background']
    legend_patches = [patches.Patch(color=room_colors[i]/255.0, label=room_labels[i])
                     for i in range(min(len(room_labels), len(room_colors)))]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=10, framealpha=0.9)

    # 3. All Icons (Raw DL Output)
    ax3 = fig.add_subplot(gs[0, 2])
    icons_argmax = torch.argmax(icons_logits, 0).cpu().data.numpy()

    # Create RGB image for icons
    icons_rgb = np.zeros((icons_argmax.shape[0], icons_argmax.shape[1], 3), dtype=np.uint8)
    icons_rgb[:] = [100, 150, 255]  # Background blue

    for icon_id, color in icon_colors.items():
        mask = (icons_argmax == icon_id)
        icons_rgb[mask] = color

    ax3.imshow(icons_rgb)
    ax3.set_title('All Icons (Raw DL Output)', fontsize=18, fontweight='bold')
    ax3.axis('off')

    # Add legend for icons
    icon_labels = ['Chimney', 'Bathtub', 'Fire Place', 'Sauna Bench', 'Sink',
                   'Toilet', 'Electrical Appliance', 'Closet', 'Door', 'Window', 'No Icon']
    icon_legend_patches = [patches.Patch(color=np.array(icon_colors.get(i, [200,200,200]))/255.0,
                                         label=icon_labels[i])
                          for i in range(min(len(icon_labels), 11))]
    ax3.legend(handles=icon_legend_patches, loc='upper right', fontsize=10, framealpha=0.9)

    # 4. Door Probability Map
    ax4 = fig.add_subplot(gs[1, 0])
    door_prob_display = ax4.imshow(door_prob, cmap='Reds', vmin=0, vmax=1)
    ax4.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), alpha=0.3)
    ax4.set_title(f'Door Probability Map\n(threshold={threshold})', fontsize=18, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(door_prob_display, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Window Probability Map
    ax5 = fig.add_subplot(gs[1, 1])
    window_prob_display = ax5.imshow(window_prob, cmap='Blues', vmin=0, vmax=1)
    ax5.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), alpha=0.3)
    ax5.set_title(f'Window Probability Map\n(threshold={threshold})', fontsize=18, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(window_prob_display, ax=ax5, fraction=0.046, pad=0.04)

    # 6. FINAL: Doors + Windows with bounding boxes
    ax6 = fig.add_subplot(gs[1, 2])
    final_img = img_display.copy()

    # Draw doors (green)
    for i, d in enumerate(doors, 1):
        x, y, w, h = d['bbox']
        cv2.rectangle(final_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Label
        label = f'D{i}'
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Red background for label
        cv2.rectangle(final_img, (x, y - text_h - 8), (x + text_w + 8, y), (0, 0, 255), -1)
        cv2.putText(final_img, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), thickness)

    # Draw windows (blue)
    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']
        cv2.rectangle(final_img, (x, y), (x+width, y+height), (255, 100, 0), 3)

        # Label
        label = f'W{i}'
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Blue background for label
        cv2.rectangle(final_img, (x, y - text_h - 8), (x + text_w + 8, y), (255, 0, 0), -1)
        cv2.putText(final_img, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), thickness)

    ax6.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    ax6.set_title(f'FINAL: {len(doors)} Doors + {len(windows)} Windows\n(Deep Learning Only)',
                 fontsize=18, fontweight='bold', color='green')
    ax6.axis('off')

    plt.suptitle('CubiCasa Floor Plan Detection Pipeline', fontsize=24, fontweight='bold', y=0.98)

    # Save
    output_path = 'plan_floor1_DL_stages.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nDetections (before geometric filtering):")
    print(f"  Doors:   {len(doors)}")
    print(f"  Windows: {len(windows)}")
    print(f"\nOutput:  {output_path}")

if __name__ == '__main__':
    main()
