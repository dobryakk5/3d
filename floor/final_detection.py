#!/usr/bin/env python
"""
Final optimized detection with multi-scale inference and post-processing
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import cv2
from scipy import ndimage

from floortrans.models import get_model

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

def multi_scale_inference(model, img_tensor, scales=[0.75, 1.0, 1.25]):
    """Run inference at multiple scales and average"""
    n_classes = 44
    h, w = img_tensor.shape[2], img_tensor.shape[3]

    # Accumulate predictions
    predictions = []

    for scale in scales:
        if scale != 1.0:
            new_h = int(h * scale)
            new_w = int(w * scale)
            scaled = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=True)
        else:
            scaled = img_tensor

        with torch.no_grad():
            pred = model(scaled)

        # Resize back to original size
        if scale != 1.0:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        predictions.append(pred)

    # Average predictions
    prediction = torch.stack(predictions).mean(dim=0)

    return prediction

def detect_architectural_elements(image_np):
    """Enhanced detection of doors (arcs) and windows using OpenCV"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Preprocess
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge detection with multiple thresholds
    edges1 = cv2.Canny(blurred, 30, 100)
    edges2 = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)

    # Morphological operations to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Detect circles (doors - shown as arcs)
    circles = cv2.HoughCircles(edges_dilated, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                               param1=50, param2=25, minRadius=8, maxRadius=80)

    door_mask = np.zeros(gray.shape, dtype=np.uint8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw arc (semi-circle for door swing)
            cv2.ellipse(door_mask, (i[0], i[1]), (i[2], i[2]),
                       0, 0, 180, 255, -1)

    # Detect windows (typically rectangular shapes with hatching)
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    window_mask = np.zeros(gray.shape, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Filter by area
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Look for rectangular shapes (windows)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Windows are typically elongated rectangles
                if 2 < aspect_ratio < 15 or (1/15 < aspect_ratio < 0.5):
                    cv2.drawContours(window_mask, [approx], 0, 255, -1)

    # Also detect straight lines (window frames)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                            minLineLength=25, maxLineGap=5)

    line_mask = np.zeros(gray.shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Only consider longer lines
            if length > 20:
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                # Horizontal or vertical
                if angle < 10 or (80 < angle < 100) or angle > 170:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Combine window detections
    window_mask = cv2.bitwise_or(window_mask, line_mask)

    return door_mask > 0, window_mask > 0

def refine_detections(dl_mask, cv_mask, min_size=20, max_size=5000):
    """Remove noise and filter by size"""
    # Combine masks
    combined = np.logical_or(dl_mask, cv_mask)

    # Label regions
    labeled, num = ndimage.label(combined)

    # Filter by size
    refined = np.zeros_like(combined)

    for i in range(1, num + 1):
        region = labeled == i
        size = region.sum()

        if min_size <= size <= max_size:
            refined[region] = True

    return refined

def main():
    print("=" * 80)
    print(" " * 20 + "FINAL OPTIMIZED DETECTION")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    max_size = 2048  # Maximum resolution
    n_classes = 44

    print(f"\n[1/6] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("   ✓ Model loaded")

    print(f"\n[2/6] Loading and preprocessing image...")
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size
    print(f"   Original: {orig_size}")

    # Resize
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"   Resized:  {img.size}")

    # Enhance for better feature visibility
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    img_np = np.array(img)
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    print(f"\n[3/6] Running multi-scale deep learning inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Extract icon predictions
    icons_logits = prediction[0, 33:44]
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    # Get masks with different thresholds
    door_prob = icons_pred[2]
    window_prob = icons_pred[1]

    # Use lower threshold to catch more
    dl_doors = door_prob > 0.25
    dl_windows = window_prob > 0.25

    print(f"   DL detected: {dl_doors.sum():6d} door pixels, {dl_windows.sum():6d} window pixels")

    print(f"\n[4/6] Running OpenCV architectural detection...")
    cv_doors, cv_windows = detect_architectural_elements(img_np)
    print(f"   CV detected: {cv_doors.sum():6d} door pixels, {cv_windows.sum():6d} window pixels")

    print(f"\n[5/6] Refining and combining detections...")
    doors_final = refine_detections(dl_doors, cv_doors, min_size=15, max_size=8000)
    windows_final = refine_detections(dl_windows, cv_windows, min_size=15, max_size=8000)

    # Count regions
    door_labels, num_doors = ndimage.label(doors_final)
    window_labels, num_windows = ndimage.label(windows_final)

    print(f"   FINAL:       {doors_final.sum():6d} door pixels, {windows_final.sum():6d} window pixels")

    print(f"\n[6/6] Creating detailed visualization...")

    # Get room segmentation too
    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()

    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

    img_show = (img_normalized + 1) / 2

    # Row 1: Input and segmentation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_show)
    ax1.set_title('Input Floorplan', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rooms_pred, cmap='tab20', vmin=0, vmax=11)
    ax2.set_title('Room Segmentation', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.argmax(icons_pred, axis=0), cmap='tab20', vmin=0, vmax=10)
    ax3.set_title('All Icons (Raw)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Row 2: Detection components
    ax4 = fig.add_subplot(gs[1, 0])
    overlay_dl = np.zeros((*dl_doors.shape, 4))
    overlay_dl[dl_doors] = [1, 0, 0, 0.7]
    overlay_dl[dl_windows] = [0, 0, 1, 0.7]
    ax4.imshow(img_show)
    ax4.imshow(overlay_dl)
    ax4.set_title(f'Deep Learning\nD={dl_doors.sum()}, W={dl_windows.sum()}',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    overlay_cv = np.zeros((*cv_doors.shape, 4))
    overlay_cv[cv_doors] = [1, 0, 0, 0.7]
    overlay_cv[cv_windows] = [0, 0, 1, 0.7]
    ax5.imshow(img_show)
    ax5.imshow(overlay_cv)
    ax5.set_title(f'Computer Vision\nD={cv_doors.sum()}, W={cv_windows.sum()}',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    overlay_final = np.zeros((*doors_final.shape, 4))
    overlay_final[doors_final] = [1, 0, 0, 0.8]
    overlay_final[windows_final] = [0, 0.5, 1, 0.8]
    ax6.imshow(img_show)
    ax6.imshow(overlay_final)
    ax6.set_title(f'COMBINED (Refined)\nD={doors_final.sum()}, W={windows_final.sum()}',
                  fontsize=12, fontweight='bold', color='green')
    ax6.axis('off')

    # Row 3: Final results with labels
    ax7 = fig.add_subplot(gs[2, :])
    ax7.imshow(img_show)

    # Colorize each detected region
    cmap_doors = plt.cm.Reds
    cmap_windows = plt.cm.Blues

    overlay_labeled = np.zeros((*doors_final.shape, 4))

    for i in range(1, num_doors + 1):
        mask = door_labels == i
        color = cmap_doors(0.5 + 0.3 * (i % 5) / 5)
        overlay_labeled[mask] = color

        # Add label number
        y, x = ndimage.center_of_mass(mask)
        if not np.isnan(x) and not np.isnan(y):
            ax7.text(x, y, f'D{i}', fontsize=10, fontweight='bold',
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

    for i in range(1, num_windows + 1):
        mask = window_labels == i
        color = cmap_windows(0.5 + 0.3 * (i % 5) / 5)
        overlay_labeled[mask] = color

        # Add label number
        y, x = ndimage.center_of_mass(mask)
        if not np.isnan(x) and not np.isnan(y):
            ax7.text(x, y, f'W{i}', fontsize=10, fontweight='bold',
                    color='white', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))

    ax7.imshow(overlay_labeled)
    ax7.set_title(f'FINAL DETECTION: {num_doors} Doors (Red) + {num_windows} Windows (Blue)',
                  fontsize=16, fontweight='bold', color='darkgreen', pad=20)
    ax7.axis('off')

    output_path = 'plan_floor1_final.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")

    plt.close('all')

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"{'FINAL DETECTION SUMMARY':^80}")
    print(f"{'=' * 80}")
    print(f"\n  {'Detected Elements:':<30} {'Count':<10} {'Pixels':<10}")
    print(f"  {'-' * 50}")
    print(f"  {'Doors:':<30} {num_doors:<10} {doors_final.sum():<10}")
    print(f"  {'Windows:':<30} {num_windows:<10} {windows_final.sum():<10}")
    print(f"\n{'=' * 80}")

    # Compare with ground truth from plan
    print(f"\n  NOTE: Please visually verify the detections against the original plan.")
    print(f"  Expected elements on plan (manual count from annotations):")
    print(f"    - Doors: Look for arc symbols (ДВ-1, ДВ-2, etc.)")
    print(f"    - Windows: Look for 'ОК-' labels (ОК-1, ОК-8, ДН-3, etc.)")
    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
