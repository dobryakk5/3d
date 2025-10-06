#!/usr/bin/env python
"""
Improved inference with preprocessing and post-processing
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import cv2

from floortrans.models import get_model

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

def preprocess_architectural_plan(image_path, max_size=1024):
    """Enhanced preprocessing for architectural drawings"""
    img = Image.open(image_path).convert('RGB')
    print(f"   Original size: {img.size}")

    # Resize
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"   Resized to: {img.size}")

    # Convert to numpy
    img_np = np.array(img)

    # Enhance contrast to make architectural features more visible
    img_pil = Image.fromarray(img_np)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.5)  # Increase contrast
    img_np = np.array(img_pil)

    # Create normalized version for model
    img_normalized = img_np.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) * 2

    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img_normalized, img_np

def detect_doors_windows_opencv(image_np):
    """Fallback: use OpenCV to detect door arcs and window rectangles"""
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Detect circles/arcs (potential doors)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=50)

    door_mask = np.zeros(gray.shape, dtype=bool)
    window_mask = np.zeros(gray.shape, dtype=bool)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Mark as potential door
            cv2.circle(door_mask.astype(np.uint8), center, radius, 1, -1)

    # Detect lines (potential windows)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=20, maxLineGap=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Horizontal or vertical lines
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 15:  # Minimum length
                cv2.line(window_mask.astype(np.uint8), (x1, y1), (x2, y2), 1, 3)

    return door_mask.astype(bool), window_mask.astype(bool)

def combine_predictions(dl_doors, dl_windows, cv_doors, cv_windows):
    """Combine deep learning and computer vision predictions"""
    # Union of detections
    doors_combined = np.logical_or(dl_doors, cv_doors)
    windows_combined = np.logical_or(dl_windows, cv_windows)

    return doors_combined, windows_combined

def main():
    print("=" * 70)
    print("IMPROVED FLOORPLAN ANALYSIS WITH HYBRID APPROACH")
    print("=" * 70)

    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    max_size = 1024  # Higher resolution
    n_classes = 44

    print(f"\n[1/5] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("   ✓ Model loaded")

    print(f"\n[2/5] Preprocessing image with enhancement...")
    img_tensor, img_normalized, img_np = preprocess_architectural_plan(image_path, max_size)

    print(f"\n[3/5] Running deep learning inference...")
    with torch.no_grad():
        prediction = model(img_tensor)

    # Extract predictions
    rooms_logits = prediction[0, 21:33]
    icons_logits = prediction[0, 33:44]

    rooms_pred = F.softmax(rooms_logits, 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)

    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()
    icons_pred_class = np.argmax(icons_pred, axis=0)

    # Get probability masks for doors and windows
    door_prob = icons_pred[2]  # Door channel
    window_prob = icons_pred[1]  # Window channel

    # Threshold with lower value to catch more detections
    dl_doors = door_prob > 0.3
    dl_windows = window_prob > 0.3

    print(f"   DL detected: {dl_doors.sum()} door pixels, {dl_windows.sum()} window pixels")

    print(f"\n[4/5] Running OpenCV detection as fallback...")
    cv_doors, cv_windows = detect_doors_windows_opencv(img_np)
    print(f"   CV detected: {cv_doors.sum()} door pixels, {cv_windows.sum()} window pixels")

    print(f"\n[5/5] Combining predictions...")
    doors_final, windows_final = combine_predictions(dl_doors, dl_windows, cv_doors, cv_windows)

    print(f"   COMBINED: {doors_final.sum()} door pixels, {windows_final.sum()} window pixels")

    # Statistics
    print(f"\n{'=' * 70}")
    print("DETECTION RESULTS:")
    print(f"{'=' * 70}")
    print(f"  Doors detected:   {doors_final.sum():6d} pixels")
    print(f"  Windows detected: {windows_final.sum():6d} pixels")

    # Count distinct regions
    from scipy import ndimage
    door_labels, num_doors = ndimage.label(doors_final)
    window_labels, num_windows = ndimage.label(windows_final)

    print(f"\n  Distinct door regions:   {num_doors}")
    print(f"  Distinct window regions: {num_windows}")
    print(f"{'=' * 70}")

    # Visualize
    print(f"\nCreating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    img_show = (img_normalized + 1) / 2

    # Original
    axes[0, 0].imshow(img_show)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Room segmentation
    im1 = axes[0, 1].imshow(rooms_pred, cmap='tab20', vmin=0, vmax=11)
    axes[0, 1].set_title('Room Segmentation', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # DL Icons
    overlay_dl = np.zeros((*icons_pred_class.shape, 4))
    overlay_dl[dl_doors] = [1, 0, 0, 0.8]
    overlay_dl[dl_windows] = [0, 0, 1, 0.8]

    axes[0, 2].imshow(img_show)
    axes[0, 2].imshow(overlay_dl)
    axes[0, 2].set_title(f'Deep Learning Only\n(Doors={dl_doors.sum()}, Windows={dl_windows.sum()})',
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # CV detection
    overlay_cv = np.zeros((*cv_doors.shape, 4))
    overlay_cv[cv_doors] = [1, 0, 0, 0.8]
    overlay_cv[cv_windows] = [0, 0, 1, 0.8]

    axes[1, 0].imshow(img_show)
    axes[1, 0].imshow(overlay_cv)
    axes[1, 0].set_title(f'Computer Vision Only\n(Doors={cv_doors.sum()}, Windows={cv_windows.sum()})',
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Combined
    overlay_combined = np.zeros((*doors_final.shape, 4))
    overlay_combined[doors_final] = [1, 0, 0, 0.8]
    overlay_combined[windows_final] = [0, 0, 1, 0.8]

    axes[1, 1].imshow(img_show)
    axes[1, 1].imshow(overlay_combined)
    axes[1, 1].set_title(f'COMBINED (Hybrid)\n(Doors={doors_final.sum()}, Windows={windows_final.sum()})',
                         fontsize=12, fontweight='bold', color='green')
    axes[1, 1].axis('off')

    # Labels with numbers
    axes[1, 2].imshow(img_show)
    overlay_labeled = np.zeros((*doors_final.shape, 4))

    # Color each region differently
    for i in range(1, num_doors+1):
        mask = door_labels == i
        overlay_labeled[mask] = [1, 0, 0, 0.7]

    for i in range(1, num_windows+1):
        mask = window_labels == i
        overlay_labeled[mask] = [0, 0, 1, 0.7]

    axes[1, 2].imshow(overlay_labeled)
    axes[1, 2].set_title(f'Labeled Regions\n({num_doors} doors, {num_windows} windows)',
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = 'plan_floor1_improved.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close('all')

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'=' * 70}\n")

if __name__ == '__main__':
    main()
