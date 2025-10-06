#!/usr/bin/env python
"""
Export detection results to structured formats (JSON, CSV, TXT)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import cv2
from scipy import ndimage
import json
import csv

from floortrans.models import get_model

def multi_scale_inference(model, img_tensor, scales=[0.75, 1.0, 1.25]):
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

def detect_architectural_elements(image_np):
    """Detect doors and windows using OpenCV"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges1 = cv2.Canny(blurred, 30, 100)
    edges2 = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)

    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Detect circles (doors)
    circles = cv2.HoughCircles(edges_dilated, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                               param1=50, param2=25, minRadius=8, maxRadius=80)

    door_mask = np.zeros(gray.shape, dtype=np.uint8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.ellipse(door_mask, (i[0], i[1]), (i[2], i[2]), 0, 0, 180, 255, -1)

    # Detect windows
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    window_mask = np.zeros(gray.shape, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0

                if 2 < aspect_ratio < 15 or (1/15 < aspect_ratio < 0.5):
                    cv2.drawContours(window_mask, [approx], 0, 255, -1)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                            minLineLength=25, maxLineGap=5)

    line_mask = np.zeros(gray.shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if length > 20:
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 10 or (80 < angle < 100) or angle > 170:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    window_mask = cv2.bitwise_or(window_mask, line_mask)

    return door_mask > 0, window_mask > 0

def refine_detections(dl_mask, cv_mask, min_size=20, max_size=5000):
    """Remove noise and filter by size"""
    combined = np.logical_or(dl_mask, cv_mask)
    labeled, num = ndimage.label(combined)

    refined = np.zeros_like(combined)

    for i in range(1, num + 1):
        region = labeled == i
        size = region.sum()

        if min_size <= size <= max_size:
            refined[region] = True

    return refined

def extract_detection_info(mask, element_type, scale_factor=1.0):
    """Extract bounding boxes and metadata for each detected region"""
    labeled, num_regions = ndimage.label(mask)

    detections = []

    for i in range(1, num_regions + 1):
        region = labeled == i

        # Get bounding box
        rows, cols = np.where(region)

        if len(rows) == 0:
            continue

        x_min, x_max = int(cols.min()), int(cols.max())
        y_min, y_max = int(rows.min()), int(rows.max())

        # Get center
        center_y, center_x = ndimage.center_of_mass(region)

        # Get area
        area = region.sum()

        # Scale to original image coordinates
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
            'area_scaled': int(area * scale_factor * scale_factor)
        }

        detections.append(detection)

    return detections

def main():
    print("=" * 80)
    print("EXPORTING DETECTION RESULTS")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    max_size = 2048
    n_classes = 44

    # Load original image to get scale factor
    img_orig = Image.open(image_path)
    orig_width, orig_height = img_orig.size

    print(f"\n[1/5] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"[2/5] Processing image...")
    img = Image.open(image_path).convert('RGB')

    # Resize
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        scale = 1.0

    processed_width, processed_height = img.size
    scale_to_original = orig_width / processed_width

    print(f"   Original size: {orig_width}x{orig_height}")
    print(f"   Processed size: {processed_width}x{processed_height}")
    print(f"   Scale factor: {scale_to_original:.2f}")

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    img_np = np.array(img)
    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    print(f"[3/5] Running detection...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    icons_logits = prediction[0, 33:44]
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    dl_doors = icons_pred[2] > 0.25
    dl_windows = icons_pred[1] > 0.25

    cv_doors, cv_windows = detect_architectural_elements(img_np)

    doors_final = refine_detections(dl_doors, cv_doors, min_size=15, max_size=8000)
    windows_final = refine_detections(dl_windows, cv_windows, min_size=15, max_size=8000)

    print(f"[4/5] Extracting detection information...")
    door_detections = extract_detection_info(doors_final, 'door', scale_to_original)
    window_detections = extract_detection_info(windows_final, 'window', scale_to_original)

    print(f"   Found {len(door_detections)} doors")
    print(f"   Found {len(window_detections)} windows")

    print(f"[5/5] Exporting results...")

    # Create summary
    results = {
        'image': {
            'filename': image_path,
            'original_size': {'width': orig_width, 'height': orig_height},
            'processed_size': {'width': processed_width, 'height': processed_height},
            'scale_factor': float(scale_to_original)
        },
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

    # Export to JSON
    json_path = 'detections.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved JSON: {json_path}")

    # Export to CSV (flat format)
    csv_path = 'detections.csv'
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

    print(f"   ✓ Saved CSV: {csv_path}")

    # Export to readable text
    txt_path = 'detections.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FLOORPLAN DETECTION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Image: {image_path}\n")
        f.write(f"Original size: {orig_width} x {orig_height} pixels\n\n")

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
            f.write(f"  Bounding Box: ({det['bbox']['x_min']}, {det['bbox']['y_min']}) to "
                   f"({det['bbox']['x_max']}, {det['bbox']['y_max']})\n")
            f.write(f"  Size: {det['bbox']['width']} x {det['bbox']['height']} pixels\n")
            f.write(f"  Area: {det['area_scaled']} pixels²\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"WINDOWS ({len(window_detections)} detected):\n")
        f.write("=" * 80 + "\n\n")

        for det in window_detections:
            f.write(f"Window W{det['id']}:\n")
            f.write(f"  Position: ({det['center']['x']}, {det['center']['y']})\n")
            f.write(f"  Bounding Box: ({det['bbox']['x_min']}, {det['bbox']['y_min']}) to "
                   f"({det['bbox']['x_max']}, {det['bbox']['y_max']})\n")
            f.write(f"  Size: {det['bbox']['width']} x {det['bbox']['height']} pixels\n")
            f.write(f"  Area: {det['area_scaled']} pixels²\n\n")

    print(f"   ✓ Saved TXT: {txt_path}")

    print(f"\n{'=' * 80}")
    print("EXPORT COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nResults saved to:")
    print(f"  - {json_path} (structured data)")
    print(f"  - {csv_path} (spreadsheet format)")
    print(f"  - {txt_path} (human-readable)")
    print(f"  - plan_floor1_final.png (visualization)")
    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
