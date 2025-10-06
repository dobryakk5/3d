#!/usr/bin/env python
"""
OCR-based detector for Russian CAD drawings
Finds doors (ДВ-*) and windows (ОК-*, ДН-*) by text labels

NO TRAINING REQUIRED - works immediately!
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not installed. Install: pip install pytesseract")

def detect_doors_by_arc(image_np):
    """Обнаружить двери по дугам (символ открывания)"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Улучшить контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Края
    edges = cv2.Canny(enhanced, 50, 150)

    # Детекция дуг/окружностей
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    door_candidates = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, (x, y, r) in enumerate(circles[0, :]):
            # Дуга двери обычно в углу или у стены
            door_candidates.append({
                'type': 'door_arc',
                'center': (int(x), int(y)),
                'radius': int(r),
                'bbox': (int(x-r), int(y-r), int(x+r), int(y+r)),
                'confidence': 0.6  # Numeric confidence
            })

    return door_candidates

def detect_text_labels_tesseract(image_np):
    """Распознать текстовые метки с помощью Tesseract OCR"""
    if not TESSERACT_AVAILABLE:
        return []

    # Конвертировать в grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Улучшить для OCR
    # Увеличить контраст
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Конфигурация для русского языка
    custom_config = r'--oem 3 --psm 6 -l rus+eng'

    try:
        # Распознать текст
        data = pytesseract.image_to_data(
            binary,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        detections = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()

            if not text:
                continue

            # Ищем паттерны дверей и окон
            is_door = False
            is_window = False

            # Двери: ДВ-, DV-, дв-
            if any(pattern in text.upper() for pattern in ['ДВ', 'DV', 'ДB']):
                is_door = True

            # Окна: ОК-, OK-, ДН-, DN-
            if any(pattern in text.upper() for pattern in ['ОК', 'OK', 'ОC', 'ДН', 'DN']):
                is_window = True

            if is_door or is_window:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                conf = data['conf'][i]

                detection = {
                    'type': 'door_text' if is_door else 'window_text',
                    'text': text,
                    'bbox': (x, y, x+w, y+h),
                    'center': (x + w//2, y + h//2),
                    'confidence': float(conf) / 100.0 if conf > 0 else 0.5
                }

                detections.append(detection)

        return detections

    except Exception as e:
        print(f"Tesseract error: {e}")
        return []

def detect_text_labels_opencv(image_np):
    """Простой детектор текста через контуры (fallback если нет Tesseract)"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Бинаризация
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Найти контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Фильтр: текст обычно небольшой и вытянутый
        if 10 < w < 100 and 5 < h < 30:
            aspect_ratio = w / h
            if 1.5 < aspect_ratio < 8:  # Текст обычно горизонтальный
                text_candidates.append({
                    'type': 'text_region',
                    'bbox': (x, y, x+w, y+h),
                    'center': (x + w//2, y + h//2),
                    'size': (w, h),
                    'confidence': 0.3  # Низкая уверенность без OCR
                })

    return text_candidates

def match_arcs_with_labels(arcs, labels, max_distance=150):
    """Связать дуги с текстовыми метками"""
    matched_doors = []

    for arc in arcs:
        arc_center = np.array(arc['center'])
        best_match = None
        min_distance = float('inf')

        for label in labels:
            if label['type'] != 'door_text':
                continue

            label_center = np.array(label['center'])
            distance = np.linalg.norm(arc_center - label_center)

            if distance < min_distance and distance < max_distance:
                min_distance = distance
                best_match = label

        if best_match:
            matched_doors.append({
                'type': 'door',
                'arc': arc,
                'label': best_match,
                'center': arc['center'],
                'text': best_match['text'],
                'confidence': (arc.get('confidence', 0.5) + best_match['confidence']) / 2
            })
        else:
            # Дуга без метки - всё равно считаем дверью
            matched_doors.append({
                'type': 'door',
                'arc': arc,
                'label': None,
                'center': arc['center'],
                'text': 'Unknown',
                'confidence': arc.get('confidence', 0.5) * 0.7  # Снижаем уверенность
            })

    return matched_doors

def find_wall_context(point, wall_mask):
    """Проверить находится ли точка рядом со стеной"""
    x, y = point

    # Проверить область вокруг точки
    margin = 50
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(wall_mask.shape[1], x + margin), min(wall_mask.shape[0], y + margin)

    region = wall_mask[y1:y2, x1:x2]

    # Если есть стена рядом - это правдоподобно
    wall_pixels = np.sum(region > 0)
    total_pixels = region.size

    wall_ratio = wall_pixels / total_pixels if total_pixels > 0 else 0

    return wall_ratio > 0.1  # Хотя бы 10% стены рядом

def main():
    print("=" * 80)
    print("OCR-BASED DOOR & WINDOW DETECTION (No Training Required)")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'

    print(f"\n[1/6] Loading image...")
    img = Image.open(image_path).convert('RGB')

    # Resize если нужно
    max_size = 2048
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"   Resized to: {img.size}")

    img_np = np.array(img)

    print(f"\n[2/6] Detecting door arcs...")
    door_arcs = detect_doors_by_arc(img_np)
    print(f"   Found {len(door_arcs)} door arc candidates")

    print(f"\n[3/6] Detecting text labels...")
    if TESSERACT_AVAILABLE:
        print("   Using Tesseract OCR (Russian + English)...")
        text_labels = detect_text_labels_tesseract(img_np)
    else:
        print("   Using OpenCV contour detection (fallback)...")
        text_labels = detect_text_labels_opencv(img_np)

    door_labels = [l for l in text_labels if 'door' in l['type']]
    window_labels = [l for l in text_labels if 'window' in l['type']]

    print(f"   Found {len(door_labels)} door labels (ДВ-*)")
    print(f"   Found {len(window_labels)} window labels (ОК-*, ДН-*)")

    print(f"\n[4/6] Matching arcs with labels...")
    doors_matched = match_arcs_with_labels(door_arcs, text_labels)
    print(f"   Matched {len(doors_matched)} doors")

    # Windows - только по меткам
    windows = []
    for label in window_labels:
        windows.append({
            'type': 'window',
            'label': label,
            'center': label['center'],
            'bbox': label['bbox'],
            'text': label['text'],
            'confidence': label['confidence']
        })

    print(f"   Found {len(windows)} windows")

    print(f"\n[5/6] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Original with all detections
    ax1 = axes[0, 0]
    ax1.imshow(img_np)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Door arcs
    ax2 = axes[0, 1]
    ax2.imshow(img_np)

    for arc in door_arcs:
        x, y = arc['center']
        r = arc['radius']
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
        ax2.add_patch(circle)
        ax2.plot(x, y, 'r+', markersize=10, markeredgewidth=2)

    ax2.set_title(f'Detected Door Arcs ({len(door_arcs)})', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Text labels
    ax3 = axes[1, 0]
    ax3.imshow(img_np)

    for label in door_labels:
        x1, y1, x2, y2 = label['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, edgecolor='red', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(x1, y1-5, label['text'], fontsize=8,
                color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for label in window_labels:
        x1, y1, x2, y2 = label['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, edgecolor='blue', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(x1, y1-5, label['text'], fontsize=8,
                color='blue', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_title(f'OCR Text Labels (D={len(door_labels)}, W={len(window_labels)})',
                  fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Final combined
    ax4 = axes[1, 1]
    ax4.imshow(img_np)

    for i, door in enumerate(doors_matched, 1):
        x, y = door['center']
        ax4.plot(x, y, 'ro', markersize=12, markeredgewidth=2)
        ax4.text(x, y-20, f"D{i}\n{door['text']}", fontsize=9,
                color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8))

    for i, window in enumerate(windows, 1):
        x, y = window['center']
        ax4.plot(x, y, 'bs', markersize=12, markeredgewidth=2)
        ax4.text(x, y-20, f"W{i}\n{window['text']}", fontsize=9,
                color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.8))

    ax4.set_title(f'FINAL: {len(doors_matched)} Doors + {len(windows)} Windows (OCR)',
                  fontsize=14, fontweight='bold', color='green')
    ax4.axis('off')

    output_path = 'plan_floor1_OCR_detection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

    print(f"\n[6/6] Exporting results...")

    # Export to JSON
    results = {
        'method': 'OCR-based detection (no training)',
        'tesseract_used': TESSERACT_AVAILABLE,
        'summary': {
            'doors': len(doors_matched),
            'windows': len(windows),
            'total': len(doors_matched) + len(windows)
        },
        'doors': doors_matched,
        'windows': windows
    }

    json_path = 'detections_OCR.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {json_path}")

    # Export to TXT
    txt_path = 'detections_OCR.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("OCR-BASED DETECTION RESULTS (No Training Required)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Method: Text recognition + Arc detection\n")
        f.write(f"Tesseract OCR: {'Yes' if TESSERACT_AVAILABLE else 'No (using fallback)'}\n\n")

        f.write(f"SUMMARY:\n")
        f.write(f"  Doors:   {len(doors_matched)}\n")
        f.write(f"  Windows: {len(windows)}\n")
        f.write(f"  Total:   {len(doors_matched) + len(windows)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DOORS:\n")
        f.write("=" * 80 + "\n\n")

        for i, door in enumerate(doors_matched, 1):
            f.write(f"Door #{i}:\n")
            f.write(f"  Text label: {door['text']}\n")
            f.write(f"  Position: {door['center']}\n")
            f.write(f"  Confidence: {door['confidence']:.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("WINDOWS:\n")
        f.write("=" * 80 + "\n\n")

        for i, window in enumerate(windows, 1):
            f.write(f"Window #{i}:\n")
            f.write(f"  Text label: {window['text']}\n")
            f.write(f"  Position: {window['center']}\n")
            f.write(f"  Confidence: {window['confidence']:.2f}\n\n")

    print(f"   ✓ Saved: {txt_path}")

    print(f"\n{'=' * 80}")
    print(f"{'OCR DETECTION COMPLETE':^80}")
    print(f"{'=' * 80}")
    print(f"\n  🚪 Doors:   {len(doors_matched)} (arcs + text labels)")
    print(f"  🪟 Windows: {len(windows)} (text labels only)")
    print(f"  📦 Total:   {len(doors_matched) + len(windows)}")

    if not TESSERACT_AVAILABLE:
        print(f"\n  ⚠️  Install Tesseract for better text recognition:")
        print(f"     macOS: brew install tesseract tesseract-lang")
        print(f"     Linux: sudo apt install tesseract-ocr tesseract-ocr-rus")
        print(f"     pip install pytesseract")

    print(f"\n  💡 This method works WITHOUT any training!")
    print(f"     It finds ДВ-*, ОК-*, ДН-* labels directly.")

    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
