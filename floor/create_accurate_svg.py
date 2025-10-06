"""
Create accurate SVG from floor plan using:
1. Previously detected doors/windows from detections_DL_only.txt
2. Wall detection from hatching
3. Junction points from heatmaps
"""
import cv2
import numpy as np
from PIL import Image
import svgwrite
import json


def detect_hatching(image, kernel_size=15, density_threshold=0.25):
    """Detect hatching patterns to find walls"""
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


def extract_wall_segments(wall_mask, min_length=20):
    """Extract wall segments as line segments"""
    skeleton = cv2.ximgproc.thinning(wall_mask)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        if len(contour) >= 2:
            contour = contour.squeeze()
            if len(contour.shape) == 1:
                continue
            if len(contour) < 2:
                continue

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


def load_previous_detections():
    """Load doors and windows from detections_DL_only.txt"""
    doors = [
        {'bbox': (1855, 363, 210, 65), 'label': 'D1'},
        {'bbox': (758, 1493, 169, 53), 'label': 'D2'},
        {'bbox': (1488, 1678, 55, 172), 'label': 'D3'},
        {'bbox': (544, 1700, 9, 101), 'label': 'D4'},
        {'bbox': (1903, 2079, 55, 220), 'label': 'D5'},
        {'bbox': (1676, 2365, 317, 70), 'label': 'D6'},
        {'bbox': (782, 2369, 222, 63), 'label': 'D7'},
    ]

    windows = [
        {'bbox': (1625, 364, 235, 58), 'label': 'W1'},
        {'bbox': (958, 400, 58, 7), 'label': 'W2'},
        {'bbox': (2616, 1348, 48, 155), 'label': 'W3'},
        {'bbox': (2617, 1610, 48, 150), 'label': 'W4'},
        {'bbox': (521, 1716, 50, 152), 'label': 'W5'},
    ]

    return doors, windows


def scale_detections(detections, scale_factor):
    """Scale detection coordinates"""
    scaled = []
    for det in detections:
        x, y, w, h = det['bbox']
        scaled.append({
            'bbox': (
                int(x * scale_factor),
                int(y * scale_factor),
                int(w * scale_factor),
                int(h * scale_factor)
            ),
            'label': det['label']
        })
    return scaled


def create_svg(image_shape, walls, doors, windows, output_path):
    """Create SVG file"""
    height, width = image_shape[:2]
    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')

    # Add original as background (commented out)
    # dwg.add(dwg.image(href='plan_floor1.jpg', insert=(0, 0), size=(width, height), opacity=0.2))

    # Draw walls (black)
    wall_group = dwg.add(dwg.g(id='walls', stroke='black', stroke_width=6, fill='none'))
    for wall in walls:
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        wall_group.add(dwg.line(
            start=(float(x1), float(y1)),
            end=(float(x2), float(y2)),
            stroke='black',
            stroke_width=6
        ))

    # Draw doors (blue rectangles)
    door_group = dwg.add(dwg.g(id='doors'))
    for door in doors:
        x, y, w, h = door['bbox']
        door_group.add(dwg.rect(
            insert=(float(x), float(y)),
            size=(float(w), float(h)),
            fill='blue',
            fill_opacity=0.5,
            stroke='darkblue',
            stroke_width=3
        ))
        # Add label
        text = dwg.text(
            door['label'],
            insert=(float(x + w/2), float(y + h/2)),
            text_anchor='middle',
            fill='white',
            font_size=min(w, h) * 0.5
        )
        text['font-weight'] = 'bold'
        door_group.add(text)

    # Draw windows (cyan rectangles)
    window_group = dwg.add(dwg.g(id='windows'))
    for window in windows:
        x, y, w, h = window['bbox']
        window_group.add(dwg.rect(
            insert=(float(x), float(y)),
            size=(float(w), float(h)),
            fill='cyan',
            fill_opacity=0.5,
            stroke='blue',
            stroke_width=3
        ))
        # Add label
        text = dwg.text(
            window['label'],
            insert=(float(x + w/2), float(y + h/2)),
            text_anchor='middle',
            fill='black',
            font_size=min(w, h) * 0.5
        )
        text['font-weight'] = 'bold'
        window_group.add(text)

    dwg.save()
    print(f"SVG saved to {output_path}")


def main():
    image_path = 'plan_floor1.jpg'
    output_svg = 'floor_plan_accurate.svg'

    print("Loading image...")
    img = Image.open(image_path).convert('RGB')
    original_w, original_h = img.size
    print(f"Original size: {original_w}x{original_h}")

    img_array = np.array(img)

    # Load previous detections (at original scale)
    print("Loading previous detections...")
    doors_original, windows_original = load_previous_detections()

    # Detect walls using hatching
    print("Detecting walls from hatching...")
    wall_mask = detect_hatching(img_array, kernel_size=25, density_threshold=0.2)

    # Extract wall segments
    print("Extracting wall segments...")
    wall_segments = extract_wall_segments(wall_mask, min_length=50)

    print(f"\nFound:")
    print(f"  Walls:   {len(wall_segments)} segments")
    print(f"  Doors:   {len(doors_original)}")
    print(f"  Windows: {len(windows_original)}")

    # Create SVG at original resolution
    print("\nCreating SVG...")
    create_svg(img_array.shape, wall_segments, doors_original, windows_original, output_svg)

    # Save summary
    summary = {
        'image': image_path,
        'size': f'{original_w}x{original_h}',
        'statistics': {
            'walls': len(wall_segments),
            'doors': len(doors_original),
            'windows': len(windows_original)
        },
        'doors': doors_original,
        'windows': windows_original,
        'wall_count': len(wall_segments)
    }

    with open('floor_plan_accurate_data.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  SVG:  {output_svg}")
    print(f"  Data: floor_plan_accurate_data.json")
    print(f"\n  Total elements: {len(wall_segments) + len(doors_original) + len(windows_original)}")


if __name__ == '__main__':
    main()
