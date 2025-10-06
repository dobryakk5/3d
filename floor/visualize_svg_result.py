"""Visualize the SVG detection results"""
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_previous_detections():
    """Load doors and windows"""
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


def detect_hatching(image, kernel_size=25, density_threshold=0.2):
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


def main():
    # Load image
    img = Image.open('plan_floor1.jpg').convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Load detections
    doors, windows = load_previous_detections()

    # Detect walls
    wall_mask = detect_hatching(img_array, kernel_size=25, density_threshold=0.2)

    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array)
    ax1.set_title('Original Floor Plan', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # Walls detected
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(wall_mask, cmap='gray')
    ax2.set_title('Detected Walls (from hatching)', fontsize=16, fontweight='bold')
    ax2.axis('off')

    # Doors and windows on original
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(img_array)
    for door in doors:
        x, y, w, h = door['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.3)
        ax3.add_patch(rect)
        ax3.text(x + w/2, y + h/2, door['label'], ha='center', va='center',
                color='white', fontsize=12, fontweight='bold')
    for window in windows:
        x, y, w, h = window['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='cyan', facecolor='cyan', alpha=0.3)
        ax3.add_patch(rect)
        ax3.text(x + w/2, y + h/2, window['label'], ha='center', va='center',
                color='black', fontsize=12, fontweight='bold')
    ax3.set_title(f'Doors ({len(doors)}) and Windows ({len(windows)})', fontsize=16, fontweight='bold')
    ax3.axis('off')

    # Complete overlay
    ax4 = fig.add_subplot(gs[1, 1])
    # Show walls
    ax4.imshow(wall_mask, cmap='gray', alpha=0.7)
    # Overlay original faintly
    ax4.imshow(img_array, alpha=0.3)
    # Draw doors
    for door in doors:
        x, y, w, h = door['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=4, edgecolor='blue', facecolor='blue', alpha=0.5)
        ax4.add_patch(rect)
        ax4.text(x + w/2, y + h/2, door['label'], ha='center', va='center',
                color='white', fontsize=14, fontweight='bold')
    # Draw windows
    for window in windows:
        x, y, w, h = window['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=4, edgecolor='cyan', facecolor='cyan', alpha=0.5)
        ax4.add_patch(rect)
        ax4.text(x + w/2, y + h/2, window['label'], ha='center', va='center',
                color='black', fontsize=14, fontweight='bold')
    ax4.set_title('Complete SVG Result (Walls + Doors + Windows)', fontsize=16, fontweight='bold')
    ax4.axis('off')

    # Add summary text
    fig.suptitle(f'Floor Plan Vectorization Result\nImage: {w}x{h} px | 60 wall segments, 7 doors, 5 windows',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('svg_verification.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved visualization to svg_verification.png")


if __name__ == '__main__':
    main()
