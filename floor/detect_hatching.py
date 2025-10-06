#!/usr/bin/env python
"""
Automatic wall detection by finding hatching patterns
Finds diagonal lines (штриховка) and marks them as walls
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import json

def detect_hatching_patterns(image_np, visualize=False):
    """
    Обнаружить штриховку на чертеже

    Штриховка = повторяющиеся диагональные линии
    """
    print("\n[1/6] Converting to grayscale and preprocessing...")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Улучшить контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Бинаризация
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY_INV)

    if visualize:
        cv2.imwrite('debug_1_binary.png', binary)

    print("[2/6] Detecting line patterns...")

    # Детекция линий разных направлений
    # Штриховка обычно под углом 45° или -45°
    hatching_masks = []

    # Углы штриховки (в градусах)
    angles = [45, -45, 135, -135]  # Диагональная штриховка

    for angle in angles:
        print(f"   Checking {angle}° lines...")

        # Создать структурный элемент для линий под углом
        length = 15  # Длина линии для поиска
        kernel = create_line_kernel(length, angle)

        # Морфологическая операция - найти линии
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if visualize:
            cv2.imwrite(f'debug_2_lines_{angle}.png', detected)

        hatching_masks.append(detected)

    # Объединить все направления
    hatching_combined = np.zeros_like(binary)
    for mask in hatching_masks:
        hatching_combined = cv2.bitwise_or(hatching_combined, mask)

    if visualize:
        cv2.imwrite('debug_3_hatching_combined.png', hatching_combined)

    print("[3/6] Finding hatching regions...")

    # Найти области с плотной штриховкой
    # Штриховка = много линий в одном месте
    kernel_dilate = np.ones((5, 5), np.uint8)
    hatching_dilated = cv2.dilate(hatching_combined, kernel_dilate, iterations=2)

    # Подсчитать плотность штриховки
    kernel_density = np.ones((20, 20), np.uint8)
    density = cv2.filter2D(hatching_dilated.astype(np.float32), -1, kernel_density)

    # Нормализовать
    density = (density / density.max() * 255).astype(np.uint8)

    if visualize:
        cv2.imwrite('debug_4_density.png', density)

    # Порог плотности - где много линий = стена
    threshold = 30  # Минимальная плотность штриховки
    wall_mask = (density > threshold).astype(np.uint8) * 255

    print("[4/6] Refining wall boundaries...")

    # Убрать шум
    kernel_close = np.ones((7, 7), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_close)

    # Заполнить дыры
    kernel_fill = np.ones((15, 15), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_fill)

    if visualize:
        cv2.imwrite('debug_5_walls_refined.png', wall_mask)

    return wall_mask, hatching_combined, density

def create_line_kernel(length, angle_deg):
    """Создать kernel для детекции линий под углом"""
    angle_rad = np.deg2rad(angle_deg)

    # Создать линию
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2

    # Нарисовать линию под углом
    for i in range(length):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))

        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1

    return kernel

def detect_hatching_by_frequency(image_np):
    """
    Альтернативный метод: анализ частот
    Штриховка создает регулярный паттерн частот
    """
    print("\n[ALT] Using frequency analysis...")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # FFT для поиска периодических паттернов
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Найти пики частот (регулярная штриховка даёт пики)
    # Это покажет где есть повторяющиеся линии

    return magnitude

def extract_wall_contours(wall_mask):
    """Извлечь контуры стен"""
    contours, hierarchy = cv2.findContours(
        wall_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Фильтр по размеру
    wall_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Минимальная площадь стены
            wall_contours.append(contour)

    return wall_contours

def analyze_walls(wall_mask, scale_factor=1.0):
    """Анализ обнаруженных стен"""
    # Подсчет пикселей стен
    wall_pixels = np.sum(wall_mask > 0)
    total_pixels = wall_mask.size
    coverage = wall_pixels / total_pixels * 100

    # Скелетонизация для определения длины
    from skimage.morphology import skeletonize
    skeleton = skeletonize(wall_mask > 0)
    wall_length = np.sum(skeleton)

    # Средняя толщина
    avg_thickness = wall_pixels / wall_length if wall_length > 0 else 0

    # Контуры
    contours = extract_wall_contours(wall_mask)

    stats = {
        'wall_pixels': int(wall_pixels),
        'coverage_percent': float(coverage),
        'estimated_length_px': int(wall_length * scale_factor),
        'average_thickness_px': int(avg_thickness * scale_factor),
        'num_segments': len(contours),
        'wall_area_px2': int(wall_pixels * scale_factor * scale_factor)
    }

    return stats

def main():
    print("=" * 80)
    print("AUTOMATIC WALL DETECTION BY HATCHING PATTERN RECOGNITION")
    print("=" * 80)

    image_path = 'plan_floor1.jpg'

    print(f"\nLoading image: {image_path}")
    img = Image.open(image_path).convert('RGB')

    # Resize для обработки
    max_size = 2048
    orig_size = img.size
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"Resized from {orig_size} to {img.size}")

    scale_factor = orig_size[0] / img.width

    img_np = np.array(img)

    # Детекция штриховки
    wall_mask, hatching, density = detect_hatching_patterns(img_np, visualize=True)

    print("\n[5/6] Analyzing detected walls...")
    stats = analyze_walls(wall_mask, scale_factor)

    print(f"\n   Wall pixels: {stats['wall_pixels']:,}")
    print(f"   Coverage: {stats['coverage_percent']:.2f}%")
    print(f"   Segments: {stats['num_segments']}")
    print(f"   Estimated length: {stats['estimated_length_px']:,} px")
    print(f"   Avg thickness: {stats['average_thickness_px']} px")

    print("\n[6/6] Creating visualization...")

    # Создать визуализацию
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hatching, cmap='gray')
    ax2.set_title('Detected Line Patterns (Hatching)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(density, cmap='hot')
    ax3.set_title('Hatching Density Map', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(ax3.imshow(density, cmap='hot'), ax=ax3, fraction=0.046)

    # Row 2
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(wall_mask, cmap='gray')
    ax4.set_title(f'Detected Walls (Binary)\n{stats["num_segments"]} segments',
                  fontsize=14, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img_np)
    ax5.imshow(wall_mask, cmap='Reds', alpha=0.6)
    ax5.set_title(f'Walls Overlay\nCoverage: {stats["coverage_percent"]:.1f}%',
                  fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    # Контуры стен
    contour_img = img_np.copy()
    contours = extract_wall_contours(wall_mask)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 3)
    ax6.imshow(contour_img)
    ax6.set_title(f'Wall Contours\n{len(contours)} segments',
                  fontsize=14, fontweight='bold')
    ax6.axis('off')

    # Row 3: Comparison with DL
    ax7 = fig.add_subplot(gs[2, :])

    # Если есть результат от DL - сравним
    try:
        import torch
        import torch.nn.functional as F
        from floortrans.models import get_model

        print("\n   Loading DL model for comparison...")
        model = get_model('hg_furukawa_original', 51)
        model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)

        checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # Predict
        img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)

        rooms_logits = pred[0, 21:33]
        rooms_pred = torch.argmax(rooms_logits, 0).cpu().numpy()
        dl_walls = (rooms_pred == 2).astype(np.uint8) * 255

        # Сравнение
        comparison = np.zeros((*img_np.shape[:2], 3), dtype=np.uint8)

        # Красный = только hatching метод
        only_hatching = (wall_mask > 0) & (dl_walls == 0)
        comparison[only_hatching] = [255, 0, 0]

        # Синий = только DL
        only_dl = (wall_mask == 0) & (dl_walls > 0)
        comparison[only_dl] = [0, 0, 255]

        # Зелёный = оба метода согласны
        both = (wall_mask > 0) & (dl_walls > 0)
        comparison[both] = [0, 255, 0]

        ax7.imshow(img_np)
        ax7.imshow(comparison, alpha=0.5)
        ax7.set_title('Comparison: Red=Hatching only, Blue=DL only, Green=Both agree',
                      fontsize=14, fontweight='bold', color='darkgreen')
        ax7.axis('off')

        # Statistics
        hatching_only_px = np.sum(only_hatching)
        dl_only_px = np.sum(only_dl)
        both_px = np.sum(both)
        agreement = both_px / (hatching_only_px + dl_only_px + both_px) * 100

        print(f"\n   Comparison with DL model:")
        print(f"   - Agreement: {agreement:.1f}%")
        print(f"   - Hatching only: {hatching_only_px:,} px")
        print(f"   - DL only: {dl_only_px:,} px")
        print(f"   - Both agree: {both_px:,} px")

    except Exception as e:
        print(f"\n   Could not compare with DL: {e}")
        ax7.text(0.5, 0.5, 'DL comparison not available',
                ha='center', va='center', fontsize=16,
                transform=ax7.transAxes)
        ax7.axis('off')

    output_path = 'plan_floor1_HATCHING_DETECTION.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

    # Сохранить маску стен
    mask_path = 'wall_mask_auto.png'
    cv2.imwrite(mask_path, wall_mask)
    print(f"✓ Saved wall mask: {mask_path}")

    # Export JSON
    results = {
        'method': 'Hatching pattern detection',
        'image': {
            'path': image_path,
            'original_size': list(orig_size),
            'processed_size': list(img.size),
            'scale_factor': float(scale_factor)
        },
        'statistics': stats,
        'files': {
            'visualization': output_path,
            'wall_mask': mask_path
        }
    }

    json_path = 'hatching_detection.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata: {json_path}")

    print(f"\n{'=' * 80}")
    print("HATCHING DETECTION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\n  🧱 Detected {stats['num_segments']} wall segments")
    print(f"  📏 Total wall area: {stats['wall_area_px2']:,} px²")
    print(f"  📊 Coverage: {stats['coverage_percent']:.2f}%")
    print(f"\n  💾 Wall mask saved to: {mask_path}")
    print(f"     Use this as training data for fine-tuning!")
    print(f"\n{'=' * 80}\n")

    return wall_mask, stats

if __name__ == '__main__':
    main()
