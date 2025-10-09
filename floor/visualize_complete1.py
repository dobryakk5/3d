#!/usr/bin/env python3
"""
Complete detection visualization with detection method details:
1. Original image
2. Junctions (labeled by type: window/door/wall)
3. Windows (with detection method info)
4. Doors (with detection method info)
5. Walls (DL vs Hatching texture)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import sys
sys.path.append('floortrans')
from floortrans.models import get_model

# Import detection functions
exec(open('cubicasa_vectorize.py').read().split('if __name__')[0])

def categorize_junctions(junctions_dict):
    """Collect all junctions without categorization"""
    all_junctions = []
    for jtype, points in junctions_dict.items():
        for p in points:
            all_junctions.append({**p, 'type': jtype})
    return all_junctions

def is_point_convex(contour, point_idx):
    """
    Determine if a point in contour is convex (external) or concave (internal)
    Returns: True if convex (external angle), False if concave
    """
    n = len(contour)
    if n < 3:
        return True

    prev = contour[(point_idx - 1) % n]
    curr = contour[point_idx]
    nxt = contour[(point_idx + 1) % n]

    # Vectors from current point
    v1 = prev - curr
    v2 = nxt - curr

    # Cross product for 2D (z-component)
    # Positive = counter-clockwise turn (convex for CCW contour)
    # Negative = clockwise turn (concave for CCW contour)
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    # For CCW contour: positive cross = convex (external)
    # For CW contour: negative cross = convex (external)
    # We assume CCW orientation (gift wrapping produces CCW)
    return cross > 0

def analyze_detection_method(detection, wall_segments, all_junctions):
    """Determine how a detection was validated"""
    methods = ['DL']  # All start with DL

    x, y, w, h = detection['bbox']
    center_x, center_y = x + w/2, y + h/2

    # Check if on wall
    on_wall = False
    for seg in wall_segments:
        x1, y1 = seg['start']
        x2, y2 = seg['end']

        # Check distance to wall line
        dx = x2 - x1
        dy = y2 - y1
        line_len_sq = dx*dx + dy*dy

        if line_len_sq == 0:
            continue

        t = max(0, min(1, ((center_x - x1) * dx + (center_y - y1) * dy) / line_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.sqrt((center_x - proj_x)**2 + (center_y - proj_y)**2)

        if dist < 30:
            on_wall = True
            break

    if on_wall:
        methods.append('Wall-aligned')

    # Aspect ratio check removed - accept all detections

    # Check for nearby junctions
    has_junctions = False
    for j in all_junctions:
        jx, jy = j['x'], j['y']
        if abs(jx - center_x) < w + 50 and abs(jy - center_y) < h + 50:
            has_junctions = True
            break

    if has_junctions:
        methods.append('Junctions')

    return methods

def main(perimeter_algorithm='alpha_shape', output_suffix=''):
    """
    Main visualization function

    Args:
        perimeter_algorithm: 'alpha_shape', 'gift_wrapping', or 'boundary_trace'
        output_suffix: suffix for output filename (e.g., '_v1', '_v2')
    """
    print("="*80)
    print(f"COMPLETE DETECTION VISUALIZATION - {perimeter_algorithm}")
    print("="*80)

    image_path = 'plan_floor1.jpg'

    # Load model
    print("\n[1/5] Loading model...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Preprocess
    print("\n[2/5] Preprocessing...")
    img_orig = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img_orig.size

    max_size = 2048
    w, h = orig_width, orig_height
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        img = img_orig

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    img_np = np.array(img)
    img_display = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Inference
    print("\n[3/5] Running DL inference...")
    prediction = multi_scale_inference(model, img_tensor, scales=[0.9, 1.0, 1.1])

    # Extract icons
    icons_logits = prediction[0, 33:44]
    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()

    door_prob = icons_pred[2]
    window_prob = icons_pred[1]

    threshold = 0.3
    door_mask = (door_prob > threshold).astype(np.uint8)
    window_mask = (window_prob > threshold).astype(np.uint8)

    _, doors = refine_dl_detections(door_mask, min_size=50, max_size=6000)
    _, windows = refine_dl_detections(window_mask, min_size=50, max_size=6000)

    # Detect walls
    print("\n[4/5] Detecting walls...")
    wall_mask_hatching = detect_hatching(img_np, kernel_size=25, density_threshold=0.2)
    wall_segments_hatching = extract_wall_segments(wall_mask_hatching, min_length=50)

    rooms_logits = prediction[0, 21:33]
    rooms_pred = torch.argmax(rooms_logits, 0).cpu().data.numpy()
    wall_mask_dl = (rooms_pred == 2).astype(np.uint8) * 255
    wall_segments_dl = extract_wall_segments(wall_mask_dl, min_length=50)

    # Tag walls with source
    for seg in wall_segments_hatching:
        seg['source'] = 'hatching'
    for seg in wall_segments_dl:
        seg['source'] = 'DL'

    wall_segments = wall_segments_hatching + wall_segments_dl

    # Extract junctions (without type categorization)
    heatmaps = prediction[0, :21].cpu().data.numpy()
    junctions_dict = extract_junctions(prediction, threshold=threshold)
    all_junctions = categorize_junctions(junctions_dict)

    # Filter windows/doors
    windows = remove_nested_boxes(windows, iou_threshold=0.15)
    doors = remove_nested_boxes(doors, iou_threshold=0.5)

    # Stricter window filtering: edges must be on walls
    def filter_windows_strict(windows, wall_segments, max_distance=15):
        """Keep only windows whose edges are on walls"""
        valid = []
        for w in windows:
            x, y, width, height = w['bbox']

            # Check only edge midpoints (not center)
            edges = [
                (x + width/2, y),              # top edge
                (x + width/2, y + height),     # bottom edge
                (x, y + height/2),             # left edge
                (x + width, y + height/2),     # right edge
            ]

            # Check if at least one edge is on a wall
            on_wall = False
            for px, py in edges:
                for seg in wall_segments:
                    x1, y1 = seg['start']
                    x2, y2 = seg['end']
                    dist = point_to_line_distance(px, py, x1, y1, x2, y2)
                    if dist < max_distance:
                        on_wall = True
                        break
                if on_wall:
                    break

            if on_wall:
                valid.append(w)

        return valid

    windows = filter_windows_strict(windows, wall_segments, max_distance=15)

    # Filter doors: reject if window_prob > door_prob at center
    def filter_door_vs_window(doors, door_prob, window_prob):
        """Reject doors where model predicts higher window probability"""
        valid_doors = []
        for door in doors:
            x, y, w, h = door['bbox']
            cx, cy = int(x + w/2), int(y + h/2)

            if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
                d_conf = door_prob[cy, cx]
                w_conf = window_prob[cy, cx]

                # Keep only if door confidence is higher than window
                if d_conf > w_conf:
                    valid_doors.append(door)
            else:
                # Keep if out of bounds (shouldn't happen)
                valid_doors.append(door)

        return valid_doors

    doors_before = len(doors)
    doors = filter_door_vs_window(doors, door_prob, window_prob)
    print(f"   After door/window disambiguation: {len(doors)}/{doors_before} doors kept")

    # Analyze detection methods
    print("\n[5/5] Analyzing detection methods...")
    for w in windows:
        w['methods'] = analyze_detection_method(w, wall_segments, all_junctions)
    for d in doors:
        d['methods'] = analyze_detection_method(d, wall_segments, all_junctions)

    print(f"   Windows: {len(windows)}, Doors: {len(doors)}")
    print(f"   Walls: {len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL")
    print(f"   Junctions: {len(all_junctions)}")

    # Print door detection details with DL predictions
    print("\nDoor detection details:")
    for i, d in enumerate(doors, 1):
        x, y, w, h = d['bbox']
        methods = ' + '.join(d['methods'])
        aspect = max(w, h) / max(min(w, h), 1)

        # Check DL prediction at door location
        cx, cy = int(x + w/2), int(y + h/2)
        if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
            door_conf = door_prob[cy, cx]
            window_conf = window_prob[cy, cx]
            print(f"   D{i}: bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f}) aspect={aspect:.1f} door_prob={door_conf:.3f} window_prob={window_conf:.3f} methods=[{methods}]")
        else:
            print(f"   D{i}: bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f}) aspect={aspect:.1f} methods=[{methods}]")

    # Build closed contour and filter noise
    print("\nBuilding house contour...")

    # Debug: check connectivity
    print(f"   Total wall segments before filtering: {len(wall_segments)}")

    # Filter isolated wall segments - keep ALL connected components (not just largest)
    def filter_connected_walls(wall_segments, min_component_size=3):
        """Keep only wall segments that belong to large connected components"""
        if not wall_segments:
            return []

        # Build adjacency graph
        from collections import defaultdict
        graph = defaultdict(set)

        def points_close(p1, p2, threshold=15):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < threshold

        # Connect segments that share endpoints
        for i, seg1 in enumerate(wall_segments):
            for j, seg2 in enumerate(wall_segments):
                if i >= j:
                    continue

                # Check if segments are connected
                if (points_close(seg1['start'], seg2['start']) or
                    points_close(seg1['start'], seg2['end']) or
                    points_close(seg1['end'], seg2['start']) or
                    points_close(seg1['end'], seg2['end'])):
                    graph[i].add(j)
                    graph[j].add(i)

        # Find connected components using DFS
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(len(wall_segments)):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(component)

        # Keep ALL components that meet size threshold (not just largest!)
        valid_segments = []
        if components:
            print(f"   Found {len(components)} connected components:")
            for idx, comp in enumerate(components):
                print(f"     Component {idx+1}: {len(comp)} segments")
                if len(comp) >= min_component_size:
                    valid_segments.extend([wall_segments[i] for i in comp])

        return valid_segments if valid_segments else wall_segments

    wall_segments_before = len(wall_segments)
    wall_segments = filter_connected_walls(wall_segments, min_component_size=5)
    print(f"   Filtered isolated walls: {len(wall_segments)}/{wall_segments_before} segments kept")

    # Collect all wall points
    all_wall_points = []
    for seg in wall_segments:
        all_wall_points.append(seg['start'])
        all_wall_points.append(seg['end'])

    if all_wall_points:
        all_wall_points = np.array(all_wall_points)

        # Find bounding box of main structure
        x_coords = all_wall_points[:, 0]
        y_coords = all_wall_points[:, 1]

        # Use percentiles to ignore outliers (moderate filtering)
        x_min = np.percentile(x_coords, 5)
        x_max = np.percentile(x_coords, 95)
        y_min = np.percentile(y_coords, 5)
        y_max = np.percentile(y_coords, 95)

        # Add margin
        margin = 50
        x_min = max(0, int(x_min - margin))
        x_max = min(img_display.shape[1], int(x_max + margin))
        y_min = max(0, int(y_min - margin))
        y_max = min(img_display.shape[0], int(y_max + margin))

        crop_bounds = (x_min, x_max, y_min, y_max)

        # Filter wall segments - keep only those FULLY inside main structure
        main_wall_segments = []
        for seg in wall_segments:
            x1, y1 = seg['start']
            x2, y2 = seg['end']

            # Both endpoints must be inside bounds
            start_inside = (x_min <= x1 <= x_max and y_min <= y1 <= y_max)
            end_inside = (x_min <= x2 <= x_max and y_min <= y2 <= y_max)

            if start_inside and end_inside:
                main_wall_segments.append(seg)

        # Filter junctions - keep only those inside main structure
        main_junctions = []
        for j in all_junctions:
            if x_min <= j['x'] <= x_max and y_min <= j['y'] <= y_max:
                main_junctions.append(j)

        print(f"   Filtered: {len(main_wall_segments)}/{len(wall_segments)} walls inside main structure")
        print(f"   Crop bounds: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    else:
        crop_bounds = None
        main_wall_segments = wall_segments
        main_junctions = all_junctions

    # Create visualization
    print("\nCreating visualization...")

    fig = plt.figure(figsize=(48, 32))

    # Compute hatching density map for visualization
    def make_gabor_kernels(ksize=31, sigmas=[4.0], thetas=None, lambd=10.0, gamma=0.5):
        """Создаёт набор gabor-ядр для разных углов и масштабов."""
        if thetas is None:
            thetas = np.linspace(0, np.pi, 8, endpoint=False)  # 8 направлений
        kernels = []
        for sigma in sigmas:
            for theta in thetas:
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                # нормируем ядро по сумме абсолютных значений, чтобы ответы были сопоставимы
                if np.sum(np.abs(kern)) > 0:
                    kern = kern / np.sum(np.abs(kern))
                kernels.append((kern, theta, sigma))
        return kernels

    def compute_hatching_density(image, kernel_size=25):
        """Compute hatching density map using Gabor filters (optimized for fine hatching)"""
        # Get Gabor energy map (normalized)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Remove GaussianBlur to preserve fine texture contrast
        
        # Prepare Gabor kernels - tuned for fine hatching in pillars
        # Увеличиваем количество направлений для лучшего детектирования всех углов
        thetas = np.linspace(0, np.pi, 72, endpoint=False)  # Увеличили до 72 направлений
        kernels = make_gabor_kernels(ksize=15, sigmas=[1.0, 2.0, 4.0, 6.0], thetas=thetas, lambd=5.0, gamma=0.5)
        
        # Apply filters and accumulate energy
        h, w = gray.shape
        energy_map = np.zeros((h, w), dtype=np.float32)
        for kern, theta, sigma in kernels:
            resp = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            energy_map += np.abs(resp)
        
        # Normalize with median + MAD
        med = np.median(energy_map)
        mad = np.median(np.abs(energy_map - med)) + 1e-6
        energy_norm = (energy_map - med) / mad
        
        # Local density (sliding average) - smaller window for fine texture
        kernel_local = np.ones((9, 9), dtype=np.float32) / 81
        density = cv2.filter2D(energy_norm, -1, kernel_local)
        
        # Применяем адаптивный порог для лучшего детектирования столбов
        # Используем более низкий порог для детектирования слабых текстур
        adaptive_threshold = np.percentile(density, 85)  # Используем 85-й перцентиль как порог
        wall_mask_adaptive = (density > adaptive_threshold * 0.3).astype(np.uint8) * 255
        
        # Дополнительная обработка для улучшения маски
        wall_mask_adaptive = cv2.morphologyEx(wall_mask_adaptive, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        wall_mask_adaptive = cv2.morphologyEx(wall_mask_adaptive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Normalize to 0-1 for visualization
        density_normalized = density / density.max() if density.max() > 0 else density
        
        # Save debug images
        debug_energy = ((energy_norm - energy_norm.min()) / (energy_norm.max() - energy_norm.min()) * 255).astype(np.uint8)
        cv2.imwrite('debug_gabor_energy.png', debug_energy)
        cv2.imwrite('debug_wall_mask.png', wall_mask_adaptive)
        print("Saved debug images: debug_gabor_energy.png, debug_wall_mask.png")
        print(f"   Adaptive threshold: {adaptive_threshold:.3f}")
        print(f"   Wall mask coverage: {wall_mask_adaptive.sum() / (h*w) * 100:.1f}%")
        
        return density_normalized

    def detect_pillars(image, wall_mask, wall_segments, min_area=100, max_area=60000, debug=False):
        """Detect pillars with wall hatching texture (optimized for small pillars)"""
        # 1. Найти компоненты в маске штриховки
        labeled, num = ndimage.label(wall_mask)

        def is_enclosed_by_walls(bbox, wall_segments, tolerance=30):
            """Check if bbox is enclosed by a rectangle of wall segments"""
            x, y, w, h = bbox
            center_x = x + w / 2
            center_y = y + h / 2

            # Check for walls on all 4 sides (within tolerance)
            has_left = False
            has_right = False
            has_top = False
            has_bottom = False

            for seg in wall_segments:
                x1, y1 = seg['start']
                x2, y2 = seg['end']

                # Check if segment is vertical (left or right wall)
                if abs(x2 - x1) < 20:  # Vertical wall
                    seg_x = (x1 + x2) / 2
                    seg_y_min = min(y1, y2)
                    seg_y_max = max(y1, y2)

                    # Check if center is between segment's y range
                    if seg_y_min - tolerance < center_y < seg_y_max + tolerance:
                        if seg_x < center_x - w/2 - tolerance:  # Left wall
                            has_left = True
                        elif seg_x > center_x + w/2 + tolerance:  # Right wall
                            has_right = True

                # Check if segment is horizontal (top or bottom wall)
                if abs(y2 - y1) < 20:  # Horizontal wall
                    seg_y = (y1 + y2) / 2
                    seg_x_min = min(x1, x2)
                    seg_x_max = max(x1, x2)

                    # Check if center is between segment's x range
                    if seg_x_min - tolerance < center_x < seg_x_max + tolerance:
                        if seg_y < center_y - h/2 - tolerance:  # Top wall
                            has_top = True
                        elif seg_y > center_y + h/2 + tolerance:  # Bottom wall
                            has_bottom = True

            return has_left and has_right and has_top and has_bottom

        pillars = []
        debug_img = None
        if debug:
            debug_img = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)

        # 2. Перебрать все компоненты с фильтрацией по форме и размеру
        for i in range(1, num + 1):
            comp = (labeled == i)
            area = comp.sum()

            if min_area < area < max_area:  # Фильтр размера
                # Получить bounding box
                y_coords, x_coords = np.where(comp)
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                aspect = max(w, h) / max(min(w, h), 1)

                # Check if pillar is enclosed by walls (standalone pillars only)
                bbox = (x_min, y_min, w, h)
                if not is_enclosed_by_walls(bbox, wall_segments, tolerance=50):
                    print(f"      Component at ({x_min},{y_min},{w}x{h}): REJECTED - not enclosed by wall rectangle")
                    if debug:
                        debug_img[comp] = [128, 128, 128]  # Gray - not enclosed
                    continue
                
                # Адаптивная фильтрация по форме и соотношению сторон в зависимости от размера
                if area < 500:  # Маленькие компоненты
                    max_aspect = 10.0  # Более разрешительные пороги для маленьких
                    min_aspect = 0.2
                    print(f"        Small component: relaxed aspect ratio limits (0.2-{max_aspect})")
                elif area < 2000:  # Средние компоненты
                    max_aspect = 6.0
                    min_aspect = 0.25
                    print(f"        Medium component: moderate aspect ratio limits (0.25-{max_aspect})")
                else:  # Большие компоненты
                    max_aspect = 4.0
                    min_aspect = 0.3
                    print(f"        Large component: strict aspect ratio limits (0.3-{max_aspect})")
                
                # Исключаем слишком вытянутые компоненты (не столбы)
                if aspect > max_aspect:  # Слишком вытянутые
                    if debug:
                        debug_img[comp] = [255, 0, 255]  # Фиолетовый — исключен по аспекту
                    print(f"      Component at ({x_min},{y_min},{w}x{h}): REJECTED - aspect ratio too high ({aspect:.1f} > {max_aspect})")
                    continue
                
                # Проверка на квадратную или близкую к квадратной форме (типичная для столбов)
                if aspect < min_aspect:  # Слишком узкие
                    if debug:
                        debug_img[comp] = [255, 165, 0]  # Оранжевый — исключен по форме
                    print(f"      Component at ({x_min},{y_min},{w}x{h}): REJECTED - unusual shape (aspect={aspect:.1f} < {min_aspect})")
                    continue
                
                # 3. Проверить наличие текстуры (диагональных линий) с более строгим порогом
                crop = wall_mask[y_min:y_max+1, x_min:x_max+1]
                
                # Детектировать диагональные линии в компоненте
                def has_diagonal_texture(crop_img, angles=[45, -45, 135, -135], min_ratio=0.15):
                    """Проверить, есть ли в компоненте диагональная штриховка (адаптивный порог в зависимости от размера)"""
                    total_pixels = crop_img.sum() / 255  # Нормализуем к количеству белых пикселей
                    if total_pixels == 0:
                        return False
                    
                    # Логируем информацию о компоненте
                    print(f"      Component at ({x_min},{y_min},{w}x{h}): area={area:.0f}, total_pixels={total_pixels:.0f}, aspect={aspect:.1f}")
                    
                    # Проверяем плотность заполнения компонента
                    fill_ratio = total_pixels / (w * h)
                    print(f"        Fill ratio: {fill_ratio:.3f}")
                    
                    # Адаптивные пороги в зависимости от размера компонента
                    if area < 500:  # Маленькие компоненты (как квадрат слева внизу)
                        # Для маленьких квадратов используем более мягкие пороги
                        min_fill_ratio, max_fill_ratio = 0.2, 0.95  # Более широкий диапазон
                        min_texture_ratio = 0.08  # Снижаем порог текстуры для маленьких
                        print(f"        Small component: using relaxed thresholds")
                    elif area < 2000:  # Средние компоненты
                        min_fill_ratio, max_fill_ratio = 0.15, 0.9
                        min_texture_ratio = 0.12
                        print(f"        Medium component: using moderate thresholds")
                    else:  # Большие компоненты
                        min_fill_ratio, max_fill_ratio = 0.1, 0.8
                        min_texture_ratio = 0.15
                        print(f"        Large component: using strict thresholds")
                    
                    # Проверка плотности заполнения с адаптивными порогами
                    if fill_ratio < min_fill_ratio or fill_ratio > max_fill_ratio:
                        print(f"        ✗ REJECTED - unusual fill ratio: {fill_ratio:.3f} (range: {min_fill_ratio}-{max_fill_ratio})")
                        return False
                    
                    # Для маленьких квадратов используем проверку на реальную диагональную структуру
                    if area < 500 and 0.8 < aspect < 1.2:  # Маленький квадрат
                        print(f"        Small square: using gradient-based texture analysis")
                        
                        # Получаем оригинальное изображение для анализа градиентов
                        x_start, y_start = x_min, y_min
                        x_end, y_end = x_min + w, y_min + h
                        
                        # Проверяем границы
                        if (x_start < 0 or y_start < 0 or
                            x_end >= image.shape[1] or y_end >= image.shape[0]):
                            print(f"        ✗ FAILED - square outside image bounds")
                            return False
                        
                        # Извлекаем область из оригинального изображения
                        crop_original = image[y_start:y_end, x_start:x_end]
                        if crop_original.size == 0:
                            print(f"        ✗ FAILED - empty crop")
                            return False
                        
                        # Конвертируем в градации серого
                        if len(crop_original.shape) == 3:
                            crop_gray = cv2.cvtColor(crop_original, cv2.COLOR_RGB2GRAY)
                        else:
                            crop_gray = crop_original
                        
                        # Вычисляем градиенты
                        grad_x = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
                        grad_y = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
                        
                        # Вычисляем направление градиентов
                        grad_direction = np.arctan2(grad_y, grad_x)
                        
                        # Проверяем наличие диагональных градиентов (45° и 135°)
                        diag_angles = [np.deg2rad(45), np.deg2rad(135), np.deg2rad(-45), np.deg2rad(-135)]
                        tolerance = np.deg2rad(15)  # Допуск 15 градусов
                        
                        diag_pixels = 0
                        for angle in diag_angles:
                            mask = np.abs(grad_direction - angle) < tolerance
                            diag_pixels += np.sum(mask)
                        
                        # Проверяем, что достаточно пикселей имеют диагональные градиенты
                        total_pixels = w * h
                        diag_ratio = diag_pixels / total_pixels
                        print(f"        Diagonal gradient ratio: {diag_ratio:.3f}")
                        
                        # Для настоящих столбов должно быть достаточно диагональных градиентов
                        if diag_ratio < 0.2:  # Менее 20% диагональных градиентов
                            print(f"        ✗ FAILED - insufficient diagonal gradients ({diag_ratio:.3f} < 0.2)")
                            return False
                        
                        # Дополнительная проверка на наличие линий с помощью Hough Transform
                        edges = cv2.Canny(crop_gray, 50, 150, apertureSize=3)
                        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(w, h)/2))
                        
                        if lines is None:
                            print(f"        ✗ FAILED - no lines detected with Hough transform")
                            return False
                        
                        # Проверяем минимальное количество линий для надежной детекции
                        min_lines_required = max(10, min(w, h))  # Минимум 10 линий или размер компонента
                        if len(lines) < min_lines_required:
                            print(f"        ✗ FAILED - insufficient lines detected ({len(lines)} < {min_lines_required})")
                            return False
                        
                        # Проверяем, что достаточно линий имеют диагональное направление
                        diag_lines = 0
                        for line in lines:
                            rho, theta = line[0]
                            for angle in diag_angles:
                                if abs(theta - angle) < tolerance or abs(theta - (angle + np.pi)) < tolerance:
                                    diag_lines += 1
                                    break
                        
                        line_ratio = diag_lines / len(lines) if len(lines) > 0 else 0
                        print(f"        Diagonal line ratio: {line_ratio:.3f} ({diag_lines}/{len(lines)})")
                        
                        if line_ratio < 0.3:  # Менее 30% диагональных линий
                            print(f"        ✗ FAILED - insufficient diagonal lines ({line_ratio:.3f} < 0.3)")
                            return False
                        
                        print(f"      ✓ PASSED gradient-based texture check for small square")
                        return True
                    
                    # Стандартная проверка для остальных компонентов
                    for angle in angles:
                        # Создать ядро для угла - увеличенный размер для лучшего детектирования
                        kernel = np.zeros((11, 11), dtype=np.uint8)
                        center = 5
                        angle_rad = np.deg2rad(angle)
                        for i in range(11):
                            offset = i - center
                            x = int(center + offset * np.cos(angle_rad))
                            y = int(center + offset * np.sin(angle_rad))
                            if 0 <= x < 11 and 0 <= y < 11:
                                kernel[y, x] = 1
                        
                        # Применить морфологию
                        detected = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel, iterations=1)
                        detected_pixels = detected.sum() / 255
                        
                        # Проверяем с адаптивным порогом текстуры
                        ratio = detected_pixels / total_pixels if total_pixels > 0 else 0
                        print(f"        Angle {angle}°: detected={detected_pixels:.0f}, ratio={ratio:.3f}, threshold={min_texture_ratio}")
                        
                        if detected_pixels >= min_texture_ratio * total_pixels:
                            print(f"      ✓ PASSED texture check at angle {angle}°")
                            return True
                    
                    print(f"      ✗ FAILED texture check (all angles)")
                    return False
                
                has_texture = has_diagonal_texture(crop)
                if has_texture:
                    pillars.append((x_min, y_min, w, h, area, aspect))
                
                if debug:
                    if has_texture:
                        debug_img[comp] = [0, 255, 0]  # Зелёный — прошёл
                    else:
                        debug_img[comp] = [0, 0, 255]  # Красный — не прошёл
        
        if debug:
            cv2.imwrite('debug_pillars_texture.png', debug_img)
        
        return pillars

    hatching_density = compute_hatching_density(img_np, kernel_size=25)

    # 1. Original
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    ax1.set_title('1. Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Junctions (all types)
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw all junctions in same color
    for j in all_junctions:
        ax2.plot(j['x'], j['y'], 'o', color='red',
                markersize=10, markeredgecolor='black', markeredgewidth=1.5)

    ax2.set_title(f"2. Junctions\n{len(all_junctions)} detected", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Windows
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']

        # Draw box
        rect = patches.Rectangle((x, y), width, height, linewidth=3,
                                edgecolor='cyan', facecolor='cyan', alpha=0.3)
        ax3.add_patch(rect)

        # Label
        methods_str = ' + '.join(w['methods'])
        label = f"W{i}"
        ax3.text(x + width/2, y - 5, label, fontsize=10, fontweight='bold',
                color='blue', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='blue', linewidth=0.5))

    ax3.set_title(f'3. Windows ({len(windows)} detected)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Add detection methods legend
    methods_text = "Detection methods:\n"
    for i, w in enumerate(windows, 1):
        methods_text += f"W{i}: {' + '.join(w['methods'])}\n"

    ax3.text(1.02, 0.5, methods_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
            family='monospace')

    # 4. Doors
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    for i, d in enumerate(doors, 1):
        x, y, width, height = d['bbox']

        # Draw box
        rect = patches.Rectangle((x, y), width, height, linewidth=3,
                                edgecolor='green', facecolor='green', alpha=0.3)
        ax4.add_patch(rect)

        # Label
        label = f"D{i}"
        ax4.text(x + width/2, y - 5, label, fontsize=10, fontweight='bold',
                color='darkgreen', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='green', linewidth=0.5))

    ax4.set_title(f'4. Doors ({len(doors)} detected)', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Add detection methods
    methods_text = "Detection methods:\n"
    for i, d in enumerate(doors, 1):
        methods_text += f"D{i}: {' + '.join(d['methods'])}\n"

    ax4.text(1.02, 0.5, methods_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            family='monospace')

    # 5. Walls (separated by source)
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), alpha=0.4)

    # Draw hatching walls in red
    for seg in wall_segments_hatching:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        ax5.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7, label='_nolegend_')

    # Draw DL walls in blue
    for seg in wall_segments_dl:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        ax5.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7, label='_nolegend_')

    title = f"5. Walls by Source\n"
    title += f"Hatching: {len(wall_segments_hatching)}  DL: {len(wall_segments_dl)}"
    ax5.set_title(title, fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Legend
    legend_patches = [
        patches.Patch(color='red', label=f'Hatching texture ({len(wall_segments_hatching)})'),
        patches.Patch(color='blue', label=f'DL segmentation ({len(wall_segments_dl)})')
    ]
    ax5.legend(handles=legend_patches, loc='upper right', fontsize=10)

    # 6. Combined result
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw walls
    for seg in wall_segments:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        color = 'red' if seg['source'] == 'hatching' else 'blue'
        ax6.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.5)

    # Draw windows
    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']
        rect = patches.Rectangle((x, y), width, height, linewidth=2,
                                edgecolor='cyan', facecolor='none')
        ax6.add_patch(rect)
        ax6.text(x + width/2, y + height/2, f"W{i}", fontsize=8,
                color='blue', ha='center', va='center', fontweight='bold')

    # Draw doors
    for i, d in enumerate(doors, 1):
        x, y, width, height = d['bbox']
        rect = patches.Rectangle((x, y), width, height, linewidth=2,
                                edgecolor='green', facecolor='none')
        ax6.add_patch(rect)
        ax6.text(x + width/2, y + height/2, f"D{i}", fontsize=8,
                color='darkgreen', ha='center', va='center', fontweight='bold')

    # Draw junctions
    for j in all_junctions:
        ax6.plot(j['x'], j['y'], 'o', color='red',
                markersize=4, markeredgecolor='black', markeredgewidth=0.5)

    ax6.set_title('6. Complete Result', fontsize=14, fontweight='bold', color='darkgreen')
    ax6.axis('off')

    # 7. FINAL - Full view with exterior perimeter
    ax7 = plt.subplot(2, 4, 7)

    # Show full image (no crop)
    ax7.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Draw all filtered walls
    for seg in wall_segments:
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        color = 'red' if seg['source'] == 'hatching' else 'blue'
        ax7.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.3)

    # Draw windows
    for i, w in enumerate(windows, 1):
        x, y, width, height = w['bbox']
        rect = patches.Rectangle((x, y), width, height,
                                linewidth=2, edgecolor='cyan', facecolor='none')
        ax7.add_patch(rect)
        ax7.text(x + width/2, y + height/2, f"W{i}",
                fontsize=8, color='blue', ha='center', va='center',
                fontweight='bold')

    # Draw doors
    for i, d in enumerate(doors, 1):
        x, y, width, height = d['bbox']
        rect = patches.Rectangle((x, y), width, height,
                                linewidth=2, edgecolor='green', facecolor='none')
        ax7.add_patch(rect)
        ax7.text(x + width/2, y + height/2, f"D{i}",
                fontsize=8, color='darkgreen', ha='center', va='center',
                fontweight='bold')

    # Draw junctions
    for j in all_junctions:
        ax7.plot(j['x'], j['y'], 'o',
                color='red', markersize=5,
                markeredgecolor='black', markeredgewidth=0.5)

    # Build exterior perimeter from walls using selected algorithm
    pillar_debug = []  # Initialize for all algorithms
    if wall_segments:
        try:
            if perimeter_algorithm == 'alpha_shape':
                # ALGORITHM 1: Alpha-shape (concave hull)
                print(f"   Using algorithm: Alpha-shape (concave hull)")

                # Collect all wall endpoints
                all_wall_points = []
                for seg in wall_segments:
                    all_wall_points.append(seg['start'])
                    all_wall_points.append(seg['end'])

                all_wall_points = np.array(all_wall_points)

                # Try alpha-shape with scipy
                from scipy.spatial import Delaunay
                tri = Delaunay(all_wall_points)

                # Build edge set with length threshold
                alpha = 150  # Threshold distance
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        p1_idx = simplex[i]
                        p2_idx = simplex[(i+1)%3]
                        p1 = all_wall_points[p1_idx]
                        p2 = all_wall_points[p2_idx]
                        length = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

                        if length < alpha:
                            edge = tuple(sorted([p1_idx, p2_idx]))
                            edges.add(edge)

                # Find boundary edges (appear only once in triangle set)
                from collections import defaultdict
                edge_count = defaultdict(int)
                for simplex in tri.simplices:
                    for i in range(3):
                        p1_idx = simplex[i]
                        p2_idx = simplex[(i+1)%3]
                        edge = tuple(sorted([p1_idx, p2_idx]))
                        if edge in edges:
                            edge_count[edge] += 1

                # Boundary edges appear in only one triangle
                boundary_edges = [e for e, count in edge_count.items() if count == 1]

                # Build adjacency graph
                graph = defaultdict(list)
                for e in boundary_edges:
                    graph[e[0]].append(e[1])
                    graph[e[1]].append(e[0])

                # Trace boundary
                if boundary_edges:
                    start_idx = min(graph.keys(), key=lambda i: (all_wall_points[i][0], all_wall_points[i][1]))
                    perimeter_indices = [start_idx]
                    current = start_idx
                    visited = {start_idx}

                    while len(visited) < len(graph):
                        neighbors = [n for n in graph[current] if n not in visited]
                        if not neighbors:
                            break
                        next_idx = neighbors[0]
                        perimeter_indices.append(next_idx)
                        visited.add(next_idx)
                        current = next_idx

                    perimeter_points = all_wall_points[perimeter_indices]
                else:
                    # Fallback to convex hull
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(all_wall_points)
                    perimeter_points = all_wall_points[hull.vertices]

            elif perimeter_algorithm == 'gift_wrapping':
                # ALGORITHM 2: Gift Wrapping from walls + openings + hatching-detected pillars
                print(f"   Using algorithm: Gift Wrapping (walls + openings + pillar detection)")

                # Collect boundary points with priorities (walls > pillars > windows/doors)
                all_boundary_points = []
                point_priorities = []  # Track source priority: 0=wall, 1=pillar, 2=window/door

                # 1. Add wall segment endpoints (already filtered from isolated components) - HIGHEST PRIORITY
                for seg in wall_segments:
                    all_boundary_points.append(seg['start'])
                    point_priorities.append(0)
                    all_boundary_points.append(seg['end'])
                    point_priorities.append(0)

                # 2. Detect support pillars with wall hatching texture - MEDIUM PRIORITY
                wall_mask = detect_hatching(img_np, kernel_size=15, density_threshold=0.15)
                pillar_list = detect_pillars(img_np, wall_mask, wall_segments, min_area=50, max_area=60000, debug=True)

                # Calculate building center to determine which pillar corners to add
                if wall_segments:
                    all_wall_points_temp = []
                    for seg in wall_segments:
                        all_wall_points_temp.extend([seg['start'], seg['end']])
                    building_center_x = np.mean([p[0] for p in all_wall_points_temp])
                    building_center_y = np.mean([p[1] for p in all_wall_points_temp])
                else:
                    building_center_x = img_np.shape[1] / 2
                    building_center_y = img_np.shape[0] / 2

                pillar_count = 0
                pillar_debug = []
                for x, y, w, h, area, aspect in pillar_list:
                    # Determine pillar position relative to building center
                    pillar_center_x = x + w / 2
                    pillar_center_y = y + h / 2

                    # Add only the external corner based on quadrant
                    is_left = pillar_center_x < building_center_x
                    is_top = pillar_center_y < building_center_y

                    if is_left and is_top:
                        # Top-left quadrant: add top-left corner only
                        all_boundary_points.append((x, y))
                        point_priorities.append(1)
                    elif is_left and not is_top:
                        # Bottom-left quadrant: add bottom-left corner only
                        all_boundary_points.append((x, y + h))
                        point_priorities.append(1)
                    elif not is_left and is_top:
                        # Top-right quadrant: add top-right corner only
                        all_boundary_points.append((x + w, y))
                        point_priorities.append(1)
                    else:
                        # Bottom-right quadrant: add bottom-right corner only
                        all_boundary_points.append((x + w, y + h))
                        point_priorities.append(1)

                    pillar_count += 1
                    pillar_debug.append(f"({x},{y},{w}x{h},area={area:.0f},aspect={aspect:.1f})")

                # 3. Add window corners - LOWEST PRIORITY
                for w in windows:
                    x, y, width, height = w['bbox']
                    all_boundary_points.extend([
                        (x, y), (x + width, y),
                        (x, y + height), (x + width, y + height)
                    ])
                    point_priorities.extend([2, 2, 2, 2])

                # 4. Add door corners - LOWEST PRIORITY
                for d in doors:
                    x, y, width, height = d['bbox']
                    all_boundary_points.extend([
                        (x, y), (x + width, y),
                        (x, y + height), (x + width, y + height)
                    ])
                    point_priorities.extend([2, 2, 2, 2])

                # Найти крайние точки для отладки
                x_right_temp = np.max([p[0] for p in all_boundary_points])
                y_bottom_temp = np.max([p[1] for p in all_boundary_points])
                print(f"   Current boundary extent: x_right={x_right_temp:.0f}, y_bottom={y_bottom_temp:.0f}")

                # Show top 5 rightmost points
                print(f"   Top 5 rightmost boundary points:")
                sorted_by_x = sorted(all_boundary_points, key=lambda p: p[0], reverse=True)[:5]
                for p in sorted_by_x:
                    print(f"      Point: ({p[0]:.0f}, {p[1]:.0f})")

                # Show top 5 bottommost points
                print(f"   Top 5 bottommost boundary points:")
                sorted_by_y = sorted(all_boundary_points, key=lambda p: p[1], reverse=True)[:5]
                for p in sorted_by_y:
                    print(f"      Point: ({p[0]:.0f}, {p[1]:.0f})")

                # Фильтруем точки, чтобы убрать те, что находятся далеко за пределами изображения
                # и оставить только одну точку для правого нижнего угла
                img_height, img_width = img_np.shape[:2]
                margin = 50  # Небольшой запас за пределами изображения
                
                # Отфильтровываем точки, которые находятся слишком далеко за пределами изображения
                filtered_points = []
                for point in all_boundary_points:
                    x, y = point
                    # Оставляем точку, если она в пределах изображения или с небольшим запасом
                    if (-margin <= x <= img_width + margin and
                        -margin <= y <= img_height + margin):
                        filtered_points.append(point)
                
                all_boundary_points = np.array(filtered_points)
                print(f"   Filtered out-of-bounds points: {len(all_boundary_points)} points remaining")
                
                # Специальная обработка правого нижнего угла: оставляем только одну точку
                # Определяем правый нижний угол изображения
                img_height, img_width = img_np.shape[:2]
                corner_threshold = 150  # Расстояние от угла для определения точек в углу (увеличено)
                
                # Находим точки в правом нижнем углу
                # Используем реальные максимальные координаты из точек, а не размер изображения
                x_right = np.max([p[0] for p in all_boundary_points])
                y_bottom = np.max([p[1] for p in all_boundary_points])

                right_bottom_points = []
                for i, point in enumerate(all_boundary_points):
                    x, y = point
                    # Проверяем расстояние от реального правого нижнего угла здания
                    dist_to_right = x_right - x
                    dist_to_bottom = y_bottom - y
                    if dist_to_right < corner_threshold and dist_to_bottom < corner_threshold:
                        right_bottom_points.append((i, point))
                        print(f"      Point near RB corner: ({x:.0f}, {y:.0f}), dist_to_right={dist_to_right:.0f}, dist_to_bottom={dist_to_bottom:.0f}")

                # Если найдено несколько точек в правом нижнем углу, оставляем только одну
                if len(right_bottom_points) > 1:
                    print(f"   Found {len(right_bottom_points)} points in right bottom corner, keeping only one")

                    # Удаляем все точки в правом нижнем углу, кроме одной
                    # Выбираем точку, которая правее и ниже всех (максимальные x и y)
                    best_point_idx = None
                    best_score = -float('inf')

                    for idx, point in right_bottom_points:
                        x, y = point
                        # Выбираем точку с максимальным x + y (самая правая и нижняя)
                        score = x + y
                        print(f"      Candidate: ({x:.0f}, {y:.0f}), score={score:.0f}")
                        if score > best_score:
                            best_score = score
                            best_point_idx = idx

                    print(f"   Selected point: ({all_boundary_points[best_point_idx][0]:.0f}, {all_boundary_points[best_point_idx][1]:.0f})")
                    
                    # Создаем новый массив точек, удаляя все точки в углу, кроме одной
                    new_points = []
                    for i, point in enumerate(all_boundary_points):
                        x, y = point
                        dist_to_right = img_width - x
                        dist_to_bottom = img_height - y
                        
                        # Оставляем точку, если она не в правом нижнем углу
                        # или если это лучшая точка в углу
                        if not (dist_to_right < corner_threshold and dist_to_bottom < corner_threshold):
                            new_points.append(point)
                        elif i == best_point_idx:
                            new_points.append(point)
                    
                    all_boundary_points = np.array(new_points)
                    print(f"   Kept 1 point in right bottom corner, total points: {len(all_boundary_points)}")
                
                wall_endpoints = len(wall_segments) * 2
                print(f"   Using {len(all_boundary_points)} points ({wall_endpoints} wall endpoints + {len(windows)*4} window corners + {len(doors)*4} door corners + {pillar_count} pillars)")
                if pillar_count > 0:
                    print(f"   Detected {pillar_count} support pillars from hatching texture:")
                    for p in pillar_debug:
                        print(f"      Pillar: {p}")

                # Filter out duplicate or very close points to avoid multiple bends at corners
                def filter_close_points(points, priorities, threshold=30):
                    """Remove points that are too close to each other, prioritizing wall points"""
                    if len(points) <= 1:
                        return points

                    # Group points that are close to each other
                    groups = []
                    used = [False] * len(points)

                    for i, point in enumerate(points):
                        if used[i]:
                            continue

                        # Find all points close to this one
                        group = [i]
                        used[i] = True

                        for j, other_point in enumerate(points):
                            if i != j and not used[j]:
                                dist = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                                if dist < threshold:
                                    group.append(j)
                                    used[j] = True

                        groups.append(group)

                    # For each group, select the best point to represent the corner
                    filtered_points = []
                    for group in groups:
                        if len(group) == 1:
                            # Only one point, keep it
                            filtered_points.append(points[group[0]])
                        else:
                            # Multiple points close together, select the best one
                            best_point = None
                            best_score = -float('inf')

                            for idx in group:
                                point = points[idx]
                                priority = priorities[idx]

                                # Additional scoring: prefer points that are more extreme
                                img_h, img_w = img_np.shape[:2]

                                # Check if in right-bottom corner area - OVERRIDE priority, use only coordinates
                                if point[0] > img_w * 0.8 and point[1] > img_h * 0.8:
                                    # In right-bottom corner: ONLY use max x+y, ignore priority
                                    score = point[0] + point[1]  # Direct coordinates, no division
                                else:
                                    # For other areas: use priority-based scoring
                                    # Start with priority score (walls=100, pillars=50, windows/doors=0)
                                    score = (2 - priority) * 100

                                    # Check if this point is an endpoint of wall segments
                                    for seg in wall_segments:
                                        start = np.array(seg['start'])
                                        end = np.array(seg['end'])

                                        dist_to_start = np.linalg.norm(point - start)
                                        dist_to_end = np.linalg.norm(point - end)

                                        if dist_to_start < 5 or dist_to_end < 5:
                                            score += 10

                                    # Prefer extremity from local center
                                    center = np.mean([points[i] for i in group], axis=0)
                                    dist_from_center = np.linalg.norm(point - center)
                                    score += dist_from_center / 10

                                if score > best_score:
                                    best_score = score
                                    best_point = point

                            filtered_points.append(best_point)

                    return np.array(filtered_points)

                # Apply Gift Wrapping to ALL boundary points (creates single unified perimeter)
                def gift_wrapping(points):
                    n = len(points)
                    if n < 3:
                        return points

                    hull = []
                    start = np.argmin(points[:, 0])  # Leftmost point
                    current = start

                    while True:
                        hull.append(current)
                        next_point = (current + 1) % n

                        for i in range(n):
                            if i == current:
                                continue
                            # Cross product to find leftmost turn
                            v1 = points[i] - points[current]
                            v2 = points[next_point] - points[current]
                            cross = v1[0] * v2[1] - v1[1] * v2[0]
                            if cross > 0:
                                next_point = i

                        current = next_point
                        if current == start:
                            break

                    return points[hull]

                # Filter out close points to avoid multiple bends at corners
                filtered_points = filter_close_points(all_boundary_points, point_priorities, threshold=30)
                print(f"   Filtered {len(all_boundary_points)} points to {len(filtered_points)} unique points")
                
                # Create unified perimeter
                perimeter_points = gift_wrapping(filtered_points)
                print(f"   Built unified perimeter with {len(perimeter_points)} vertices")

            else:  # 'boundary_trace'
                # ALGORITHM 3: Boundary tracing along wall segments
                print(f"   Using algorithm: Boundary tracing (right-hand rule)")

                # This is the existing implementation
                # Build graph of connected wall segments
                from collections import defaultdict
                graph = defaultdict(list)

                # Round coordinates to handle floating point issues
                def pt_key(pt):
                    return (round(pt[0]), round(pt[1]))

                # Build adjacency graph from filtered walls
                for seg in wall_segments:
                    start_key = pt_key(seg['start'])
                    end_key = pt_key(seg['end'])
                    graph[start_key].append(end_key)
                    graph[end_key].append(start_key)

                # Find exterior perimeter by walking along outer boundary
                # Start from leftmost-bottommost point (for right-hand rule going DOWN first)
                all_points = list(graph.keys())
                start_pt = min(all_points, key=lambda p: (p[0], -p[1]))  # Left, then bottom

                # Trace perimeter using right-hand rule (exterior on right)
                # Starting from left-bottom and going clockwise
                perimeter = [start_pt]
                visited_edges = set()
                current = start_pt
                prev_direction = np.pi / 2  # Start going down (positive Y)

                max_iterations = len(wall_segments) * 3
                iterations = 0

                while iterations < max_iterations:
                    iterations += 1
                    neighbors = graph[current]

                    if len(neighbors) == 0:
                        break

                    # Choose next point: prefer rightmost turn (exterior boundary)
                    next_pt = None
                    if prev_direction is not None:
                        # Calculate angles to all neighbors
                        angles = []
                        for neighbor in neighbors:
                            edge = tuple(sorted([current, neighbor]))
                            if edge in visited_edges:
                                continue

                            # Direction to neighbor
                            dx = neighbor[0] - current[0]
                            dy = neighbor[1] - current[1]
                            angle = np.arctan2(dy, dx)

                            # Relative angle from previous direction
                            relative_angle = (angle - prev_direction + 2*np.pi) % (2*np.pi)
                            angles.append((relative_angle, neighbor, edge))

                        if angles:
                            # Pick rightmost turn (smallest positive angle)
                            angles.sort()
                            _, next_pt, next_edge = angles[0]
                            visited_edges.add(next_edge)

                    if next_pt is None:
                        # First step or no unvisited neighbors with angle preference
                        for neighbor in neighbors:
                            edge = tuple(sorted([current, neighbor]))
                            if edge not in visited_edges:
                                next_pt = neighbor
                                visited_edges.add(edge)
                                break

                    if next_pt is None:
                        break  # No more unvisited edges

                    if next_pt == start_pt and len(perimeter) > 2:
                        break  # Completed loop

                    # Update direction for next iteration
                    prev_direction = np.arctan2(next_pt[1] - current[1], next_pt[0] - current[0])

                    perimeter.append(next_pt)
                    current = next_pt

                perimeter_points = np.array(perimeter)

            # Common code for all algorithms: draw the perimeter
            perimeter_x = np.append(perimeter_points[:, 0], perimeter_points[0, 0])
            perimeter_y = np.append(perimeter_points[:, 1], perimeter_points[0, 1])

            # Estimate wall thickness from wall segments
            wall_thickness = 10  # Default in pixels

            # Better estimate: measure distance between parallel wall segments
            for i, seg1 in enumerate(wall_segments[:20]):  # Sample first 20
                p1_start = np.array(seg1['start'])
                p1_end = np.array(seg1['end'])
                vec1 = p1_end - p1_start
                len1 = np.linalg.norm(vec1)

                if len1 < 10:
                    continue

                vec1_norm = vec1 / len1

                # Find parallel segments
                for seg2 in wall_segments[i+1:]:
                    p2_start = np.array(seg2['start'])
                    p2_end = np.array(seg2['end'])
                    vec2 = p2_end - p2_start
                    len2 = np.linalg.norm(vec2)

                    if len2 < 10:
                        continue

                    vec2_norm = vec2 / len2

                    # Check if parallel (dot product ~ ±1)
                    dot = abs(np.dot(vec1_norm, vec2_norm))
                    if dot > 0.95:  # Parallel
                        # Measure perpendicular distance
                        to_seg2 = p2_start - p1_start
                        dist_to_seg2 = np.linalg.norm(to_seg2)

                        if dist_to_seg2 > 0:
                            to_seg2_norm = to_seg2 / dist_to_seg2
                            # Cross product for 2D vectors
                            perp_dist = abs(vec1_norm[0] * to_seg2_norm[1] - vec1_norm[1] * to_seg2_norm[0]) * dist_to_seg2

                            if 5 < perp_dist < 50:  # Reasonable wall thickness
                                wall_thickness = max(wall_thickness, perp_dist)
                                break

            wall_thickness = min(wall_thickness, 20)  # Cap at 20 pixels

            # Make line 3x thinner
            display_thickness = wall_thickness / 3

            # Draw perimeter line
            ax7.plot(perimeter_x, perimeter_y, color='darkblue',
                    linewidth=display_thickness, alpha=0.8,
                    linestyle='-', solid_capstyle='round', solid_joinstyle='round',
                    label=f'Unified Exterior Perimeter', zorder=5)

            ax7.legend(loc='upper right', fontsize=10)

        except Exception as e:
            print(f"   Warning: Could not create perimeter: {e}")

    ax7.set_title(f'7. FINAL - Exterior Perimeter\n({len(wall_segments)} walls, full view)',
                 fontsize=14, fontweight='bold', color='blue')
    ax7.axis('off')

    # 8. Hatching Texture Detection
    ax8 = plt.subplot(2, 4, 8)
    # Show hatching density as heatmap
    im = ax8.imshow(hatching_density, cmap='hot', vmin=0, vmax=1)
    
    # Draw pillar bboxes and numbers for debug
    for i, p_str in enumerate(pillar_debug, 1):
        # Parse string: (x,y,w x h,area=...,aspect=...)
        parts = p_str.strip('()').split(',')
        x = int(parts[0])
        y = int(parts[1])
        wh = parts[2].split('x')
        w = int(wh[0])
        h = int(wh[1])
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='white', facecolor='none', alpha=0.7)
        ax8.add_patch(rect)
        ax8.text(x + w/2, y - 5, f'P{i}', fontsize=8, color='white', ha='center', fontweight='bold')
    
    ax8.set_title('8. Hatching Density Map\n(Wall Texture Detection)', fontsize=14, fontweight='bold')
    ax8.axis('off')

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Density')

    # Overall title
    plt.suptitle('Floor Plan Detection Analysis - Method Breakdown',
                fontsize=18, fontweight='bold', y=0.99)

    plt.tight_layout(pad=0.05, h_pad=0.05, w_pad=0.05)

    # Save with reduced DPI to avoid exceeding max size
    output_path = f'plan_floor1_COMPLETE_analysis{output_suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

    plt.close()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Windows:   {len(windows)}")
    print(f"  Doors:     {len(doors)}")
    print(f"  Walls:     {len(wall_segments)} ({len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL)")
    print(f"  Junctions: {len(all_junctions)}")
    print(f"\nOutput: {output_path}")

if __name__ == '__main__':
    import sys

    # v11: Debug pillar detection with relaxed thresholds
    main(perimeter_algorithm='gift_wrapping', output_suffix='_v12_final_pillars')
