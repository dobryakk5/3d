#!/usr/bin/env python3
"""
Export floor plan objects to JSON for Blender import:
1. Walls with thickness and height
2. Windows and doors with wall references
3. Pillars (standalone, not intersecting with walls)
4. Rooms defined by wall boundaries
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import json
import datetime
from scipy import ndimage
from scipy.spatial import Delaunay
import sys
sys.path.append('floortrans')
from floortrans.models import get_model
import svgwrite

# Import detection functions from existing scripts
exec(open('cubicasa_vectorize.py').read().split('if __name__')[0])

def categorize_junctions(junctions_dict):
    """Collect all junctions without categorization"""
    all_junctions = []
    for jtype, points in junctions_dict.items():
        for p in points:
            all_junctions.append({**p, 'type': jtype})
    return all_junctions

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

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def find_wall_for_opening(opening, wall_segments):
    """Find which wall segment an opening belongs to"""
    x, y, w, h = opening['bbox']
    center_x, center_y = x + w/2, y + h/2
    
    best_wall = None
    min_distance = float('inf')
    
    for i, seg in enumerate(wall_segments):
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        
        # Calculate distance from opening center to wall line
        dist = point_to_line_distance(center_x, center_y, x1, y1, x2, y2)
        
        # Check if opening is within wall bounds (with some tolerance)
        wall_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if wall_length > 0:
            # Project opening center onto wall line
            t = max(0, min(1, ((center_x - x1) * (x2 - x1) + (center_y - y1) * (y2 - y1)) / (wall_length**2)))
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            
            # Check if projection is within opening bounds
            if (abs(proj_x - center_x) <= w/2 + 10 and 
                abs(proj_y - center_y) <= h/2 + 10 and
                dist < min_distance):
                min_distance = dist
                best_wall = i
    
    return best_wall

def detect_pillars_for_export(image, wall_mask, wall_segments, min_area=100, max_area=60000):
    """Detect pillars with wall hatching texture (standalone only)"""
    # Find components in hatching mask
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
    
    # Process all components with shape and size filtering
    for i in range(1, num + 1):
        comp = (labeled == i)
        area = comp.sum()

        if min_area < area < max_area:  # Size filter
            # Get bounding box
            y_coords, x_coords = np.where(comp)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            aspect = max(w, h) / max(min(w, h), 1)

            # Check if pillar is enclosed by walls (standalone pillars only)
            bbox = (x_min, y_min, w, h)
            if is_enclosed_by_walls(bbox, wall_segments, tolerance=50):
                print(f"      Component at ({x_min},{y_min},{w}x{h}): REJECTED - enclosed by wall rectangle")
                continue
            
            # Adaptive shape filtering based on size
            if area < 500:  # Small components
                max_aspect = 10.0
                min_aspect = 0.2
            elif area < 2000:  # Medium components
                max_aspect = 6.0
                min_aspect = 0.25
            else:  # Large components
                max_aspect = 4.0
                min_aspect = 0.3
            
            # Exclude too elongated components
            if aspect > max_aspect:
                print(f"      Component at ({x_min},{y_min},{w}x{h}): REJECTED - aspect ratio too high ({aspect:.1f} > {max_aspect})")
                continue
            
            # Check for square-like shape (typical for pillars)
            if aspect < min_aspect:
                print(f"      Component at ({x_min},{y_min},{w}x{h}): REJECTED - unusual shape (aspect={aspect:.1f} < {min_aspect})")
                continue
            
            # Check for diagonal texture
            crop = wall_mask[y_min:y_max+1, x_min:x_max+1]
            
            def has_diagonal_texture(crop_img, angles=[45, -45, 135, -135], min_ratio=0.15):
                """Check if component has diagonal hatching texture"""
                total_pixels = crop_img.sum() / 255
                if total_pixels == 0:
                    return False
                
                # Adaptive thresholds based on component size
                if area < 500:
                    min_fill_ratio, max_fill_ratio = 0.2, 0.95
                    min_texture_ratio = 0.08
                elif area < 2000:
                    min_fill_ratio, max_fill_ratio = 0.15, 0.9
                    min_texture_ratio = 0.12
                else:
                    min_fill_ratio, max_fill_ratio = 0.1, 0.8
                    min_texture_ratio = 0.15
                
                # Check fill ratio
                fill_ratio = total_pixels / (w * h)
                if fill_ratio < min_fill_ratio or fill_ratio > max_fill_ratio:
                    return False
                
                # Check for diagonal lines
                for angle in angles:
                    kernel = np.zeros((11, 11), dtype=np.uint8)
                    center = 5
                    angle_rad = np.deg2rad(angle)
                    for i in range(11):
                        offset = i - center
                        x = int(center + offset * np.cos(angle_rad))
                        y = int(center + offset * np.sin(angle_rad))
                        if 0 <= x < 11 and 0 <= y < 11:
                            kernel[y, x] = 1
                    
                    detected = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel, iterations=1)
                    detected_pixels = detected.sum() / 255
                    
                    ratio = detected_pixels / total_pixels if total_pixels > 0 else 0
                    if detected_pixels >= min_texture_ratio * total_pixels:
                        return True
                
                return False
            
            has_texture = has_diagonal_texture(crop)
            if has_texture:
                pillars.append({
                    'x': x_min,
                    'y': y_min,
                    'width': w,
                    'height': h,
                    'area': area,
                    'aspect_ratio': aspect
                })
    
    return pillars

def extract_rooms_from_walls(wall_segments):
    """Extract room polygons from wall segments"""
    if not wall_segments:
        return []
    
    # Collect all wall endpoints
    all_points = []
    for seg in wall_segments:
        all_points.append(seg['start'])
        all_points.append(seg['end'])
    
    all_points = np.array(all_points)
    
    if len(all_points) < 3:
        return []
    
    # Create Delaunay triangulation
    try:
        tri = Delaunay(all_points)
    except:
        return []
    
    # Build edge set with length threshold
    alpha = 150  # Threshold distance
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx = simplex[i]
            p2_idx = simplex[(i+1)%3]
            p1 = all_points[p1_idx]
            p2 = all_points[p2_idx]
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
        start_idx = min(graph.keys(), key=lambda i: (all_points[i][0], all_points[i][1]))
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

        perimeter_points = all_points[perimeter_indices]
        
        # Create room from perimeter
        room = {
            'id': 'room_1',
            'vertices': [{'x': float(p[0]), 'y': float(p[1])} for p in perimeter_points],
            'wall_ids': [f'wall_{i+1}' for i in range(len(wall_segments))]
        }
        
        return [room]
    
    return []

def estimate_wall_thickness(wall_segments):
    """Estimate wall thickness from parallel segments"""
    if len(wall_segments) < 2:
        return 10  # Default thickness in pixels
    
    thickness_samples = []
    
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
                        thickness_samples.append(perp_dist)

    if thickness_samples:
        median_thickness = np.median(thickness_samples)
        return min(median_thickness, 20)  # Cap at 20 pixels
    
    return 10  # Default thickness

def create_colored_svg(output_path, image_shape, walls, doors, windows, pillars, rooms, scale_factor):
    """Create colored SVG file with all detected objects"""
    height, width = image_shape[:2]
    
    # Scale back to original resolution
    orig_width = int(width * scale_factor)
    orig_height = int(height * scale_factor)
    
    dwg = svgwrite.Drawing(output_path, size=(orig_width, orig_height), profile='tiny')
    
    # Define colors for different object types
    colors = {
        'wall_hatching': '#8B4513',  # Brown for hatching walls
        'wall_dl': '#4169E1',         # Royal blue for DL walls
        'window': '#00CED1',          # Dark turquoise for windows
        'door': '#228B22',            # Forest green for doors
        'pillar': '#FF6347',          # Tomato red for pillars
        'room': '#F0F8FF',            # Alice blue (light) for rooms
        'room_border': '#708090'      # Slate gray for room borders
    }
    
    # Draw rooms (background)
    if rooms:
        room_group = dwg.add(dwg.g(id='rooms'))
        for room in rooms:
            points = [(float(v['x'] * scale_factor), float(v['y'] * scale_factor)) for v in room['vertices']]
            if len(points) >= 3:
                room_group.add(dwg.polygon(
                    points,
                    fill=colors['room'],
                    stroke=colors['room_border'],
                    stroke_width=2,
                    opacity=0.3
                ))
    
    # Draw walls
    wall_group = dwg.add(dwg.g(id='walls'))
    for wall in walls:
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        color = colors['wall_hatching'] if wall['source'] == 'hatching' else colors['wall_dl']
        wall_group.add(dwg.line(
            start=(float(x1 * scale_factor), float(y1 * scale_factor)),
            end=(float(x2 * scale_factor), float(y2 * scale_factor)),
            stroke=color,
            stroke_width=int(6 * scale_factor),
            stroke_linecap='round'
        ))
    
    # Draw pillars
    pillar_group = dwg.add(dwg.g(id='pillars'))
    for i, pillar in enumerate(pillars, 1):
        x, y, w, h = pillar['x'], pillar['y'], pillar['width'], pillar['height']
        pillar_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill=colors['pillar'],
            fill_opacity=0.7,
            stroke='darkred',
            stroke_width=int(2 * scale_factor)
        ))
        # Add label
        pillar_group.add(dwg.text(
            f'P{i}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='white',
            font_size=int(min(w, h) * 0.4 * scale_factor),
            font_weight='bold'
        ))
    
    # Draw windows
    window_group = dwg.add(dwg.g(id='windows'))
    for i, window in enumerate(windows, 1):
        x, y, w, h = window['bbox']
        window_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill=colors['window'],
            fill_opacity=0.7,
            stroke='darkcyan',
            stroke_width=int(3 * scale_factor)
        ))
        # Add label
        window_group.add(dwg.text(
            f'W{i}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='black',
            font_size=int(min(w, h) * 0.5 * scale_factor),
            font_weight='bold'
        ))
    
    # Draw doors
    door_group = dwg.add(dwg.g(id='doors'))
    for i, door in enumerate(doors, 1):
        x, y, w, h = door['bbox']
        door_group.add(dwg.rect(
            insert=(float(x * scale_factor), float(y * scale_factor)),
            size=(float(w * scale_factor), float(h * scale_factor)),
            fill=colors['door'],
            fill_opacity=0.7,
            stroke='darkgreen',
            stroke_width=int(3 * scale_factor)
        ))
        # Add label
        door_group.add(dwg.text(
            f'D{i}',
            insert=(float((x + w/2) * scale_factor), float((y + h/2) * scale_factor)),
            text_anchor='middle',
            fill='white',
            font_size=int(min(w, h) * 0.5 * scale_factor),
            font_weight='bold'
        ))
    
    # Add legend
    legend_group = dwg.add(dwg.g(id='legend', font_size=int(14 * scale_factor)))
    legend_x = int(20 * scale_factor)
    legend_y = int(30 * scale_factor)
    line_height = int(25 * scale_factor)
    
    legend_items = [
        ('Walls (Hatching)', colors['wall_hatching']),
        ('Walls (DL)', colors['wall_dl']),
        ('Windows', colors['window']),
        ('Doors', colors['door']),
        ('Pillars', colors['pillar']),
        ('Rooms', colors['room'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + i * line_height
        
        # Draw color box
        legend_group.add(dwg.rect(
            insert=(legend_x, y_pos - int(10 * scale_factor)),
            size=(int(20 * scale_factor), int(15 * scale_factor)),
            fill=color,
            stroke='black',
            stroke_width=1
        ))
        
        # Add label
        legend_group.add(dwg.text(
            label,
            insert=(legend_x + int(30 * scale_factor), y_pos),
            fill='black',
            font_family='Arial'
        ))
    
    # Add title
    title_group = dwg.add(dwg.g(id='title'))
    title_group.add(dwg.text(
        'Floor Plan Detection Results',
        insert=(orig_width // 2, int(30 * scale_factor)),
        text_anchor='middle',
        fill='black',
        font_size=int(20 * scale_factor),
        font_weight='bold',
        font_family='Arial'
    ))
    
    # Add statistics
    stats_group = dwg.add(dwg.g(id='statistics', font_size=int(12 * scale_factor)))
    stats_text = [
        f"Walls: {len(walls)}",
        f"Windows: {len(windows)}",
        f"Doors: {len(doors)}",
        f"Pillars: {len(pillars)}",
        f"Rooms: {len(rooms)}"
    ]
    
    for i, text in enumerate(stats_text):
        stats_group.add(dwg.text(
            text,
            insert=(orig_width - int(150 * scale_factor), legend_y + i * line_height),
            fill='black',
            font_family='Arial'
        ))
    
    dwg.save()
    return orig_width, orig_height

def main():
    """Main export function"""
    print("="*80)
    print("EXPORTING FLOOR PLAN OBJECTS TO JSON")
    print("="*80)

    image_path = 'plan_floor1.jpg'
    output_path = 'plan_floor1_objects.json'

    # Load model
    print("\n[1/6] Loading model...")
    try:
        model = get_model('hg_furukawa_original', 51)
        model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
        checkpoint = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print("   Model loaded successfully")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return

    # Preprocess
    print("\n[2/6] Preprocessing...")
    try:
        img_orig = Image.open(image_path).convert('RGB')
        orig_width, orig_height = img_orig.size

        max_size = 2048
        w, h = orig_width, orig_height
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            img = img_orig
            scale = 1.0

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img_np = np.array(img)

        img_normalized = (img_np.astype(np.float32) / 255.0 - 0.5) * 2
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        print(f"   Image processed: {img_np.shape[1]}x{img_np.shape[0]} (scale: {scale:.2f})")
    except Exception as e:
        print(f"   Error preprocessing image: {e}")
        return

    # Inference
    print("\n[3/6] Running DL inference...")
    try:
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
        
        print(f"   Raw detections: {len(doors)} doors, {len(windows)} windows")
    except Exception as e:
        print(f"   Error during inference: {e}")
        return

    # Detect walls
    print("\n[4/6] Detecting walls...")
    try:
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
        
        # Filter isolated wall segments
        def filter_connected_walls(wall_segments, min_component_size=5):
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

            # Keep components that meet size threshold
            valid_segments = []
            if components:
                for idx, comp in enumerate(components):
                    if len(comp) >= min_component_size:
                        valid_segments.extend([wall_segments[i] for i in comp])

            return valid_segments if valid_segments else wall_segments

        wall_segments = filter_connected_walls(wall_segments, min_component_size=5)
        
        print(f"   Detected: {len(wall_segments)} wall segments ({len(wall_segments_hatching)} hatching + {len(wall_segments_dl)} DL)")
    except Exception as e:
        print(f"   Error detecting walls: {e}")
        return

    # Filter windows/doors
    print("\n[5/6] Processing openings...")
    try:
        # Remove nested boxes
        windows = remove_nested_boxes(windows, iou_threshold=0.15)
        doors = remove_nested_boxes(doors, iou_threshold=0.5)

        # Filter windows: edges must be on walls
        def filter_windows_strict(windows, wall_segments, max_distance=15):
            """Keep only windows whose edges are on walls"""
            valid = []
            for w in windows:
                x, y, width, height = w['bbox']

                # Check only edge midpoints
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
                    valid_doors.append(door)

            return valid_doors

        doors_before = len(doors)
        doors = filter_door_vs_window(doors, door_prob, window_prob)
        print(f"   After filtering: {len(windows)} windows, {len(doors)}/{doors_before} doors")

        # Extract junctions
        heatmaps = prediction[0, :21].cpu().data.numpy()
        junctions_dict = extract_junctions(prediction, threshold=threshold)
        all_junctions = categorize_junctions(junctions_dict)

        # Analyze detection methods
        for w in windows:
            w['methods'] = analyze_detection_method(w, wall_segments, all_junctions)
        for d in doors:
            d['methods'] = analyze_detection_method(d, wall_segments, all_junctions)

        # Detect pillars (standalone, not enclosed by walls)
        wall_mask = detect_hatching(img_np, kernel_size=15, density_threshold=0.15)
        pillars = detect_pillars_for_export(img_np, wall_mask, wall_segments, min_area=50, max_area=60000)
        
        print(f"   Detected: {len(pillars)} pillars, {len(all_junctions)} junctions")
    except Exception as e:
        print(f"   Error processing openings: {e}")
        return

    # Create JSON structure
    print("\n[6/6] Creating JSON structure...")
    try:
        # Estimate wall thickness
        wall_thickness_pixels = estimate_wall_thickness(wall_segments)
        wall_thickness_meters = 0.2  # Standard wall thickness in meters
        
        # Create JSON structure
        json_data = {
            "metadata": {
                "source_image": image_path,
                "scale_factor": float(scale),
                "units": "pixels",
                "wall_height": 3.0,
                "wall_thickness": wall_thickness_meters,
                "door_height": 2.1,
                "window_height": 1.5,
                "window_sill_height": 1.0,
                "timestamp": datetime.datetime.now().isoformat(),
                "model_info": {
                    "name": "hg_furukawa_original",
                    "weights": "model_best_val_loss_var.pkl"
                }
            },
            "walls": [],
            "openings": [],
            "pillars": [],
            "rooms": [],
            "junctions": [],
            "statistics": {
                "walls": 0,
                "windows": 0,
                "doors": 0,
                "pillars": 0,
                "rooms": 0,
                "junctions": 0
            }
        }

        # Export walls
        for i, seg in enumerate(wall_segments):
            wall_data = {
                "id": f"wall_{i+1}",
                "start": {"x": float(seg['start'][0]), "y": float(seg['start'][1])},
                "end": {"x": float(seg['end'][0]), "y": float(seg['end'][1])},
                "thickness": wall_thickness_meters,
                "height": 3.0,
                "source": seg['source']
            }
            json_data["walls"].append(wall_data)

        # Export windows
        for i, window in enumerate(windows):
            x, y, w, h = window['bbox']
            wall_id = find_wall_for_opening(window, wall_segments)
            
            window_data = {
                "id": f"window_{i+1}",
                "type": "window",
                "bbox": {"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                "wall_id": f"wall_{wall_id+1}" if wall_id is not None else None,
                "height": 1.5,
                "sill_height": 1.0,
                "methods": window['methods']
            }
            
            # Add confidence values
            cx, cy = int(x + w/2), int(y + h/2)
            if 0 <= cy < window_prob.shape[0] and 0 <= cx < window_prob.shape[1]:
                window_data["confidence"] = {
                    "window_prob": float(window_prob[cy, cx]),
                    "door_prob": float(door_prob[cy, cx])
                }
            
            json_data["openings"].append(window_data)

        # Export doors
        for i, door in enumerate(doors):
            x, y, w, h = door['bbox']
            wall_id = find_wall_for_opening(door, wall_segments)
            
            door_data = {
                "id": f"door_{i+1}",
                "type": "door",
                "bbox": {"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                "wall_id": f"wall_{wall_id+1}" if wall_id is not None else None,
                "height": 2.1,
                "methods": door['methods']
            }
            
            # Add confidence values
            cx, cy = int(x + w/2), int(y + h/2)
            if 0 <= cy < door_prob.shape[0] and 0 <= cx < door_prob.shape[1]:
                door_data["confidence"] = {
                    "door_prob": float(door_prob[cy, cx]),
                    "window_prob": float(window_prob[cy, cx])
                }
            
            json_data["openings"].append(door_data)

        # Export pillars
        for i, pillar in enumerate(pillars):
            pillar_data = {
                "id": f"pillar_{i+1}",
                "bbox": {
                    "x": float(pillar['x']),
                    "y": float(pillar['y']),
                    "width": float(pillar['width']),
                    "height": float(pillar['height'])
                },
                "height": 3.0,
                "area": float(pillar['area']),
                "aspect_ratio": float(pillar['aspect_ratio'])
            }
            json_data["pillars"].append(pillar_data)

        # Export junctions
        for junction in all_junctions:
            junction_data = {
                "x": float(junction['x']),
                "y": float(junction['y']),
                "type": junction['type']
            }
            json_data["junctions"].append(junction_data)

        # Extract rooms
        rooms = extract_rooms_from_walls(wall_segments)
        for room in rooms:
            json_data["rooms"].append(room)

        # Update statistics
        json_data["statistics"]["walls"] = len(json_data["walls"])
        json_data["statistics"]["windows"] = len([o for o in json_data["openings"] if o["type"] == "window"])
        json_data["statistics"]["doors"] = len([o for o in json_data["openings"] if o["type"] == "door"])
        json_data["statistics"]["pillars"] = len(json_data["pillars"])
        json_data["statistics"]["rooms"] = len(json_data["rooms"])
        json_data["statistics"]["junctions"] = len(json_data["junctions"])

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"   JSON saved to: {output_path}")
        
        # Create colored SVG visualization
        print("\n   Creating SVG visualization...")
        svg_output_path = output_path.replace('.json', '_colored.svg')
        try:
            svg_w, svg_h = create_colored_svg(
                svg_output_path,
                img_np.shape,
                wall_segments,
                doors,
                windows,
                pillars,
                rooms,
                scale
            )
            print(f"   SVG saved to: {svg_output_path} ({svg_w}x{svg_h})")
        except Exception as e:
            print(f"   Error creating SVG: {e}")

    except Exception as e:
        print(f"   Error creating JSON: {e}")
        return

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Walls:   {len(json_data['walls'])}")
    print(f"  Windows: {len([o for o in json_data['openings'] if o['type'] == 'window'])}")
    print(f"  Doors:   {len([o for o in json_data['openings'] if o['type'] == 'door'])}")
    print(f"  Pillars: {len(json_data['pillars'])}")
    print(f"  Rooms:   {len(json_data['rooms'])}")
    print(f"\nOutput:")
    print(f"  JSON: {output_path}")
    print(f"  SVG:  {output_path.replace('.json', '_colored.svg')}")

if __name__ == '__main__':
    main()