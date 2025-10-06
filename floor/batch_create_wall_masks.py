#!/usr/bin/env python
"""
Batch processing: automatically create wall masks for multiple floorplans
Finds hatching patterns and generates training masks
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import json

def create_line_kernel(length, angle_deg):
    """Create kernel for line detection at specific angle"""
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

def detect_walls_by_hatching(image_path, max_size=2048):
    """
    Автоматически найти стены по штриховке
    Returns: wall_mask (numpy array), metadata (dict)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size

    # Resize if needed
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img_np = np.array(img)
    scale_factor = orig_size[0] / img.width

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Binarize
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect diagonal lines (hatching)
    hatching_masks = []
    angles = [45, -45, 135, -135]

    for angle in angles:
        kernel = create_line_kernel(15, angle)
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        hatching_masks.append(detected)

    # Combine all directions
    hatching_combined = np.zeros_like(binary)
    for mask in hatching_masks:
        hatching_combined = cv2.bitwise_or(hatching_combined, mask)

    # Dilate to connect lines
    kernel_dilate = np.ones((5, 5), np.uint8)
    hatching_dilated = cv2.dilate(hatching_combined, kernel_dilate, iterations=2)

    # Calculate density of hatching
    kernel_density = np.ones((20, 20), np.uint8)
    density = cv2.filter2D(hatching_dilated.astype(np.float32), -1, kernel_density)
    density = (density / density.max() * 255).astype(np.uint8)

    # Threshold - where many lines = wall
    threshold = 30
    wall_mask = (density > threshold).astype(np.uint8) * 255

    # Refine boundaries
    kernel_close = np.ones((7, 7), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_close)

    # Fill holes
    kernel_fill = np.ones((15, 15), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_fill)

    # Calculate statistics
    wall_pixels = np.sum(wall_mask > 0)
    total_pixels = wall_mask.size
    coverage = wall_pixels / total_pixels * 100

    # Count segments
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_segments = sum(1 for c in contours if cv2.contourArea(c) > 100)

    metadata = {
        'original_size': orig_size,
        'processed_size': (img.width, img.height),
        'scale_factor': float(scale_factor),
        'wall_pixels': int(wall_pixels),
        'coverage_percent': float(coverage),
        'num_segments': num_segments
    }

    return wall_mask, metadata

def process_batch(input_dir, output_dir, max_size=2048):
    """
    Обработать все планы в папке
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    masks_dir = output_path / 'masks'
    masks_dir.mkdir(exist_ok=True)

    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(list(input_path.glob(f'*{ext}')))

    if len(images) == 0:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"Found {len(images)} images to process")
    print(f"Output directory: {output_path}")

    results = []

    # Process each image
    for img_path in tqdm(images, desc="Processing floorplans"):
        try:
            # Detect walls
            wall_mask, metadata = detect_walls_by_hatching(str(img_path), max_size)

            # Save mask
            stem = img_path.stem
            mask_path = masks_dir / f'{stem}_wall.png'
            cv2.imwrite(str(mask_path), wall_mask)

            # Add to results
            result = {
                'image': img_path.name,
                'mask': str(mask_path.relative_to(output_path)),
                'metadata': metadata
            }
            results.append(result)

            print(f"  ✓ {img_path.name}: {metadata['num_segments']} segments, {metadata['coverage_percent']:.1f}% coverage")

        except Exception as e:
            print(f"  ✗ {img_path.name}: Error - {e}")
            continue

    # Save summary
    summary = {
        'total_images': len(images),
        'processed': len(results),
        'output_dir': str(output_path),
        'results': results
    }

    summary_path = output_path / 'processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Processed: {len(results)}/{len(images)} images")
    print(f"  Masks saved to: {masks_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}\n")

    # Create dataset structure
    create_dataset_structure(output_path, results)

    return results

def create_dataset_structure(output_path, results):
    """
    Создать структуру датасета для обучения
    """
    dataset = {
        'version': '1.0',
        'description': 'Auto-generated wall masks from hatching detection',
        'num_samples': len(results),
        'samples': []
    }

    for result in results:
        dataset['samples'].append({
            'image': result['image'],
            'wall_mask': result['mask'],
            'stats': result['metadata']
        })

    dataset_path = output_path / 'dataset.json'
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"  Dataset config: {dataset_path}")
    print(f"\n  You can now use this for training!")
    print(f"  Next step: python fine_tune_walls.py --dataset {dataset_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch create wall masks from floorplans'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with floorplan images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='training_data',
        help='Output directory for masks (default: training_data)'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=2048,
        help='Maximum image dimension for processing (default: 2048)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BATCH WALL MASK GENERATION")
    print("=" * 70)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Max size: {args.max_size}px\n")

    process_batch(args.input, args.output, args.max_size)

if __name__ == '__main__':
    main()
