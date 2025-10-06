#!/usr/bin/env python
"""
Simplified inference script for floorplan analysis
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

# Import only what we need
from floortrans.models import get_model

# Class definitions
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

def main():
    print("=" * 60)
    print("FLOORPLAN ANALYSIS - SIMPLE INFERENCE")
    print("=" * 60)

    # Configuration
    image_path = 'plan_floor1.jpg'
    weights_path = 'model_best_val_loss_var.pkl'
    output_path = 'plan_floor1_result.png'
    max_size = 512
    n_classes = 44

    print(f"\n1. Loading model from {weights_path}...")
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("   Model loaded successfully!")

    print(f"\n2. Loading and preprocessing image {image_path}...")
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size
    print(f"   Original size: {orig_size}")

    # Resize if needed
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        print(f"   Resized to: {img.size}")

    # Convert to tensor
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - 0.5) * 2  # Normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    print(f"\n3. Running inference...")
    with torch.no_grad():
        prediction = model(img_tensor)
        print(f"   Output shape: {prediction.shape}")

    # Split prediction
    heatmaps = prediction[0, :21].cpu().data.numpy()
    rooms_logits = prediction[0, 21:33]
    icons_logits = prediction[0, 33:44]

    # Get class predictions
    rooms_pred = F.softmax(rooms_logits, 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)

    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()
    icons_pred = np.argmax(icons_pred, axis=0)

    print(f"\n4. Analyzing results...")

    # Room statistics
    print(f"\n   Room types detected:")
    for i, room_name in enumerate(room_classes):
        count = np.sum(rooms_pred == i)
        if count > 100:  # Only show if significant area
            percentage = 100 * count / rooms_pred.size
            print(f"      {room_name}: {percentage:.2f}%")

    # Icon statistics
    print(f"\n   Icons detected:")
    door_pixels = np.sum(icons_pred == 2)  # Door class
    window_pixels = np.sum(icons_pred == 1)  # Window class

    for i, icon_name in enumerate(icon_classes):
        count = np.sum(icons_pred == i)
        if count > 10:  # Only show if detected
            percentage = 100 * count / icons_pred.size
            print(f"      {icon_name}: {count} pixels ({percentage:.3f}%)")

    print(f"\n5. Creating visualization...")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Original image
    img_show = (img_np + 1) / 2  # Denormalize
    axes[0, 0].imshow(img_show)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Room segmentation
    im1 = axes[0, 1].imshow(rooms_pred, cmap='tab20', vmin=0, vmax=11)
    axes[0, 1].set_title('Room Segmentation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], ticks=np.arange(12), fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels(room_classes, fontsize=9)

    # Icon segmentation (DOORS AND WINDOWS!)
    im2 = axes[1, 0].imshow(icons_pred, cmap='tab20', vmin=0, vmax=10)
    axes[1, 0].set_title(f'Icons: Doors={door_pixels}px, Windows={window_pixels}px',
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], ticks=np.arange(11), fraction=0.046, pad=0.04)
    cbar2.ax.set_yticklabels(icon_classes, fontsize=9)

    # Combined heatmap
    heatmap_combined = np.max(heatmaps[:10], axis=0)
    axes[1, 1].imshow(img_show)
    im3 = axes[1, 1].imshow(heatmap_combined, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title('Junction Heatmap Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")

    # Save individual icon detection
    fig2, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_show)

    # Highlight doors and windows
    door_mask = icons_pred == 2
    window_mask = icons_pred == 1

    overlay = np.zeros((*icons_pred.shape, 4))
    overlay[door_mask] = [1, 0, 0, 0.7]  # Red for doors
    overlay[window_mask] = [0, 0, 1, 0.7]  # Blue for windows

    ax.imshow(overlay)
    ax.set_title(f'Door and Window Detection (Red=Doors, Blue=Windows)', fontsize=14, fontweight='bold')
    ax.axis('off')

    door_window_path = output_path.replace('.png', '_doors_windows.png')
    plt.savefig(door_window_path, dpi=150, bbox_inches='tight')
    print(f"   Saved door/window overlay to: {door_window_path}")

    plt.close('all')

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'=' * 60}\n")

    if door_pixels < 100:
        print("⚠️  WARNING: Very few door pixels detected!")
        print("   This may indicate:")
        print("   - Model not trained on this type of floorplan")
        print("   - Image preprocessing issues")
        print("   - Need for fine-tuning or different approach")

    if window_pixels < 100:
        print("⚠️  WARNING: Very few window pixels detected!")

if __name__ == '__main__':
    main()
