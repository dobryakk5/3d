#!/usr/bin/env python
"""
Inference script for floorplan analysis
Detects rooms, doors, windows and other architectural elements
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from floortrans.models import get_model
from floortrans.loaders.augmentations import RotateNTurns
import argparse

# Class definitions
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

def load_model(weights_path, n_classes=44):
    """Load pretrained model"""
    model = get_model('hg_furukawa_original', 51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Use CUDA if available
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    return model

def preprocess_image(image_path, max_size=1024):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')

    # Resize if too large
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h}")

    img_np = np.array(img, dtype=np.float32) / 255.0

    # Normalize to [-1, 1]
    img_np = (img_np - 0.5) * 2

    # Convert to torch tensor [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img_np

def predict_with_rotation(model, image_tensor, n_classes=44, use_rotation=False):
    """Run inference with rotation averaging"""
    rot = RotateNTurns()

    height = image_tensor.shape[2]
    width = image_tensor.shape[3]

    # Test-time augmentation with rotations (disabled by default for speed)
    if use_rotation:
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    else:
        rotations = [(0, 0)]  # No rotation for faster inference

    pred_count = len(rotations)
    prediction = torch.zeros([pred_count, n_classes, height, width])

    with torch.no_grad():
        for i, r in enumerate(rotations):
            forward, back = r

            # Rotate image
            if torch.cuda.is_available():
                rot_image = rot(image_tensor.cuda(), 'tensor', forward)
            else:
                rot_image = rot(image_tensor, 'tensor', forward)

            # Predict
            pred = model(rot_image)

            # Rotate prediction back
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)

            # Resize to original size
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)

            prediction[i] = pred[0].cpu()

    # Average predictions
    prediction = torch.mean(prediction, 0, True)

    return prediction

def visualize_results(image_np, prediction, output_path=None):
    """Visualize segmentation results"""
    split = [21, 12, 11]

    # Split prediction
    heatmaps = prediction[0, :21].cpu().data.numpy()
    rooms_logits = prediction[0, 21:21+12]
    icons_logits = prediction[0, 21+12:]

    # Get class predictions
    rooms_pred = F.softmax(rooms_logits, 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)

    icons_pred = F.softmax(icons_logits, 0).cpu().data.numpy()
    icons_pred = np.argmax(icons_pred, axis=0)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Original image
    img_show = (image_np + 1) / 2  # Denormalize
    axes[0, 0].imshow(img_show)
    axes[0, 0].set_title('Original Image', fontsize=16)
    axes[0, 0].axis('off')

    # Room segmentation
    im1 = axes[0, 1].imshow(rooms_pred, cmap='tab20', vmin=0, vmax=11)
    axes[0, 1].set_title('Room Segmentation', fontsize=16)
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], ticks=np.arange(12), fraction=0.046)
    cbar1.ax.set_yticklabels(room_classes, fontsize=10)

    # Icon segmentation
    im2 = axes[1, 0].imshow(icons_pred, cmap='tab20', vmin=0, vmax=10)
    axes[1, 0].set_title('Icons (Doors, Windows, etc.)', fontsize=16)
    axes[1, 0].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], ticks=np.arange(11), fraction=0.046)
    cbar2.ax.set_yticklabels(icon_classes, fontsize=10)

    # Combined heatmap (sum of first 5 channels for junctions)
    heatmap_combined = np.sum(heatmaps[:5], axis=0)
    axes[1, 1].imshow(img_show)
    im3 = axes[1, 1].imshow(heatmap_combined, cmap='hot', alpha=0.6)
    axes[1, 1].set_title('Wall Junctions Heatmap', fontsize=16)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.close('all')  # Close without showing

    # Print statistics
    print("\n=== Detection Statistics ===")
    print(f"\nRoom types detected:")
    for i, room_name in enumerate(room_classes):
        count = np.sum(rooms_pred == i)
        if count > 100:  # Only show if significant area
            percentage = 100 * count / rooms_pred.size
            print(f"  {room_name}: {percentage:.2f}%")

    print(f"\nIcons detected:")
    for i, icon_name in enumerate(icon_classes):
        count = np.sum(icons_pred == i)
        if count > 50:  # Only show if detected
            percentage = 100 * count / icons_pred.size
            print(f"  {icon_name}: {percentage:.2f}%")

    return rooms_pred, icons_pred, heatmaps

def main():
    parser = argparse.ArgumentParser(description='Floorplan inference')
    parser.add_argument('--image', type=str, required=True, help='Input floorplan image')
    parser.add_argument('--weights', type=str, default='model_best_val_loss_var.pkl',
                        help='Model weights')
    parser.add_argument('--output', type=str, default=None,
                        help='Output visualization path')
    parser.add_argument('--max-size', type=int, default=1024,
                        help='Max image dimension for processing')
    parser.add_argument('--use-rotation', action='store_true',
                        help='Use rotation averaging (slower but better)')
    args = parser.parse_args()

    print(f"Loading model from {args.weights}...")
    model = load_model(args.weights)

    print(f"Processing image {args.image}...")
    image_tensor, image_np = preprocess_image(args.image, max_size=args.max_size)

    print("Running inference...")
    prediction = predict_with_rotation(model, image_tensor, use_rotation=args.use_rotation)

    print("Visualizing results...")
    output_path = args.output or args.image.replace('.jpg', '_result.png').replace('.png', '_result.png')
    visualize_results(image_np, prediction, output_path)

    print("Done!")

if __name__ == '__main__':
    main()
