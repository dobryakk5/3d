"""Analyze window texture pattern on the floor plan"""
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load image
img = Image.open('plan_floor1.jpg').convert('RGB')
img_np = np.array(img)

# Window coordinates from detections (raw, unfiltered)
windows = [
    {'id': 1, 'bbox': (1625, 364, 235, 58), 'name': 'W1 - верх центр'},
    {'id': 2, 'bbox': (958, 400, 58, 7), 'name': 'W2 - верх маленькое'},
    {'id': 3, 'bbox': (2616, 1348, 48, 155), 'name': 'W3 - право верх'},
    {'id': 4, 'bbox': (2617, 1610, 48, 150), 'name': 'W4 - право низ'},
    {'id': 5, 'bbox': (521, 1716, 50, 152), 'name': 'W5 - лево'},
    # Predicted large windows
    {'id': 6, 'bbox': (100, 80, 180, 400), 'name': 'W? - лево верх большое'},
    {'id': 7, 'bbox': (300, 550, 400, 150), 'name': 'W? - низ большое'},
]

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for idx, window in enumerate(windows[:9]):
    x, y, w, h = window['bbox']

    # Extract window patch
    patch = img_np[max(0, y):min(img_np.shape[0], y+h),
                   max(0, x):min(img_np.shape[1], x+w)]

    if patch.size == 0:
        continue

    axes[idx].imshow(patch)
    axes[idx].set_title(f"{window['name']}\n{w}x{h} px", fontsize=10)
    axes[idx].axis('off')

    # Draw rectangle on full image
    cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Hide unused subplots
for idx in range(len(windows), 9):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('window_texture_analysis.png', dpi=150, bbox_inches='tight')
print("Saved window_texture_analysis.png")

# Save annotated full image
cv2.imwrite('windows_annotated.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
print("Saved windows_annotated.jpg")

# Print window stats
print("\nWindow Statistics:")
for window in windows:
    x, y, w, h = window['bbox']
    area = w * h
    aspect = w / h if h > 0 else 0
    print(f"{window['name']:30} {w:4}x{h:4} = {area:7} px²  aspect={aspect:.2f}")
