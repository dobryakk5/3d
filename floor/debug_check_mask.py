#!/usr/bin/env python3
"""
Quick debug script to check the hatching mask generated with strict parameters
"""
import cv2
import numpy as np

# Load the enhanced_hatching_strict_mask.png (the reference - what we WANT)
reference_mask = cv2.imread('enhanced_hatching_strict_mask.png', cv2.IMREAD_GRAYSCALE)

print("="*80)
print("REFERENCE MASK (enhanced_hatching_strict_mask.png)")
print("="*80)
print(f"Shape: {reference_mask.shape}")
print(f"Values: min={reference_mask.min()}, max={reference_mask.max()}")
print(f"White pixels: {np.sum(reference_mask > 0)}")

# Try to find contours in the reference mask
contours_ext, _ = cv2.findContours(reference_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_list, _ = cv2.findContours(reference_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nContours (RETR_EXTERNAL): {len(contours_ext)}")
if contours_ext:
    areas_ext = [cv2.contourArea(c) for c in contours_ext]
    areas_ext_sorted = sorted(areas_ext, reverse=True)
    print(f"Top contour areas (RETR_EXTERNAL): {areas_ext_sorted[:10]}")
    print(f"Contours with area > 100: {sum(1 for a in areas_ext if a > 100)}")

print(f"\nContours (RETR_LIST): {len(contours_list)}")
if contours_list:
    areas_list = [cv2.contourArea(c) for c in contours_list]
    areas_list_sorted = sorted(areas_list, reverse=True)
    print(f"Top contour areas (RETR_LIST): {areas_list_sorted[:10]}")
    print(f"Contours with area > 100: {sum(1 for a in areas_list if a > 100)}")

# Create visualization with all contours drawn
debug_vis = cv2.cvtColor(reference_mask, cv2.COLOR_GRAY2BGR)
for i, contour in enumerate(contours_ext[:20]):  # Draw first 20 contours
    area = cv2.contourArea(contour)
    if area > 100:
        color = (0, 255, 0)  # Green for large contours
        cv2.drawContours(debug_vis, [contour], -1, color, 2)
        # Draw contour number
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(debug_vis, f"{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imwrite('debug_contours_visualization.png', debug_vis)
print(f"\nSaved contours visualization to: debug_contours_visualization.png")
