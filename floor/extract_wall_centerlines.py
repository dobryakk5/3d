#!/usr/bin/env python3
"""
Извлечение центральных линий стен из маски штриховки
Правильный подход: скелетизация вместо контуров
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, medial_axis

def extract_wall_centerlines():
    """Извлекает центральные линии стен методом скелетизации"""

    # Загружаем маску стен
    wall_mask = cv2.imread('enhanced_hatching_strict_mask.png', cv2.IMREAD_GRAYSCALE)

    print(f"Mask shape: {wall_mask.shape}")
    print(f"White pixels: {np.sum(wall_mask > 0)}")

    # Бинаризация
    _, binary = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)

    # Метод 1: Морфологическая скелетизация
    skeleton = skeletonize(binary > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    cv2.imwrite('debug_skeleton.png', skeleton_img)
    print(f"Skeleton saved: debug_skeleton.png")
    print(f"Skeleton pixels: {np.sum(skeleton)}")

    # Метод 2: Медиальная ось (более точная центральная линия)
    medial, distance = medial_axis(binary > 0, return_distance=True)
    medial_img = (medial * 255).astype(np.uint8)

    cv2.imwrite('debug_medial_axis.png', medial_img)
    print(f"Medial axis saved: debug_medial_axis.png")
    print(f"Medial pixels: {np.sum(medial)}")

    # Метод 3: Истончение (thinning) - классический алгоритм Zhang-Suen
    # Используем морфологическое утончение OpenCV
    thinned = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    iterations = 0
    while not done:
        eroded = cv2.erode(thinned, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(thinned, temp)
        thinned = cv2.bitwise_or(eroded, temp)

        zeros = cv2.countNonZero(thinned)
        if zeros == 0 or iterations > 100:
            done = True
        iterations += 1

    cv2.imwrite('debug_thinned.png', thinned)
    print(f"Thinned saved: debug_thinned.png (iterations: {iterations})")
    print(f"Thinned pixels: {np.sum(thinned > 0)}")

    # Теперь из скелета извлекаем линии стен
    # Используем HoughLinesP для поиска отрезков
    lines = cv2.HoughLinesP(skeleton_img, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)

    if lines is not None:
        print(f"\nНайдено линий (HoughLinesP): {len(lines)}")

        # Визуализация линий
        line_vis = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite('debug_hough_lines.png', line_vis)
        print(f"Lines visualization saved: debug_hough_lines.png")

    return skeleton_img, medial_img, lines

if __name__ == '__main__':
    extract_wall_centerlines()
