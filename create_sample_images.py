#!/usr/bin/env python3
"""
Create sample test images for warehouse optimization system
This script generates simple geometric shapes as test objects
"""

import cv2
import numpy as np
import os


def create_sample_images():
    """Create sample object images for testing"""

    # Create output directory
    output_dir = "input/images"
    os.makedirs(output_dir, exist_ok=True)

    # Image dimensions
    img_width, img_height = 400, 300

    # Sample object 1: Rectangle (box)
    img1 = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # white background
    cv2.rectangle(img1, (100, 80), (300, 220), (100, 100, 100), -1)  # gray rectangle
    cv2.rectangle(img1, (100, 80), (300, 220), (0, 0, 0), 2)  # black border
    cv2.imwrite(os.path.join(output_dir, "box1.jpg"), img1)
    print("✓ Created box1.jpg (200x140 pixels)")

    # Sample object 2: Smaller rectangle
    img2 = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    cv2.rectangle(img2, (120, 100), (280, 200), (150, 150, 150), -1)  # light gray
    cv2.rectangle(img2, (120, 100), (280, 200), (0, 0, 0), 2)  # black border
    cv2.imwrite(os.path.join(output_dir, "box2.jpg"), img2)
    print("✓ Created box2.jpg (160x100 pixels)")

    # Sample object 3: Square
    img3 = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    cv2.rectangle(img3, (150, 100), (250, 200), (80, 80, 80), -1)  # dark gray
    cv2.rectangle(img3, (150, 100), (250, 200), (0, 0, 0), 2)  # black border
    cv2.imwrite(os.path.join(output_dir, "box3.jpg"), img3)
    print("✓ Created box3.jpg (100x100 pixels)")

    # Sample object 4: Tall rectangle
    img4 = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    cv2.rectangle(img4, (160, 50), (240, 250), (120, 120, 120), -1)  # medium gray
    cv2.rectangle(img4, (160, 50), (240, 250), (0, 0, 0), 2)  # black border
    cv2.imwrite(os.path.join(output_dir, "box4.jpg"), img4)
    print("✓ Created box4.jpg (80x200 pixels)")

    print(f"\n🎯 Sample images created in {output_dir}/")
    print("These images simulate objects with different dimensions for testing.")
    print(
        "When using 'contour' method, the system will detect these shapes automatically."
    )
    print("Recommended pixels_per_cm calibration: 10.0 (so 100 pixels = 10 cm)")


if __name__ == "__main__":
    create_sample_images()
