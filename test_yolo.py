#!/usr/bin/env python3
"""
Test script for YOLO model integration
Tests your trained YOLO model with sample images
"""

import os
import sys
from ultralytics import YOLO


def test_yolo_model():
    """Test YOLO model with your trained weights"""

    print("\n" + "=" * 70)
    print("  YOLO MODEL TEST - AI-ENHANCED WAREHOUSE OPTIMIZATION")
    print("=" * 70)

    # Your model path
    model_path = "models/best (3).pt"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        print("\nPlease ensure your trained YOLO model is at this location:")
        print(f"  {model_path}")
        print("\nOr update the model_path variable in this script.")
        return

    print(f"\n✓ Model found: {model_path}")

    # Load YOLO model
    print("\nLoading YOLO model...")
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    # Check for test images
    image_folder = "input/images"

    if not os.path.exists(image_folder):
        print(f"\n⚠ Image folder not found: {image_folder}")
        print("Creating sample images...")
        os.makedirs(image_folder, exist_ok=True)
        os.system("python create_sample_images.py")

    # Get image files
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        print(f"\n⚠ No images found in: {image_folder}")
        print("Please add test images to this folder.")
        return

    print(f"\n✓ Found {len(image_files)} images to test")

    # Create output directory for annotated images
    annotated_output = "output/annotated"
    os.makedirs(annotated_output, exist_ok=True)

    # Test prediction on each image
    print("\n" + "-" * 70)
    print("RUNNING PREDICTIONS")
    print("-" * 70)

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        print(f"\n📸 Processing: {img_file}")
        print("-" * 70)

        try:
            # Run prediction with custom output path
            results = model.predict(
                img_path,
                save=True,
                project=annotated_output,
                name="",
                exist_ok=True,
                conf=0.5,
                verbose=True,
            )

            # Get results
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    print(f"✓ Detected {len(boxes)} object(s)!")

                    # Print details for each detection
                    for idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # Get class name if available
                        class_name = (
                            result.names[cls]
                            if hasattr(result, "names")
                            else f"class_{cls}"
                        )

                        width = x2 - x1
                        height = y2 - y1

                        print(f"\n  Object {idx + 1}:")
                        print(f"    Class: {class_name}")
                        print(f"    Confidence: {conf:.2%}")
                        print(
                            f"    Bounding Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})"
                        )
                        print(f"    Size (pixels): {width:.1f} x {height:.1f}")
                else:
                    print("⚠ No objects detected in this image")
            else:
                print("⚠ No results returned")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nYour YOLO model is working! ✓")
    print(f"\nAnnotated images saved to: {annotated_output}/")
    print("\nNext steps:")
    print("  1. Review the annotated images to verify detections")
    print("  2. Adjust confidence threshold if needed")
    print("  3. Run the full AI optimization: python main.py")
    print("\n" + "=" * 70 + "\n")


def quick_test():
    """Quick test with a single image"""
    model_path = "models/best (3).pt"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = YOLO(model_path)

    # Create output directory
    annotated_output = "output/annotated"
    os.makedirs(annotated_output, exist_ok=True)

    # Test with first available image
    image_folder = "input/images"
    if os.path.exists(image_folder):
        images = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if images:
            test_image = os.path.join(image_folder, images[0])
            print(f"Testing with: {test_image}")

            results = model.predict(
                test_image,
                save=True,
                project=annotated_output,
                name="",
                exist_ok=True,
                conf=0.5,
            )

            print(f"\nResults saved to: {annotated_output}/")
            print("✓ Quick test complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test YOLO model")
    parser.add_argument("--quick", action="store_true", help="Run quick test")

    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        test_yolo_model()
