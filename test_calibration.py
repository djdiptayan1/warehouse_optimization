"""
Test Script for Reference-Based Dimension Calibration

This demonstrates how to use the DimensionCalibrator to get actual dimensions
of objects by using a reference box with known dimensions.

HOW IT WORKS:
1. Place TWO boxes in each image:
   - SMALLER box = Reference box (with KNOWN dimensions)
   - LARGER box = Unknown box (dimensions to be measured)

2. Update config.json with reference box dimensions under "reference_box"

3. Run this script - it will automatically:
   - Detect both boxes
   - Identify the smaller one as the reference
   - Calculate the calibration factor (pixels per cm)
   - Calculate the actual dimensions of the larger box

EXAMPLE SETUP:
- Reference box: 10cm × 10cm × 10cm (small box)
- Unknown box: ? × ? × ? (larger box to measure)
- Both boxes in the same image

The system will output actual dimensions of the unknown box!
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.dimension_calibrator import DimensionCalibrator


def load_config(config_path="input/config.json"):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def test_single_image_calibration():
    """
    Test calibration with a single image containing reference + unknown boxes
    """
    print("\n" + "=" * 80)
    print("TEST: Single Image Calibration")
    print("=" * 80)

    # Load config
    config = load_config()

    # Get reference box dimensions from config
    reference_dims = config.get(
        "reference_box", {"length": 10.0, "breadth": 10.0, "height": 10.0}
    )

    print(f"\nReference box (known dimensions): {reference_dims}")

    # Initialize calibrator
    calibrator = DimensionCalibrator(
        model_path=config["ai_detection"]["model_path"],
        reference_dimensions=reference_dims,
        confidence_threshold=config["ai_detection"]["confidence_threshold"],
    )

    # Test with first image (make sure it has 2 boxes!)
    test_image = "input/images/image1.jpeg"

    if not os.path.exists(test_image):
        print(f"\n⚠ Test image not found: {test_image}")
        print("Please ensure you have an image with TWO boxes:")
        print("  1. Smaller box = reference (known dimensions)")
        print("  2. Larger box = unknown (to be measured)")
        return

    # Run calibration
    result = calibrator.detect_and_calibrate(test_image, save_visualization=True)

    if result:
        print("\n" + "=" * 80)
        print("CALIBRATION SUCCESSFUL!")
        print("=" * 80)
        print(f"\nCalibration Factor: {result['calibration_factor']} pixels/cm")
        print(f"Total objects measured: {result['total_objects']}")

        print("\nCalibrated Objects:")
        for obj in result["calibrated_objects"]:
            print(f"\n  Object #{obj['object_id']}:")
            print(f"    Class: {obj['class_name']}")
            print(
                f"    Dimensions: {obj['length']} × {obj['breadth']} × {obj['height']} cm (L×B×H)"
            )
            print(f"    Volume: {obj['volume']} cm³")
            print(f"    Confidence: {obj['confidence']}")

        # Save results
        output_file = "output/calibration_result_single.json"
        calibrator.export_calibration_results([result], output_file)

    else:
        print("\n❌ Calibration failed!")
        print("Make sure your image contains at least 2 boxes:")
        print("  - 1 small reference box (with known dimensions)")
        print("  - 1+ larger unknown boxes (to be measured)")


def test_batch_calibration():
    """
    Test calibration with multiple images
    Each image should contain reference + unknown boxes
    """
    print("\n" + "=" * 80)
    print("TEST: Batch Image Calibration")
    print("=" * 80)

    # Load config
    config = load_config()

    # Get reference box dimensions from config
    reference_dims = config.get(
        "reference_box", {"length": 10.0, "breadth": 10.0, "height": 10.0}
    )

    print(f"\nReference box (known dimensions): {reference_dims}")

    # Initialize calibrator
    calibrator = DimensionCalibrator(
        model_path=config["ai_detection"]["model_path"],
        reference_dimensions=reference_dims,
        confidence_threshold=config["ai_detection"]["confidence_threshold"],
    )

    # Process all images in folder
    image_folder = config["paths"]["image_folder"]

    all_calibrated_objects = calibrator.process_calibration_batch(
        image_folder, save_visualizations=True
    )

    if all_calibrated_objects:
        print("\n" + "=" * 80)
        print("BATCH CALIBRATION RESULTS")
        print("=" * 80)
        print(f"\nTotal objects calibrated: {len(all_calibrated_objects)}")

        # Group by filename
        by_file = {}
        for obj in all_calibrated_objects:
            filename = obj["filename"]
            if filename not in by_file:
                by_file[filename] = []
            by_file[filename].append(obj)

        print(f"Images processed: {len(by_file)}")

        for filename, objects in by_file.items():
            print(f"\n{filename}:")
            for obj in objects:
                print(
                    f"  - {obj['length']} × {obj['breadth']} × {obj['height']} cm (Vol: {obj['volume']} cm³)"
                )

        # Save all results
        output_file = "output/calibration_results_batch.json"
        calibrator.export_calibration_results(all_calibrated_objects, output_file)

    else:
        print("\n❌ No objects calibrated!")


def test_comparison():
    """
    Compare original AI detection vs reference-based calibration
    Shows the difference between using fixed pixels_per_cm vs actual calibration
    """
    print("\n" + "=" * 80)
    print("TEST: Comparison - AI Detection vs Reference Calibration")
    print("=" * 80)

    from modules.ai_object_detector import AIObjectDetector

    # Load config
    config = load_config()

    test_image = "input/images/image1.jpeg"

    if not os.path.exists(test_image):
        print(f"\n⚠ Test image not found: {test_image}")
        return

    # Method 1: Original AI detection (fixed pixels_per_cm)
    print("\n" + "-" * 80)
    print("METHOD 1: Original AI Detection (Fixed pixels_per_cm)")
    print("-" * 80)

    detector = AIObjectDetector(
        model_path=config["ai_detection"]["model_path"],
        pixels_per_cm=config["ai_detection"]["pixels_per_cm"],
        confidence_threshold=config["ai_detection"]["confidence_threshold"],
    )

    original_objects = detector.detect_objects_in_image(
        test_image, save_visualization=False
    )

    print("\nOriginal detections:")
    for obj in original_objects:
        print(f"  - {obj['length']} × {obj['breadth']} × {obj['height']} cm")

    # Method 2: Reference-based calibration
    print("\n" + "-" * 80)
    print("METHOD 2: Reference-Based Calibration (Dynamic calibration)")
    print("-" * 80)

    reference_dims = config.get(
        "reference_box", {"length": 10.0, "breadth": 10.0, "height": 10.0}
    )

    calibrator = DimensionCalibrator(
        model_path=config["ai_detection"]["model_path"],
        reference_dimensions=reference_dims,
        confidence_threshold=config["ai_detection"]["confidence_threshold"],
    )

    calibrated_result = calibrator.detect_and_calibrate(
        test_image, save_visualization=False
    )

    if calibrated_result:
        print("\nCalibrated detections:")
        for obj in calibrated_result["calibrated_objects"]:
            print(f"  - {obj['length']} × {obj['breadth']} × {obj['height']} cm")

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(
            f"\nOriginal method: Uses fixed pixels_per_cm = {config['ai_detection']['pixels_per_cm']}"
        )
        print(
            f"Calibration method: Uses calculated pixels_per_cm = {calibrated_result['calibration_factor']}"
        )
        print("\nCalibration method is MORE ACCURATE because it adapts to:")
        print("  - Camera distance")
        print("  - Camera angle")
        print("  - Image resolution")
        print("  - Lens distortion")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("REFERENCE-BASED DIMENSION CALIBRATION TEST SUITE")
    print("=" * 80)
    print("\nThis test suite demonstrates how to get ACTUAL dimensions from 2D images")
    print("by using a reference box with known dimensions.")
    print("\nREQUIREMENTS:")
    print("  1. Each image must contain at least 2 boxes:")
    print("     - SMALLER box = Reference (known dimensions from config.json)")
    print("     - LARGER box = Unknown (dimensions to be measured)")
    print("  2. Update config.json with reference box dimensions")
    print("  3. The system will automatically calibrate and measure!")

    # Run tests
    try:
        # Test 1: Single image
        test_single_image_calibration()

        # Test 2: Batch processing
        # Uncomment to test batch processing:
        # test_batch_calibration()

        # Test 3: Comparison
        # Uncomment to compare methods:
        # test_comparison()

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print("\nCheck output/annotated_calibrated/ for annotated images:")
    print("  - GREEN box = Reference box (known dimensions)")
    print("  - RED box = Unknown box (measured dimensions)")
