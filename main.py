"""
AI-Enhanced Warehouse Optimization System
Uses YOLO for automatic object detection and dimension extraction
"""

import sys
import os
import json
from modules.ai_object_detector import AIObjectDetector
from modules.bin_packing import BinPacking3D, Warehouse
from modules.visualizer import Visualizer3D
import pandas as pd
from datetime import datetime


def setup_ai_config():
    """Setup configuration for AI-powered detection"""
    print("\n" + "=" * 70)
    print("  AI-ENHANCED WAREHOUSE OPTIMIZATION SYSTEM")
    print("=" * 70)

    config_path = "input/config.json"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"\n✓ Configuration loaded from: {config_path}")
        return config

    print("\nInitial Setup - AI Mode")
    print("-" * 70)

    # Get warehouse dimensions
    print("\nEnter warehouse dimensions (in centimeters):")
    length = float(input("  Length (cm): "))
    breadth = float(input("  Breadth (cm): "))
    height = float(input("  Height (cm): "))
    aisle_width = float(input("  Aisle width (cm) [default: 100]: ") or "100")

    # YOLO model configuration
    print("\nYOLO Model Configuration:")
    model_path = (
        input("  Model path [default: models/best (3).pt]: ") or "models/best (3).pt"
    )

    confidence = float(
        input("  Confidence threshold [0.0-1.0, default: 0.5]: ") or "0.5"
    )

    # Calibration
    print("\nCalibration:")
    pixels_per_cm = float(input("  Pixels per cm [default: 10.0]: ") or "10.0")

    # Create configuration
    config = {
        "warehouse": {
            "length": length,
            "breadth": breadth,
            "height": height,
            "aisle_width": aisle_width,
        },
        "ai_detection": {
            "model_path": model_path,
            "confidence_threshold": confidence,
            "pixels_per_cm": pixels_per_cm,
            "save_visualizations": True,
        },
        "paths": {"image_folder": "input/images", "output_folder": "output"},
    }

    # Save configuration
    os.makedirs("input", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("\n✓ AI configuration saved successfully!")
    return config


def check_yolo_model(model_path: str) -> bool:
    """Check if YOLO model exists"""
    if os.path.exists(model_path):
        print(f"✓ YOLO model found: {model_path}")
        return True
    else:
        print(f"⚠ YOLO model not found: {model_path}")
        print(f"  Please ensure your trained model is at: {model_path}")
        return False


def run_ai_optimization():
    """Run warehouse optimization with AI-powered object detection"""
    print("\n" + "=" * 70)
    print("  AI-ENHANCED WAREHOUSE OPTIMIZATION")
    print("=" * 70)

    # Load or create config
    config = setup_ai_config()

    # Check for images
    image_folder = config["paths"]["image_folder"]
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"\n⚠ Created folder: {image_folder}")
        print("  Please add your object images to this folder and run again.")
        return

    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not images:
        print(f"\n⚠ No images found in: {image_folder}")
        print("  Please add your object images to this folder and run again.")
        return

    print(f"\n✓ Found {len(images)} images in: {image_folder}")

    # Check YOLO model
    model_path = config["ai_detection"]["model_path"]
    if not check_yolo_model(model_path):
        print("\nContinue without YOLO? (will use manual input)")
        choice = input("(y/n): ").lower()
        if choice != "y":
            print("\nExiting... Please add your YOLO model and run again.")
            return

    # Confirm before starting
    print("\n" + "-" * 70)
    input("Press Enter to start AI-powered optimization... ")

    # STEP 1: AI Object Detection
    print("\n" + "=" * 70)
    print("[STEP 1/4] AI-Powered Object Detection")
    print("=" * 70)

    detector = AIObjectDetector(
        model_path=model_path,
        pixels_per_cm=config["ai_detection"]["pixels_per_cm"],
        confidence_threshold=config["ai_detection"]["confidence_threshold"],
    )

    # Process images with YOLO
    detected_objects = detector.process_image_batch(
        image_folder, save_visualizations=config["ai_detection"]["save_visualizations"]
    )

    if not detected_objects:
        print("\n❌ No objects detected or processed!")
        return

    # Save detection results
    output_folder = config["paths"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    detector.export_detections(
        detected_objects, os.path.join(output_folder, "ai_detections.json")
    )

    # STEP 2: 3D Bin Packing
    print("\n" + "=" * 70)
    print("[STEP 2/4] Running 3D Bin Packing Algorithm")
    print("=" * 70)

    warehouse = Warehouse(
        length=config["warehouse"]["length"],
        breadth=config["warehouse"]["breadth"],
        height=config["warehouse"]["height"],
        aisle_width=config["warehouse"]["aisle_width"],
    )

    packer = BinPacking3D(warehouse)
    results = packer.pack_boxes(detected_objects)

    # STEP 3: Save Results
    print("\n" + "=" * 70)
    print("[STEP 3/4] Saving Results")
    print("=" * 70)

    # Save coordinates
    df = pd.DataFrame(results["coordinates"])
    csv_path = os.path.join(output_folder, "placement_coordinates.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Coordinates saved to: {csv_path}")

    # Save statistics
    txt_path = os.path.join(output_folder, "statistics.txt")
    with open(txt_path, "w") as f:
        f.write("AI-ENHANCED WAREHOUSE OPTIMIZATION STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Detection Method: YOLO AI\n")
        f.write(f"Model: {model_path}\n\n")

        f.write("PLACEMENT SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Objects: {results['statistics']['total_boxes']}\n")
        f.write(f"Successfully Placed: {results['statistics']['placed_boxes']}\n")
        f.write(f"Failed to Place: {results['statistics']['failed_boxes']}\n")
        f.write(f"Placement Rate: {results['statistics']['placement_rate']}%\n\n")

        f.write("SPACE UTILIZATION\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Warehouse Volume: {results['statistics']['warehouse_volume']:,.2f} cm³\n"
        )
        f.write(f"Used Volume: {results['statistics']['used_volume']:,.2f} cm³\n")
        f.write(f"Wasted Volume: {results['statistics']['wasted_volume']:,.2f} cm³\n")
        f.write(f"Space Utilization: {results['statistics']['space_utilization']}%\n")

    print(f"✓ Statistics saved to: {txt_path}")

    # STEP 4: Create Visualizations
    print("\n" + "=" * 70)
    print("[STEP 4/4] Creating Visualizations")
    print("=" * 70)

    visualizer = Visualizer3D()

    # 3D placement visualization
    visualizer.visualize_placement(
        results["coordinates"],
        config["warehouse"],
        os.path.join(output_folder, "placement_3d.html"),
    )

    # Statistics visualization
    visualizer.create_statistics_visualization(
        results["statistics"], os.path.join(output_folder, "statistics.html")
    )

    # Final summary
    print("\n" + "=" * 70)
    print("  AI-POWERED OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(
        f"\n📦 Objects Placed: {results['statistics']['placed_boxes']}/{results['statistics']['total_boxes']}"
    )
    print(f"📊 Space Utilization: {results['statistics']['space_utilization']}%")
    print(f"✅ Placement Rate: {results['statistics']['placement_rate']}%")
    print(f"🤖 Detection: YOLO AI-Powered")
    print(f"\n💾 All results saved to: {output_folder}/")
    print("\nGenerated files:")
    print("  • ai_detections.json - YOLO detection results")
    print("  • placement_coordinates.csv - Box positions")
    print("  • statistics.txt - Summary statistics")
    print("  • placement_3d.html - Interactive 3D visualization")
    print("  • statistics.html - Statistics dashboard")

    # Ask to open visualization
    try:
        open_viz = input("\nOpen 3D visualization in browser? (y/n): ").lower()
        if open_viz == "y":
            import webbrowser

            webbrowser.open(os.path.join(output_folder, "placement_3d.html"))
    except:
        pass

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        run_ai_optimization()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
