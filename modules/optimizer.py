"""
AI-Powered Optimizer Module
Orchestrates the entire AI-enhanced warehouse optimization pipeline
"""

import json
import pandas as pd
from typing import Dict, List
import os
from datetime import datetime


class WarehouseOptimizer:
    """Main class that orchestrates the entire optimization process"""

    def __init__(self, config_path: str = "input/config.json"):
        """
        Initialize the optimizer

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.results = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"✓ Configuration loaded from {config_path}")
            return config
        else:
            # Create default AI config
            default_config = {
                "warehouse": {
                    "length": 1000,  # cm
                    "breadth": 800,  # cm
                    "height": 400,  # cm
                    "aisle_width": 100,  # cm
                },
                "ai_detection": {
                    "model_path": "models/best (3).pt",
                    "confidence_threshold": 0.5,
                    "pixels_per_cm": 10.0,
                    "save_visualizations": True,
                },
                "paths": {"image_folder": "input/images", "output_folder": "output"},
            }

            # Create config file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=4)

            print(f"✓ Default AI configuration created at {config_path}")
            print(
                "  Please update the warehouse dimensions and YOLO model path as needed."
            )

            return default_config

    def run_optimization(self) -> Dict:
        """
        Run the complete AI-powered optimization pipeline

        Returns:
            Dictionary containing all results
        """
        print("\n" + "=" * 70)
        print("  AI-ENHANCED WAREHOUSE OPTIMIZATION SYSTEM")
        print("=" * 70)

        # Import required modules
        from modules.ai_object_detector import AIObjectDetector
        from modules.bin_packing import BinPacking3D, Warehouse
        from modules.visualizer import Visualizer3D

        # Step 1: AI Object Detection
        print("\n[STEP 1/4] AI-Powered Object Detection...")
        print("-" * 70)

        detector = AIObjectDetector(
            model_path=self.config["ai_detection"]["model_path"],
            pixels_per_cm=self.config["ai_detection"]["pixels_per_cm"],
            confidence_threshold=self.config["ai_detection"]["confidence_threshold"],
        )

        image_folder = self.config["paths"]["image_folder"]
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
            raise ValueError(f"Please place object images in: {image_folder}")

        objects = detector.process_image_batch(
            image_folder,
            save_visualizations=self.config["ai_detection"]["save_visualizations"],
        )

        if not objects:
            raise ValueError("No objects were successfully detected!")

        # Save detection results
        output_folder = self.config["paths"]["output_folder"]
        os.makedirs(output_folder, exist_ok=True)
        detector.export_detections(
            objects, os.path.join(output_folder, "ai_detections.json")
        )

        # Step 2: Run bin packing algorithm
        print("\n[STEP 2/4] Running 3D Bin Packing Algorithm...")
        print("-" * 70)

        warehouse = Warehouse(
            length=self.config["warehouse"]["length"],
            breadth=self.config["warehouse"]["breadth"],
            height=self.config["warehouse"]["height"],
            aisle_width=self.config["warehouse"]["aisle_width"],
        )

        packer = BinPacking3D(warehouse)
        results = packer.pack_boxes(objects)

        # Step 3: Save results
        print("\n[STEP 3/4] Saving Results...")
        print("-" * 70)

        output_folder = self.config["paths"]["output_folder"]
        os.makedirs(output_folder, exist_ok=True)

        self._save_coordinates(results["coordinates"], output_folder)
        self._save_statistics(results["statistics"], output_folder)

        # Step 4: Create visualizations
        print("\n[STEP 4/4] Creating Visualizations...")
        print("-" * 70)

        visualizer = Visualizer3D()

        # 3D placement visualization
        visualizer.visualize_placement(
            results["coordinates"],
            self.config["warehouse"],
            os.path.join(output_folder, "placement_3d.html"),
        )

        # Statistics visualization
        visualizer.create_statistics_visualization(
            results["statistics"], os.path.join(output_folder, "statistics.html")
        )

        # Final summary
        self._print_summary(results["statistics"])

        self.results = results
        return results

    def _save_coordinates(self, coordinates: List[Dict], output_folder: str):
        """Save placement coordinates to CSV"""
        df = pd.DataFrame(coordinates)
        csv_path = os.path.join(output_folder, "placement_coordinates.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Coordinates saved to: {csv_path}")

    def _save_statistics(self, statistics: Dict, output_folder: str):
        """Save statistics to text file"""
        txt_path = os.path.join(output_folder, "statistics.txt")

        with open(txt_path, "w") as f:
            f.write("AI-ENHANCED WAREHOUSE OPTIMIZATION STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detection Method: YOLO AI\n")
            f.write(f"Model: {self.config['ai_detection']['model_path']}\n\n")

            f.write("PLACEMENT SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Objects: {statistics['total_boxes']}\n")
            f.write(f"Successfully Placed: {statistics['placed_boxes']}\n")
            f.write(f"Failed to Place: {statistics['failed_boxes']}\n")
            f.write(f"Placement Rate: {statistics['placement_rate']}%\n\n")

            f.write("SPACE UTILIZATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Warehouse Volume: {statistics['warehouse_volume']:,.2f} cm³\n")
            f.write(f"Used Volume: {statistics['used_volume']:,.2f} cm³\n")
            f.write(f"Wasted Volume: {statistics['wasted_volume']:,.2f} cm³\n")
            f.write(f"Space Utilization: {statistics['space_utilization']}%\n")

        print(f"✓ Statistics saved to: {txt_path}")

    def _print_summary(self, statistics: Dict):
        """Print final summary to console"""
        print("\n" + "=" * 70)
        print("  AI-POWERED OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(
            f"\n📦 Objects Placed: {statistics['placed_boxes']}/{statistics['total_boxes']}"
        )
        print(f"📊 Space Utilization: {statistics['space_utilization']}%")
        print(f"✅ Placement Rate: {statistics['placement_rate']}%")
        print(f"🤖 Detection: YOLO AI-Powered")
        print(f"\n💾 All results saved to: {self.config['paths']['output_folder']}/")
        print("\n" + "=" * 70 + "\n")

    def export_results(self, format: str = "json", output_path: str = None):
        """
        Export results in different formats

        Args:
            format: 'json', 'excel', or 'csv'
            output_path: Custom output path (optional)
        """
        if not self.results:
            raise ValueError("No results to export. Run optimization first.")

        output_folder = self.config["paths"]["output_folder"]

        if format == "json":
            path = output_path or os.path.join(output_folder, "results.json")
            with open(path, "w") as f:
                json.dump(self.results, f, indent=4)
            print(f"✓ Results exported to JSON: {path}")

        elif format == "excel":
            path = output_path or os.path.join(output_folder, "results.xlsx")
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame(self.results["coordinates"]).to_excel(
                    writer, sheet_name="Coordinates", index=False
                )
                pd.DataFrame([self.results["statistics"]]).to_excel(
                    writer, sheet_name="Statistics", index=False
                )
            print(f"✓ Results exported to Excel: {path}")

        elif format == "csv":
            path = output_path or os.path.join(
                output_folder, "placement_coordinates.csv"
            )
            pd.DataFrame(self.results["coordinates"]).to_csv(path, index=False)
            print(f"✓ Results exported to CSV: {path}")

        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage
if __name__ == "__main__":
    # Create optimizer instance
    optimizer = WarehouseOptimizer()

    # Run optimization
    results = optimizer.run_optimization()

    # Export results in different formats
    optimizer.export_results("json")
    optimizer.export_results("excel")
