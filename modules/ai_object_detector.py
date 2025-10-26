"""
AI-Powered Object Detection Module
Uses YOLO for automatic bounding box detection and dimension extraction
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import json


class AIObjectDetector:
    """
    AI-powered object detector using YOLO for automatic dimension extraction
    """

    def __init__(
        self,
        model_path: str = "models/best (3).pt",
        pixels_per_cm: float = 10.0,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the AI Object Detector

        Args:
            model_path: Path to your trained YOLO model weights
            pixels_per_cm: Calibration factor (pixels per centimeter)
            confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        """
        self.pixels_per_cm = pixels_per_cm
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None

        # Load YOLO model
        self._load_model()

    def _load_model(self):
        """Load the YOLO model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading YOLO model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                print("✓ YOLO model loaded successfully!")
            else:
                print(f"⚠ Model not found at: {self.model_path}")
                print("  Falling back to manual input mode")
                self.model = None
        except Exception as e:
            print(f"⚠ Error loading YOLO model: {str(e)}")
            print("  Falling back to manual input mode")
            self.model = None

    def detect_objects_in_image(
        self, image_path: str, save_visualization: bool = True
    ) -> List[Dict]:
        """
        Detect objects in an image using YOLO and extract dimensions

        Args:
            image_path: Path to the image file
            save_visualization: Whether to save annotated image

        Returns:
            List of detected objects with dimensions
        """
        if self.model is None:
            print(f"⚠ Model not available for {image_path}")
            return self._manual_fallback(image_path)

        try:
            # Create output directory for annotated images
            annotated_output = "output/annotated"
            os.makedirs(annotated_output, exist_ok=True)

            # Run YOLO prediction with custom project path
            results = self.model.predict(
                image_path,
                save=save_visualization,
                project=annotated_output,
                name="",
                exist_ok=True,
                conf=self.confidence_threshold,
                verbose=False,
            )

            # Extract bounding boxes
            detected_objects = []

            if len(results) > 0:
                result = results[0]  # Get first result
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    # Process each detected box
                    for idx, box in enumerate(boxes):
                        # Get box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Calculate dimensions in pixels
                        width_px = x2 - x1
                        height_px = y2 - y1

                        # Convert to centimeters
                        length_cm = width_px / self.pixels_per_cm
                        breadth_cm = height_px / self.pixels_per_cm

                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # Get class name if available
                        class_name = (
                            result.names[class_id]
                            if hasattr(result, "names")
                            else f"object_{class_id}"
                        )

                        # AI-based height estimation (no user input required)
                        # Estimate height based on object dimensions using intelligent heuristics
                        height_cm = self._estimate_height(
                            length_cm, breadth_cm, class_name
                        )

                        print(f"\n  ✓ Detected: {class_name}")
                        print(f"    Confidence: {confidence:.2%}")
                        print(
                            f"    Dimensions: {length_cm:.1f} × {breadth_cm:.1f} × {height_cm:.1f} cm (L×B×H)"
                        )
                        print(
                            f"    Volume: {(length_cm * breadth_cm * height_cm):.1f} cm³"
                        )

                        obj_data = {
                            "filename": os.path.basename(image_path),
                            "object_id": idx,
                            "class_name": class_name,
                            "confidence": round(confidence, 3),
                            "length": round(length_cm, 2),
                            "breadth": round(breadth_cm, 2),
                            "height": round(height_cm, 2),
                            "volume": round(length_cm * breadth_cm * height_cm, 2),
                            "bbox_pixels": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            },
                        }

                        detected_objects.append(obj_data)

                    print(
                        f"\n✓ Detected {len(detected_objects)} object(s) in {os.path.basename(image_path)}"
                    )
                else:
                    print(f"\n⚠ No objects detected in {os.path.basename(image_path)}")
                    print("  Falling back to manual input...")
                    return self._manual_fallback(image_path)

            return detected_objects

        except Exception as e:
            print(f"\n❌ Error during detection: {str(e)}")
            print("  Falling back to manual input...")
            return self._manual_fallback(image_path)

    def _estimate_height(
        self, length_cm: float, breadth_cm: float, class_name: str = ""
    ) -> float:
        """
        AI-based height estimation using intelligent heuristics

        Args:
            length_cm: Object length in cm
            breadth_cm: Object breadth in cm
            class_name: Detected class name (optional)

        Returns:
            Estimated height in cm
        """
        # Method 1: Geometric mean (good for regular objects)
        geometric_height = np.sqrt(length_cm * breadth_cm)

        # Method 2: Average of dimensions
        avg_height = (length_cm + breadth_cm) / 2

        # Method 3: Proportional to smaller dimension (for boxes)
        min_dim = min(length_cm, breadth_cm)
        proportional_height = min_dim * 0.8  # Typically 80% of smaller dimension

        # Method 4: Class-based estimation
        class_lower = class_name.lower()

        # Heuristic rules based on common object types
        if any(
            word in class_lower for word in ["box", "package", "carton", "container"]
        ):
            # Boxes tend to be roughly cubic or slightly shorter
            height = min(geometric_height, avg_height * 0.9)
        elif any(word in class_lower for word in ["pallet", "platform"]):
            # Pallets are typically flat
            height = max(length_cm, breadth_cm) * 0.15  # ~15% of larger dimension
        elif any(word in class_lower for word in ["crate", "bin"]):
            # Crates are often taller
            height = geometric_height * 1.1
        elif "book" in class_lower or "flat" in class_lower:
            # Flat objects
            height = min(length_cm, breadth_cm) * 0.1
        else:
            # Default: Use weighted combination
            height = geometric_height * 0.5 + proportional_height * 0.5

        # Apply reasonable constraints
        # Height should be between 20% and 200% of the smaller dimension
        min_height = min_dim * 0.2
        max_height = max(length_cm, breadth_cm) * 2.0

        # Clamp to reasonable range
        height = max(min_height, min(height, max_height))

        # Round to 1 decimal place
        return round(height, 1)

    def _manual_fallback(self, image_path: str) -> List[Dict]:
        """Fallback using basic image analysis (no user input)"""
        print(f"\nUsing basic image analysis for: {os.path.basename(image_path)}")

        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")

            # Simple contour-based dimension extraction
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Convert to cm
                length = w / self.pixels_per_cm
                breadth = h / self.pixels_per_cm
                height = self._estimate_height(length, breadth, "unknown")

                print(f"  ✓ Estimated: {length:.1f} × {breadth:.1f} × {height:.1f} cm")
            else:
                # Use default dimensions if no contours found
                length, breadth, height = 20.0, 15.0, 10.0
                print(
                    f"  ⚠ Using default dimensions: {length} × {breadth} × {height} cm"
                )

        except Exception as e:
            print(f"  ⚠ Fallback failed: {e}")
            # Use reasonable default dimensions
            length, breadth, height = 20.0, 15.0, 10.0
            print(f"  Using default dimensions: {length} × {breadth} × {height} cm")

        return [
            {
                "filename": os.path.basename(image_path),
                "object_id": 0,
                "class_name": "fallback_estimated",
                "confidence": 0.5,
                "length": round(length, 2),
                "breadth": round(breadth, 2),
                "height": round(height, 2),
                "volume": round(length * breadth * height, 2),
                "bbox_pixels": None,
            }
        ]

    def process_image_batch(
        self, image_folder: str, save_visualizations: bool = True
    ) -> List[Dict]:
        """
        Process multiple images from a folder using YOLO

        Args:
            image_folder: Path to folder containing images
            save_visualizations: Whether to save annotated images

        Returns:
            List of all detected objects with dimensions
        """
        all_objects = []

        # Get all image files
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = [
            f
            for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if not image_files:
            raise ValueError(f"No valid images found in {image_folder}")

        print(f"\n{'=' * 70}")
        print(f"AI-POWERED OBJECT DETECTION")
        print(f"{'=' * 70}")
        print(f"Model: {self.model_path}")
        print(f"Found {len(image_files)} images to process")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Calibration: {self.pixels_per_cm} pixels/cm")
        print(f"{'=' * 70}\n")

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            print(f"\nProcessing: {img_file}")
            print("-" * 70)

            try:
                objects = self.detect_objects_in_image(img_path, save_visualizations)
                all_objects.extend(objects)
                print(f"✓ Successfully processed: {img_file}")

            except Exception as e:
                print(f"✗ Error processing {img_file}: {str(e)}")
                # Try manual input as fallback
                try:
                    obj_data = self._manual_fallback(img_path)
                    all_objects.extend(obj_data)
                except:
                    print(f"  Skipping {img_file}")

        print(f"\n{'=' * 70}")
        print(
            f"Successfully processed {len(all_objects)} objects from {len(image_files)} images"
        )
        print(f"{'=' * 70}\n")

        return all_objects

    def calibrate_from_reference(
        self, reference_image_path: str, known_width_cm: float
    ):
        """
        Calibrate pixels_per_cm using a reference object with known dimensions

        Args:
            reference_image_path: Path to image with known object
            known_width_cm: Known width of the reference object in cm
        """
        print(f"\n{'=' * 70}")
        print("CALIBRATION MODE")
        print(f"{'=' * 70}")

        if self.model is None:
            print("⚠ YOLO model not available for calibration")
            return

        try:
            results = self.model.predict(
                reference_image_path, conf=self.confidence_threshold, verbose=False
            )

            if len(results) > 0 and results[0].boxes is not None:
                box = results[0].boxes[0]  # Use first detected object
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width_px = x2 - x1

                self.pixels_per_cm = width_px / known_width_cm

                print(f"✓ Calibration successful!")
                print(f"  Reference width: {known_width_cm} cm")
                print(f"  Detected width: {width_px:.1f} pixels")
                print(f"  Calibration factor: {self.pixels_per_cm:.2f} pixels/cm")
            else:
                print("⚠ No object detected in reference image")

        except Exception as e:
            print(f"❌ Calibration failed: {str(e)}")

        print(f"{'=' * 70}\n")

    def export_detections(self, detections: List[Dict], output_path: str):
        """Export detection results to JSON"""

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Convert detections to JSON-serializable format
        serializable_detections = convert_to_native(detections)

        with open(output_path, "w") as f:
            json.dump(serializable_detections, f, indent=4)
        print(f"✓ Detections exported to: {output_path}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded YOLO model"""
        if self.model is None:
            return {"status": "not_loaded", "model_path": self.model_path}

        return {
            "status": "loaded",
            "model_path": self.model_path,
            "pixels_per_cm": self.pixels_per_cm,
            "confidence_threshold": self.confidence_threshold,
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector with your model
    detector = AIObjectDetector(
        model_path="runs/train/boxes_experiment/weights/best.pt",
        pixels_per_cm=10.0,
        confidence_threshold=0.5,
    )

    # Test single image
    # objects = detector.detect_objects_in_image('input/images/box1.jpg')
    # print(objects)

    # Process batch of images
    # all_objects = detector.process_image_batch('input/images/')

    # Calibrate using reference object
    # detector.calibrate_from_reference('reference_image.jpg', known_width_cm=20.0)
