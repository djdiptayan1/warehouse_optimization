"""
Dimension Calibration Module
Uses a reference object with known dimensions to calculate actual dimensions of other objects in the same image.

How it works:
1. Place TWO boxes in the image: one KNOWN (reference) and one UNKNOWN (object to measure)
2. Add the known box dimensions to config.json under "reference_box"
3. The system will automatically identify the smaller box as the reference
4. Calculate the actual dimensions of the unknown box based on the reference

This solves the 2D-to-3D dimension problem by using a physical reference object.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import json


class DimensionCalibrator:
    """
    Calibrates object dimensions using a reference box with known dimensions
    """

    def __init__(
        self,
        model_path: str = "models/best (3).pt",
        reference_dimensions: Dict[str, float] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the Dimension Calibrator

        Args:
            model_path: Path to YOLO model weights
            reference_dimensions: Dict with 'length', 'breadth', 'height' of reference box in cm
            confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        """
        self.model_path = model_path
        self.reference_dimensions = reference_dimensions or {
            "length": 10.0,
            "breadth": 10.0,
            "height": 10.0,
        }
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.calibration_factor = None  # pixels per cm, calculated from reference

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
                self.model = None
        except Exception as e:
            print(f"⚠ Error loading YOLO model: {str(e)}")
            self.model = None

    def detect_and_calibrate(
        self, image_path: str, save_visualization: bool = True
    ) -> Dict:
        """
        Detect objects in image, identify reference box, and calculate actual dimensions

        Process:
        1. Detect all boxes in the image using YOLO
        2. Identify the SMALLER box as the reference box (known dimensions)
        3. Calculate calibration factor (pixels per cm) from reference box
        4. Use calibration to get actual dimensions of the LARGER box (unknown)

        Args:
            image_path: Path to the image file containing reference and unknown boxes
            save_visualization: Whether to save annotated image

        Returns:
            Dict containing reference box, unknown box(es), and calibration info
        """
        if self.model is None:
            print(f"⚠ Model not available for {image_path}")
            return self._manual_calibration(image_path)

        try:
            print(f"\n{'=' * 70}")
            print(f"REFERENCE-BASED DIMENSION CALIBRATION")
            print(f"{'=' * 70}")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Reference box dimensions: {self.reference_dimensions}")
            print(f"{'=' * 70}\n")

            # Run YOLO detection
            results = self.model.predict(
                image_path, conf=self.confidence_threshold, verbose=False
            )

            if (
                len(results) == 0
                or results[0].boxes is None
                or len(results[0].boxes) < 2
            ):
                print("⚠ Need at least 2 boxes detected (reference + unknown)")
                print(
                    f"  Found: {len(results[0].boxes) if results[0].boxes is not None else 0} boxes"
                )
                return None

            result = results[0]
            boxes = result.boxes

            # Extract all detected boxes
            detected_boxes = []
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width_px = x2 - x1
                height_px = y2 - y1
                area_px = width_px * height_px

                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = (
                    result.names[class_id]
                    if hasattr(result, "names")
                    else f"object_{class_id}"
                )

                detected_boxes.append(
                    {
                        "index": idx,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "width_px": width_px,
                        "height_px": height_px,
                        "area_px": area_px,
                    }
                )

            # Sort by area - SMALLEST is reference box
            detected_boxes.sort(key=lambda x: x["area_px"])

            # Reference box is the smallest one
            reference_box = detected_boxes[0]
            unknown_boxes = detected_boxes[1:]

            print(f"✓ Detected {len(detected_boxes)} boxes total")
            print(f"  - Reference box (smallest): Box #{reference_box['index']}")
            print(f"  - Unknown boxes: {len(unknown_boxes)}\n")

            # STEP 1: Calculate calibration factor from reference box
            # We'll use the average of width and height calibration for better accuracy
            ref_width_cm = self.reference_dimensions["length"]
            ref_height_cm = self.reference_dimensions["breadth"]

            pixels_per_cm_width = reference_box["width_px"] / ref_width_cm
            pixels_per_cm_height = reference_box["height_px"] / ref_height_cm

            # Use average for calibration factor
            self.calibration_factor = (pixels_per_cm_width + pixels_per_cm_height) / 2

            print(f"CALIBRATION FROM REFERENCE BOX:")
            print(f"-" * 70)
            print(
                f"  Reference box dimensions (known): {ref_width_cm} × {ref_height_cm} cm"
            )
            print(
                f"  Reference box pixels: {reference_box['width_px']:.1f} × {reference_box['height_px']:.1f} px"
            )
            print(f"  Calibration factor (width): {pixels_per_cm_width:.2f} px/cm")
            print(f"  Calibration factor (height): {pixels_per_cm_height:.2f} px/cm")
            print(f"  ✓ Final calibration factor: {self.calibration_factor:.2f} px/cm")
            print(f"-" * 70 + "\n")

            # STEP 2: Calculate actual dimensions of unknown boxes
            calibrated_objects = []

            for idx, unknown_box in enumerate(unknown_boxes):
                # Convert pixels to actual cm using calibration factor
                actual_length_cm = unknown_box["width_px"] / self.calibration_factor
                actual_breadth_cm = unknown_box["height_px"] / self.calibration_factor

                # Estimate height based on known dimensions
                # Method: Use ratio from reference box
                ref_height_ratio = self.reference_dimensions["height"] / ref_width_cm
                estimated_height_cm = actual_length_cm * ref_height_ratio

                # Alternative: Use geometric mean if reference box is cubic
                if abs(ref_width_cm - ref_height_cm) < 1.0:  # Nearly square reference
                    estimated_height_cm = np.sqrt(actual_length_cm * actual_breadth_cm)

                print(f"UNKNOWN BOX #{idx + 1}:")
                print(f"-" * 70)
                print(
                    f"  Pixels: {unknown_box['width_px']:.1f} × {unknown_box['height_px']:.1f} px"
                )
                print(
                    f"  ✓ Actual dimensions: {actual_length_cm:.1f} × {actual_breadth_cm:.1f} × {estimated_height_cm:.1f} cm (L×B×H)"
                )
                print(
                    f"  Volume: {(actual_length_cm * actual_breadth_cm * estimated_height_cm):.1f} cm³"
                )
                print(f"  Confidence: {unknown_box['confidence']:.2%}")
                print(f"-" * 70 + "\n")

                calibrated_objects.append(
                    {
                        "filename": os.path.basename(image_path),
                        "object_id": unknown_box["index"],
                        "class_name": unknown_box["class_name"],
                        "confidence": round(unknown_box["confidence"], 3),
                        "length": round(actual_length_cm, 2),
                        "breadth": round(actual_breadth_cm, 2),
                        "height": round(estimated_height_cm, 2),
                        "volume": round(
                            actual_length_cm * actual_breadth_cm * estimated_height_cm,
                            2,
                        ),
                        "bbox_pixels": {
                            "x1": float(unknown_box["bbox"]["x1"]),
                            "y1": float(unknown_box["bbox"]["y1"]),
                            "x2": float(unknown_box["bbox"]["x2"]),
                            "y2": float(unknown_box["bbox"]["y2"]),
                        },
                        "calibration_method": "reference_box",
                        "pixels_per_cm": round(self.calibration_factor, 2),
                    }
                )

            # STEP 3: Save visualization with annotations
            if save_visualization:
                self._save_annotated_image(
                    image_path, reference_box, unknown_boxes, calibrated_objects
                )

            return {
                "image": os.path.basename(image_path),
                "reference_box": {
                    "index": reference_box["index"],
                    "known_dimensions": self.reference_dimensions,
                    "pixels": {
                        "width": round(reference_box["width_px"], 1),
                        "height": round(reference_box["height_px"], 1),
                    },
                },
                "calibration_factor": round(self.calibration_factor, 2),
                "calibrated_objects": calibrated_objects,
                "total_objects": len(calibrated_objects),
            }

        except Exception as e:
            print(f"\n❌ Error during calibration: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def _save_annotated_image(
        self,
        image_path: str,
        reference_box: Dict,
        unknown_boxes: List[Dict],
        calibrated_objects: List[Dict],
    ):
        """
        Save annotated image with reference box (GREEN) and unknown boxes (RED)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print("⚠ Could not read image for annotation")
                return

            # Draw reference box in GREEN
            ref_bbox = reference_box["bbox"]
            cv2.rectangle(
                img,
                (int(ref_bbox["x1"]), int(ref_bbox["y1"])),
                (int(ref_bbox["x2"]), int(ref_bbox["y2"])),
                (0, 255, 0),  # GREEN
                3,
            )

            # Add label for reference box
            ref_label = f"REFERENCE: {self.reference_dimensions['length']}x{self.reference_dimensions['breadth']} cm"
            cv2.putText(
                img,
                ref_label,
                (int(ref_bbox["x1"]), int(ref_bbox["y1"]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw unknown boxes in RED with actual dimensions
            for idx, (unknown_box, calibrated_obj) in enumerate(
                zip(unknown_boxes, calibrated_objects)
            ):
                bbox = unknown_box["bbox"]
                cv2.rectangle(
                    img,
                    (int(bbox["x1"]), int(bbox["y1"])),
                    (int(bbox["x2"]), int(bbox["y2"])),
                    (0, 0, 255),  # RED
                    3,
                )

                # Add label with actual dimensions
                obj_label = f"OBJECT #{idx+1}: {calibrated_obj['length']}x{calibrated_obj['breadth']}x{calibrated_obj['height']} cm"
                cv2.putText(
                    img,
                    obj_label,
                    (int(bbox["x1"]), int(bbox["y1"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Save annotated image
            output_dir = "output/annotated_calibrated"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"calibrated_{os.path.basename(image_path)}"
            )
            cv2.imwrite(output_path, img)
            print(f"✓ Annotated image saved: {output_path}")

        except Exception as e:
            print(f"⚠ Could not save annotated image: {e}")

    def _manual_calibration(self, image_path: str) -> Dict:
        """
        Fallback manual calibration using OpenCV contour detection
        When YOLO model is not available
        """
        print(f"\nUsing manual calibration for: {os.path.basename(image_path)}")

        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")

            # Convert to grayscale and detect contours
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) < 2:
                print("⚠ Need at least 2 contours (reference + unknown)")
                return None

            # Get bounding rectangles for all contours
            boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter out noise
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append({"x": x, "y": y, "w": w, "h": h, "area": area})

            # Sort by area - smallest is reference
            boxes.sort(key=lambda x: x["area"])

            if len(boxes) < 2:
                print("⚠ Need at least 2 valid boxes")
                return None

            reference = boxes[0]
            unknown = boxes[1]

            # Calculate calibration factor
            ref_width_cm = self.reference_dimensions["length"]
            self.calibration_factor = reference["w"] / ref_width_cm

            # Calculate unknown dimensions
            actual_length = unknown["w"] / self.calibration_factor
            actual_breadth = unknown["h"] / self.calibration_factor
            actual_height = self.reference_dimensions["height"]  # Assume same height

            print(f"  ✓ Reference: {reference['w']}×{reference['h']} px")
            print(f"  ✓ Unknown: {actual_length:.1f}×{actual_breadth:.1f} cm")

            return {
                "image": os.path.basename(image_path),
                "calibration_factor": round(self.calibration_factor, 2),
                "calibrated_objects": [
                    {
                        "filename": os.path.basename(image_path),
                        "object_id": 0,
                        "length": round(actual_length, 2),
                        "breadth": round(actual_breadth, 2),
                        "height": round(actual_height, 2),
                        "volume": round(
                            actual_length * actual_breadth * actual_height, 2
                        ),
                        "calibration_method": "manual_contour",
                    }
                ],
            }

        except Exception as e:
            print(f"  ⚠ Manual calibration failed: {e}")
            return None

    def process_calibration_batch(
        self, image_folder: str, save_visualizations: bool = True
    ) -> List[Dict]:
        """
        Process multiple images with reference-based calibration

        Each image should contain:
        - 1 reference box (smaller, with known dimensions)
        - 1+ unknown boxes (larger, to be measured)

        Args:
            image_folder: Path to folder containing images
            save_visualizations: Whether to save annotated images

        Returns:
            List of all calibrated objects from all images
        """
        all_calibrated_objects = []

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
        print(f"BATCH CALIBRATION MODE")
        print(f"{'=' * 70}")
        print(f"Found {len(image_files)} images to process")
        print(f"Reference dimensions: {self.reference_dimensions}")
        print(f"{'=' * 70}\n")

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)

            try:
                result = self.detect_and_calibrate(img_path, save_visualizations)

                if result and "calibrated_objects" in result:
                    all_calibrated_objects.extend(result["calibrated_objects"])
                    print(f"✓ Successfully calibrated: {img_file}\n")
                else:
                    print(f"✗ Calibration failed for: {img_file}\n")

            except Exception as e:
                print(f"✗ Error processing {img_file}: {str(e)}\n")

        print(f"\n{'=' * 70}")
        print(
            f"Successfully calibrated {len(all_calibrated_objects)} objects from {len(image_files)} images"
        )
        print(f"{'=' * 70}\n")

        return all_calibrated_objects

    def export_calibration_results(self, results: List[Dict], output_path: str):
        """Export calibration results to JSON"""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✓ Calibration results exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Setup: Define your reference box dimensions (the smaller box you'll place in images)
    reference_dims = {
        "length": 10.0,  # cm
        "breadth": 10.0,  # cm
        "height": 10.0,  # cm
    }

    # Initialize calibrator
    calibrator = DimensionCalibrator(
        model_path="models/best (3).pt",
        reference_dimensions=reference_dims,
        confidence_threshold=0.5,
    )

    # Process single image with reference box
    # result = calibrator.detect_and_calibrate('input/images/test_with_reference.jpg')

    # Process batch of images
    # all_objects = calibrator.process_calibration_batch('input/images/')

    # Export results
    # calibrator.export_calibration_results(all_objects, 'output/calibrated_dimensions.json')
