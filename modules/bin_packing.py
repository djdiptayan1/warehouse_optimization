"""
3D Bin Packing Algorithm
Implements efficient placement strategy with rotation and stacking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class Position:
    """Represents a 3D position in the warehouse"""

    x: float
    y: float
    z: float


@dataclass
class Box:
    """Represents an object/box to be placed"""

    id: int
    filename: str
    length: float
    breadth: float
    height: float
    volume: float
    position: Optional[Position] = None
    rotation: int = 0  # 0, 90, 180, 270 degrees

    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get dimensions based on rotation (only horizontal rotation)"""
        if self.rotation in [90, 270]:
            return (self.breadth, self.length, self.height)
        return (self.length, self.breadth, self.height)

    def get_rotations(self) -> List[Tuple[float, float, float]]:
        """Get all possible orientations"""
        # 4 horizontal rotations
        return [
            (self.length, self.breadth, self.height),  # 0°
            (self.breadth, self.length, self.height),  # 90°
            (
                self.length,
                self.breadth,
                self.height,
            ),  # 180° (same as 0° for rectangular)
            (self.breadth, self.length, self.height),  # 270° (same as 90°)
        ]


@dataclass
class Warehouse:
    """Represents the warehouse space"""

    length: float
    breadth: float
    height: float
    aisle_width: float = 100.0  # cm - space for access paths

    def get_volume(self) -> float:
        return self.length * self.breadth * self.height


class BinPacking3D:
    """3D Bin Packing with rotation and stacking support"""

    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.placed_boxes: List[Box] = []
        self.failed_boxes: List[Box] = []

        # Create occupancy grid for efficient collision detection
        # Using 10cm resolution for the grid
        self.grid_resolution = 10  # cm
        self.grid_x = int(np.ceil(warehouse.length / self.grid_resolution))
        self.grid_y = int(np.ceil(warehouse.breadth / self.grid_resolution))
        self.grid_z = int(np.ceil(warehouse.height / self.grid_resolution))
        self.occupancy_grid = np.zeros(
            (self.grid_x, self.grid_y, self.grid_z), dtype=bool
        )

    def pack_boxes(self, boxes: List[Dict]) -> Dict:
        """
        Main packing algorithm

        Args:
            boxes: List of box dictionaries with dimensions

        Returns:
            Dictionary with placement results and statistics
        """
        # Convert dictionaries to Box objects
        box_objects = [
            Box(
                id=i,
                filename=box["filename"],
                length=box["length"],
                breadth=box["breadth"],
                height=box["height"],
                volume=box["volume"],
            )
            for i, box in enumerate(boxes)
        ]

        # Sort boxes by volume (largest first) - greedy heuristic
        box_objects.sort(key=lambda b: b.volume, reverse=True)

        print(f"\nPacking {len(box_objects)} boxes into warehouse...")
        print(
            f"Warehouse: {self.warehouse.length} x {self.warehouse.breadth} x {self.warehouse.height} cm"
        )
        print("=" * 60)

        # Try to place each box
        for box in box_objects:
            placed = self._place_box(box)
            if placed:
                self.placed_boxes.append(box)
                print(
                    f"✓ Placed: {box.filename} at ({box.position.x:.1f}, {box.position.y:.1f}, {box.position.z:.1f})"
                )
            else:
                self.failed_boxes.append(box)
                print(f"✗ Failed: {box.filename} - No suitable position found")

        return self._generate_results()

    def _place_box(self, box: Box) -> bool:
        """
        Try to place a box in the warehouse

        Returns:
            True if placement successful, False otherwise
        """
        # Try different rotations
        for rotation_idx in range(2):  # Only need 2 rotations (0° and 90°)
            box.rotation = rotation_idx * 90
            dims = box.get_dimensions()

            # Try to find a valid position
            position = self._find_position(dims)
            if position:
                box.position = position
                self._mark_occupied(box)
                return True

        return False

    def _find_position(
        self, dimensions: Tuple[float, float, float]
    ) -> Optional[Position]:
        """
        Find a valid position for a box with given dimensions
        Uses a layer-by-layer approach with aisle consideration
        """
        length, breadth, height = dimensions

        # Account for aisles - leave space between rows
        aisle_space = self.warehouse.aisle_width

        # Try to place at different heights (z-levels)
        for z in np.arange(0, self.warehouse.height - height + 1, self.grid_resolution):
            # Try different y positions
            for y in np.arange(
                0, self.warehouse.breadth - breadth + 1, self.grid_resolution
            ):
                # Skip some y-positions to create aisles
                if int(y / aisle_space) % 2 == 1:
                    continue

                # Try different x positions
                for x in np.arange(
                    0, self.warehouse.length - length + 1, self.grid_resolution
                ):
                    if self._is_position_valid(x, y, z, length, breadth, height):
                        return Position(x, y, z)

        return None

    def _is_position_valid(
        self, x: float, y: float, z: float, length: float, breadth: float, height: float
    ) -> bool:
        """
        Check if a position is valid (no collisions and has support)
        """
        # Check bounds
        if (
            x + length > self.warehouse.length
            or y + breadth > self.warehouse.breadth
            or z + height > self.warehouse.height
        ):
            return False

        # Convert to grid coordinates
        gx1 = int(x / self.grid_resolution)
        gy1 = int(y / self.grid_resolution)
        gz1 = int(z / self.grid_resolution)
        gx2 = int(np.ceil((x + length) / self.grid_resolution))
        gy2 = int(np.ceil((y + breadth) / self.grid_resolution))
        gz2 = int(np.ceil((z + height) / self.grid_resolution))

        # Ensure within grid bounds
        gx2 = min(gx2, self.grid_x)
        gy2 = min(gy2, self.grid_y)
        gz2 = min(gz2, self.grid_z)

        # Check for collision
        if np.any(self.occupancy_grid[gx1:gx2, gy1:gy2, gz1:gz2]):
            return False

        # Check for support (if not on ground)
        if z > 0:
            # At least 80% of the bottom should be supported
            support_area = 0
            total_area = (gx2 - gx1) * (gy2 - gy1)

            if gz1 > 0:
                support_layer = self.occupancy_grid[gx1:gx2, gy1:gy2, gz1 - 1]
                support_area = np.sum(support_layer)

                if support_area < 0.8 * total_area:
                    return False

        return True

    def _mark_occupied(self, box: Box):
        """Mark the space occupied by a box in the grid"""
        dims = box.get_dimensions()
        x, y, z = box.position.x, box.position.y, box.position.z
        length, breadth, height = dims

        gx1 = int(x / self.grid_resolution)
        gy1 = int(y / self.grid_resolution)
        gz1 = int(z / self.grid_resolution)
        gx2 = int(np.ceil((x + length) / self.grid_resolution))
        gy2 = int(np.ceil((y + breadth) / self.grid_resolution))
        gz2 = int(np.ceil((z + height) / self.grid_resolution))

        # Ensure within grid bounds
        gx2 = min(gx2, self.grid_x)
        gy2 = min(gy2, self.grid_y)
        gz2 = min(gz2, self.grid_z)

        self.occupancy_grid[gx1:gx2, gy1:gy2, gz1:gz2] = True

    def _generate_results(self) -> Dict:
        """Generate placement results and statistics"""
        total_boxes = len(self.placed_boxes) + len(self.failed_boxes)
        placed_count = len(self.placed_boxes)

        # Calculate volumes
        warehouse_volume = self.warehouse.get_volume()
        used_volume = sum(box.volume for box in self.placed_boxes)

        # Calculate space utilization
        utilization = (used_volume / warehouse_volume) * 100

        # Prepare coordinates list
        coordinates = []
        for box in self.placed_boxes:
            dims = box.get_dimensions()
            coordinates.append(
                {
                    "id": box.id,
                    "filename": box.filename,
                    "x": round(box.position.x, 2),
                    "y": round(box.position.y, 2),
                    "z": round(box.position.z, 2),
                    "length": round(dims[0], 2),
                    "breadth": round(dims[1], 2),
                    "height": round(dims[2], 2),
                    "rotation": box.rotation,
                }
            )

        statistics = {
            "total_boxes": total_boxes,
            "placed_boxes": placed_count,
            "failed_boxes": len(self.failed_boxes),
            "placement_rate": (
                round((placed_count / total_boxes) * 100, 2) if total_boxes > 0 else 0
            ),
            "warehouse_volume": round(warehouse_volume, 2),
            "used_volume": round(used_volume, 2),
            "wasted_volume": round(warehouse_volume - used_volume, 2),
            "space_utilization": round(utilization, 2),
        }

        return {
            "coordinates": coordinates,
            "statistics": statistics,
            "failed_boxes": [
                {"id": b.id, "filename": b.filename} for b in self.failed_boxes
            ],
        }
