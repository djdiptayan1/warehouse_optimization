"""
3D Visualization Module
Creates interactive 3D visualizations of warehouse placement
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict
import random


class Visualizer3D:
    """Create 3D visualizations of warehouse packing"""

    def __init__(self):
        self.colors = self._generate_colors(50)

    def _generate_colors(self, n: int) -> List[str]:
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = i * 137.508  # Golden angle
            colors.append(f"hsl({hue % 360}, 70%, 60%)")
        return colors

    def visualize_placement(
        self,
        coordinates: List[Dict],
        warehouse_dims: Dict,
        output_path: str = "output/visualization.html",
    ):
        """
        Create interactive 3D visualization

        Args:
            coordinates: List of placed box coordinates
            warehouse_dims: Dictionary with warehouse dimensions
            output_path: Path to save HTML file
        """
        fig = go.Figure()

        # Add warehouse boundaries
        self._add_warehouse_boundaries(fig, warehouse_dims)

        # Add each box
        for i, box in enumerate(coordinates):
            self._add_box(fig, box, self.colors[i % len(self.colors)])

        # Update layout
        fig.update_layout(
            title={
                "text": f"Warehouse 3D Placement Visualization<br>"
                + f"<sub>Placed {len(coordinates)} objects</sub>",
                "x": 0.5,
                "xanchor": "center",
            },
            scene=dict(
                xaxis_title="Length (cm)",
                yaxis_title="Breadth (cm)",
                zaxis_title="Height (cm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            showlegend=True,
            width=1200,
            height=800,
            hovermode="closest",
        )

        # Save to HTML
        fig.write_html(output_path)
        print(f"\n3D Visualization saved to: {output_path}")

        return fig

    def _add_warehouse_boundaries(self, fig: go.Figure, dims: Dict):
        """Add warehouse boundary wireframe"""
        length, breadth, height = dims["length"], dims["breadth"], dims["height"]

        # Define edges of the warehouse
        edges = [
            # Bottom rectangle
            ([0, length], [0, 0], [0, 0]),
            ([length, length], [0, breadth], [0, 0]),
            ([length, 0], [breadth, breadth], [0, 0]),
            ([0, 0], [breadth, 0], [0, 0]),
            # Top rectangle
            ([0, length], [0, 0], [height, height]),
            ([length, length], [0, breadth], [height, height]),
            ([length, 0], [breadth, breadth], [height, height]),
            ([0, 0], [breadth, 0], [height, height]),
            # Vertical edges
            ([0, 0], [0, 0], [0, height]),
            ([length, length], [0, 0], [0, height]),
            ([length, length], [breadth, breadth], [0, height]),
            ([0, 0], [breadth, breadth], [0, height]),
        ]

        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=edge[0],
                    y=edge[1],
                    z=edge[2],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add floor grid
        self._add_floor_grid(fig, length, breadth)

    def _add_floor_grid(self, fig: go.Figure, length: float, breadth: float):
        """Add grid lines on the floor"""
        grid_spacing = max(length, breadth) / 10

        # X-direction lines
        for y in np.arange(0, breadth + grid_spacing, grid_spacing):
            fig.add_trace(
                go.Scatter3d(
                    x=[0, length],
                    y=[y, y],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Y-direction lines
        for x in np.arange(0, length + grid_spacing, grid_spacing):
            fig.add_trace(
                go.Scatter3d(
                    x=[x, x],
                    y=[0, breadth],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def _add_box(self, fig: go.Figure, box: Dict, color: str):
        """Add a single box to the visualization"""
        x, y, z = box["x"], box["y"], box["z"]
        l, b, h = box["length"], box["breadth"], box["height"]

        # Define vertices of the box
        vertices = np.array(
            [
                [x, y, z],
                [x + l, y, z],
                [x + l, y + b, z],
                [x, y + b, z],
                [x, y, z + h],
                [x + l, y, z + h],
                [x + l, y + b, z + h],
                [x, y + b, z + h],
            ]
        )

        # Define faces (triangles)
        faces = [
            [0, 1, 2],
            [0, 2, 3],  # Bottom
            [4, 5, 6],
            [4, 6, 7],  # Top
            [0, 1, 5],
            [0, 5, 4],  # Front
            [2, 3, 7],
            [2, 7, 6],  # Back
            [0, 3, 7],
            [0, 7, 4],  # Left
            [1, 2, 6],
            [1, 6, 5],  # Right
        ]

        # Extract coordinates for mesh
        i, j, k = [], [], []
        for face in faces:
            i.append(face[0])
            j.append(face[1])
            k.append(face[2])

        # Hover text
        hover_text = (
            f"<b>{box['filename']}</b><br>"
            + f"Position: ({x:.1f}, {y:.1f}, {z:.1f})<br>"
            + f"Dimensions: {l:.1f} × {b:.1f} × {h:.1f} cm<br>"
            + f"Rotation: {box['rotation']}°"
        )

        # Add mesh
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=0.7,
                name=box["filename"],
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=True,
            )
        )

        # Add edges for better visibility
        self._add_box_edges(fig, vertices, color)

    def _add_box_edges(self, fig: go.Figure, vertices: np.ndarray, color: str):
        """Add edges to a box for better visibility"""
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Bottom
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Top
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # Vertical
        ]

        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[vertices[edge[0], 0], vertices[edge[1], 0]],
                    y=[vertices[edge[0], 1], vertices[edge[1], 1]],
                    z=[vertices[edge[0], 2], vertices[edge[1], 2]],
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def create_statistics_visualization(
        self, statistics: Dict, output_path: str = "output/statistics.html"
    ):
        """Create visualization of statistics"""
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Placement Success Rate",
                "Space Utilization",
                "Volume Distribution",
                "Placement Summary",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "pie"}, {"type": "bar"}],
            ],
        )

        # Placement rate indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=statistics["placement_rate"],
                title={"text": "Placement Rate (%)"},
                delta={"reference": 100},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # Space utilization indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=statistics["space_utilization"],
                title={"text": "Space Utilization (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 40], "color": "lightgray"},
                        {"range": [40, 70], "color": "gray"},
                    ],
                },
            ),
            row=1,
            col=2,
        )

        # Volume pie chart
        fig.add_trace(
            go.Pie(
                labels=["Used Volume", "Wasted Volume"],
                values=[statistics["used_volume"], statistics["wasted_volume"]],
                marker=dict(colors=["#2ecc71", "#e74c3c"]),
            ),
            row=2,
            col=1,
        )

        # Placement summary bar chart
        fig.add_trace(
            go.Bar(
                x=["Placed", "Failed"],
                y=[statistics["placed_boxes"], statistics["failed_boxes"]],
                marker=dict(color=["#2ecc71", "#e74c3c"]),
                text=[statistics["placed_boxes"], statistics["failed_boxes"]],
                textposition="auto",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Warehouse Placement Statistics Dashboard",
            showlegend=False,
            height=800,
        )

        fig.write_html(output_path)
        print(f"Statistics visualization saved to: {output_path}")

        return fig
