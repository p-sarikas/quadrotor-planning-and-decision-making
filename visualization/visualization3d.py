from typing import Optional, Sequence, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from CONSTANTS import MAP_WIDTH, MAP_HEIGHT, MAP_LENGTH


def _get_xyz(item: Any) -> Optional[Tuple[float, float, float]]:
    """Return (x,y,z) from a Node-like object or a sequence. Missing z -> 0.

    Returns None when the item cannot be interpreted as a 2/3-element point.
    This makes the 3D drawing tolerant to noisy inputs (mirrors 2D behavior).
    """
    try:
        p = getattr(item, 'position', item)
        arr = np.asarray(p)
        if arr.size < 2:
            return None
        x = float(arr[0]); y = float(arr[1]); z = float(arr[2]) if arr.size > 2 else 0.0
        return x, y, z
    except Exception:
        return None


def plot_walls_3d(ax: plt.Axes, walls: Sequence[Any]):
    """Plot walls as 3D boxes on the provided Axes3D."""
    for wall in walls:
        x0, x1 = float(wall.xmin), float(wall.xmax)
        y0, y1 = float(wall.ymin), float(wall.ymax)
        z0, z1 = float(getattr(wall, 'zmin', 0.0)), float(getattr(wall, 'zmax', 1.0))

        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]

        faces = [
            [vertices[i] for i in [0,1,2,3]],
            [vertices[i] for i in [4,5,6,7]],
            [vertices[i] for i in [0,1,5,4]],
            [vertices[i] for i in [1,2,6,5]],
            [vertices[i] for i in [2,3,7,6]],
            [vertices[i] for i in [3,0,4,7]],
        ]

        box = Poly3DCollection(
            faces,
            facecolor=wall.rgbaColor[:3],
            alpha=1,
            linewidths=0.5,
            edgecolors="k"
        )

        ax.add_collection3d(box)


def draw_tool_3d(nodes_list: Optional[Sequence[Any]] = None,
              goal_list: Optional[Sequence[Any]] = None,
              path_nodes: Optional[Sequence[Any]] = None,
              positions: Optional[Sequence[Any]] = None,
              walls: Optional[Sequence[Any]] = None,
              xlim: Tuple[float, float] = (0, MAP_WIDTH),
              ylim: Tuple[float, float] = (0, MAP_HEIGHT),
              zlim: Tuple[float, float] = (0, MAP_LENGTH),
              ax: Optional[plt.Axes] = None,
              grid: bool = False,
              show: bool = True,
              save: Optional[str] = None) -> plt.Axes:
    """3D-aware version of `draw_tool`.

    Mirrors the 2D `draw_tool` API but plots in 3D. All inputs accept
    Node-like objects (with `.position`) or plain sequences. Missing z
    coordinates are treated as 0.
    """
    nodes_list = nodes_list or []
    goal_list = goal_list or []
    path_nodes = path_nodes or []
    positions = positions or []
    walls = walls or []

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # walls
    if walls:
        plot_walls_3d(ax, walls)

    # nodes (blue)
    if nodes_list:
        pts = [_get_xyz(n) for n in nodes_list]
        pts = [p for p in pts if p is not None]
        if pts:
            xs, ys, zs = zip(*pts)
            ax.scatter(xs, ys, zs, c='blue', s=10, depthshade=True)

    # goals (green X)
    if goal_list:
        pts = [_get_xyz(g) for g in goal_list]
        pts = [p for p in pts if p is not None]
        if pts:
            xs, ys, zs = zip(*pts)
            ax.scatter(xs, ys, zs, c='green', marker='X', s=60)

    # path nodes (black)
    if path_nodes:
        pts = [_get_xyz(n) for n in path_nodes]
        pts = [p for p in pts if p is not None]
        if pts:
            xs, ys, zs = zip(*pts)
            ax.scatter(xs, ys, zs, c='black', s=20)

    # trajectory (orange line)
    if positions:
        pts = [_get_xyz(p) for p in positions]
        pts = [p for p in pts if p is not None]
        if pts:
            xs, ys, zs = zip(*pts)
            ax.plot(xs, ys, zs, c='orange', linewidth=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    return ax


__all__ = ["draw_tool_3d", "plot_walls_3d"]
