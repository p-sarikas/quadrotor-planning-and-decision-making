
from typing import Optional, Sequence, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from CONSTANTS import MAP_HEIGHT, MAP_WIDTH


def draw_tool(nodes_list: Optional[Sequence[Any]] = None,
                   goal_list: Optional[Sequence[Any]] = None,
                   path_nodes: Optional[Sequence[Any]] = None,
                   positions: Optional[Sequence[Any]] = None,
                   walls: Optional[Sequence[Any]] = None,
                   map_width: float = MAP_WIDTH,
                   map_height: float = MAP_HEIGHT,
                   ax: Optional[plt.Axes] = None,
                   grid: bool = True,
                   show: bool = True,
                   save: Optional[str] = None,
                   draw_prm_edges: bool = False,        
                   dedup_prm_edges: bool = True,
                   draw_solution_edges: bool = True)-> plt.Axes:
    """Draw a simple map showing walls, nodes, path nodes and a trajectory.

    Parameters are permissive: pass Node-like objects (with `.position`) or
    plain (x,y) pairs. Any of the collections may be None. Returns the
    matplotlib `Axes` used.
    """
    if ax is None:
        fig, ax = plt.subplots()

    nodes_list = nodes_list or []
    goal_list = goal_list or []
    path_nodes = path_nodes or []
    positions = positions or []
    walls = walls or []


    Z_GRID = 0
    Z_PRM_EDGES = 1
    Z_NODES = 2
    Z_PRM_PATH_EDGES = 3
    Z_PATH_NODES = 4
    Z_GOALS = 5
    Z_TRAJ = 6
    Z_OBSTACLES = 10 


    # set bounds and optional grid
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_aspect('equal')
    if grid:
        ax.set_xticks(range(int(map_width) + 1))
        ax.set_yticks(range(int(map_height) + 1))
        ax.grid(True, linestyle=':', linewidth=0.5, zorder = Z_GRID)


    if draw_prm_edges and nodes_list:
            
            node_dict = {n.id: n for n in nodes_list if hasattr(n, "id")}

            drawn = set()  # dedup undirected edges

            for n in nodes_list:
                if not hasattr(n, "connections") or not hasattr(n, "id"):
                    continue

                p = np.asarray(n.position)
                x0, y0 = float(p[0]), float(p[1])

                for e in n.connections:
                    nb_id = getattr(e, "second_node", None)
                    if nb_id is None:
                        continue

                    if dedup_prm_edges:
                        key = tuple(sorted((n.id, nb_id)))
                        if key in drawn:
                            continue
                        drawn.add(key)

                    nb = node_dict.get(nb_id)
                    if nb is None:
                        continue

                    q = np.asarray(nb.position)
                    x1, y1 = float(q[0]), float(q[1])

                    ax.plot(
                        [x0, x1], [y0, y1],
                        color="blue",
                        alpha=0.12,
                        linewidth=0.7,
                        zorder=Z_PRM_EDGES
                    )


    # nodes (blue points)
    x_nodes, y_nodes = [], []
    for n in nodes_list:
        try:
            p = getattr(n, 'position', n)
            p = np.asarray(p)
            x_nodes.append(float(p[0])); y_nodes.append(float(p[1]))
        except Exception:
            continue
    if x_nodes:
        ax.scatter(x_nodes, y_nodes, c='blue',alpha=0.3, s=10, zorder=Z_NODES)


    # goals (green stars)
    x_goals, y_goals = [], []
    for g in goal_list:
        try:
            gg = np.asarray(g)
            x_goals.append(float(gg[0])); y_goals.append(float(gg[1]))
        except Exception:
            continue
    if x_goals:
        ax.scatter(x_goals, y_goals, c='green', marker='*', s=60, zorder=Z_GOALS)



    # path nodes (black points)
    x_path, y_path = [], []
    path_positions_2d = []
    for n in path_nodes:
        try:
            p = getattr(n, 'position', n)
            p = np.asarray(p)
            x_path.append(float(p[0])); y_path.append(float(p[1]))
            path_positions_2d.append((float(p[0]), float(p[1])))
        except Exception:
            continue

    if draw_solution_edges and len(path_positions_2d) >= 2:
        for (x0, y0), (x1, y1) in zip(path_positions_2d[:-1], path_positions_2d[1:]):
            ax.plot(
                [x0, x1], [y0, y1],
                color="black",
                linewidth=2.0,
                alpha=0.9,
                zorder=Z_PRM_PATH_EDGES
            )

    if x_path:
        ax.scatter(x_path, y_path, c='black', s=20, zorder=Z_PATH_NODES)


    # positions/trajectory (orange line)
    x_pos, y_pos = [], []
    for pt in positions:
        try:
            p = np.asarray(getattr(pt, 'position', pt))
            x_pos.append(float(p[0])); y_pos.append(float(p[1]))
        except Exception:
            continue
    if x_pos:
        ax.plot(x_pos, y_pos, c='orange', linewidth=2, zorder=Z_TRAJ)

    # draw walls (objects with xmin,xmax,ymin,ymax)
    for w in walls:
        try:
            xmin, xmax = float(w.xmin), float(w.xmax)
            ymin, ymax = float(w.ymin), float(w.ymax)
        except Exception:
            continue
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=w.color, zorder=Z_OBSTACLES))

    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    return ax


__all__ = ["draw_tool"]
