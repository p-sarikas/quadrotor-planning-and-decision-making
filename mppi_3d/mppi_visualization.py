import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from environmentBuilder.getWalls import getLayout
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mppi_position(global_ref, traj, N, K, walls):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    
    plot_walls(ax, walls)     # Walls

    ax.plot(global_ref[:, 0], global_ref[:, 1], linewidth=3, label="Global reference")
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label="MPPI executed")
    ax.scatter(global_ref[0, 0], global_ref[0, 1], s=120, marker="X", label="Start")
    ax.scatter(global_ref[-1, 0], global_ref[-1, 1], s=120, marker="X", label="Goal")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"MPPI (Horizon={N} & Rollouts={K}")
    plt.show()


def plot_mppi_position_3d(global_ref, traj, N, K, walls):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(global_ref[:, 0], global_ref[:, 1], global_ref[:, 2], linewidth=3, label="Global reference")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="MPPI executed")
    ax.scatter(global_ref[0, 0], global_ref[0, 1], global_ref[0, 2], s=80, marker="X", label="Start")
    ax.scatter(global_ref[-1, 0], global_ref[-1, 1], global_ref[-1, 2], s=80, marker="X", label="Goal")

    plot_walls_3d(ax, walls)     # Walls

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"MPPI 3D View (Horizon={N}, Rollouts={K})")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_mppi_orientation(global_ref, traj, yaw_hist, N, K, walls, step=10, scale=0.4):
    """
    Second plot: quadrotor orientation (yaw) along the executed trajectory.

    traj     : (T,3) executed positions
    yaw_hist : (T,) yaw angle history [rad]
    step     : draw one arrow every 'step' timesteps
    scale    : arrow length
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)

    plot_walls(ax, walls)

    # Plot executed trajectory
    ax.plot(global_ref[:, 0], global_ref[:, 1], linewidth=3, label="Global reference")
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label="MPPI executed")
    ax.scatter(global_ref[0, 0], global_ref[0, 1], s=120, marker="X", label="Start")
    ax.scatter(global_ref[-1, 0], global_ref[-1, 1], s=120, marker="X", label="Goal")

    # Plot yaw orientation arrows
    for i in range(0, len(traj), step):
        x, y = traj[i, 0], traj[i, 1]
        psi = yaw_hist[i]

        dx = scale * np.cos(psi)
        dy = scale * np.sin(psi)

        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.15, fc="tab:orange", ec="tab:orange", alpha=0.9)

    ax.grid(True)
    ax.legend()
    ax.set_title(f"Quadrotor orientation (yaw) along trajectory \n with Horizon={N} & Rollouts={K}")
    plt.show()


def plot_mppi_orientation_3d(global_ref, traj, yaw_hist, N, K, walls, step=10, scale=0.4):
    """
    Second plot: quadrotor orientation (yaw) along the executed trajectory.

    traj     : (T,3) executed positions
    yaw_hist : (T,) yaw angle history [rad]
    step     : draw one arrow every 'step' timesteps
    scale    : arrow length
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(global_ref[:, 0], global_ref[:, 1], global_ref[:, 2], linewidth=3, label="Global reference")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="MPPI executed")
    ax.scatter(global_ref[0, 0], global_ref[0, 1], global_ref[0, 2], s=80, marker="X", label="Start")
    ax.scatter(global_ref[-1, 0], global_ref[-1, 1], global_ref[-1, 2], s=80, marker="X", label="Goal")

    plot_walls_3d(ax, walls)     # Walls

    # Plot yaw orientation arrows
    for i in range(0, len(traj), step):
        x, y, z, = traj[i, 0], traj[i, 1], traj[i,2]
        psi = yaw_hist[i]

        dx = scale * np.cos(psi)
        dy = scale * np.sin(psi)
        dz = 0

        ax.quiver(x, y, z, dx, dy, dz, color="tab:orange", linewidth=1.5, arrow_length_ratio=0.3)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_title(f"Quadrotor yaw in 3D \n with Horizon={N} & Rollouts={K}")
    plt.tight_layout()
    plt.show()



def plot_mppi_tracking_error(error_hist, dt):
    """
    Plot MPPI tracking error (global-local) in x, y, z over time.

    error_hist : (T,3) array
    dt         : timestep [s]
    """

    T = np.asarray(error_hist.shape[0])
    t = np.arange(T) * dt

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    labels = ["x error [m]", "y error [m]", "z error [m]"]

    for i in range(3):
        ax[i].plot(t, error_hist[:, i])
        ax[i].grid(True)
        ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("MPPI Tracking Error (Global-Local Path) vs Time")
    plt.tight_layout()
    plt.show()


def plot_walls(ax, walls):
    """
    Plot walls / obstacles on a matplotlib Axes

    Parameters
    ax: matplotlib.axes.Axes
        Axis to draw on
    walls : list
        List of Wall objects from getLayout()
    """

    for wall in walls:
        width  = wall.xmax - wall.xmin
        length = wall.ymax - wall.ymin
        height = wall.zmax - wall.zmin

        rect = Rectangle((wall.xmin, wall.ymin),
            width,
            length,
            facecolor=wall.rgbaColor[:3],
            alpha=wall.rgbaColor[3] if len(wall.rgbaColor) == 4 else 1.0,
            edgecolor="k",
            linewidth=1)

        ax.add_patch(rect)



def plot_walls_3d(ax, walls):
    """
    Plot walls as 3D boxes in (x, y, z).
    """
    for wall in walls:
        x0, x1 = wall.xmin, wall.xmax
        y0, y1 = wall.ymin, wall.ymax
        z0, z1 = wall.zmin, wall.zmax

        # 8 vertices of the box
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]

        # 6 faces (each face is a list of 4 vertices)
        faces = [
            [vertices[i] for i in [0,1,2,3]],  # bottom
            [vertices[i] for i in [4,5,6,7]],  # top
            [vertices[i] for i in [0,1,5,4]],
            [vertices[i] for i in [1,2,6,5]],
            [vertices[i] for i in [2,3,7,6]],
            [vertices[i] for i in [3,0,4,7]],
        ]

        box = Poly3DCollection(
            faces,
            facecolor=wall.rgbaColor[:3],
            alpha=wall.rgbaColor[3] if len(wall.rgbaColor) == 4 else 0.5,
            linewidths=0.5,
            edgecolors="k"
        )

        ax.add_collection3d(box)
