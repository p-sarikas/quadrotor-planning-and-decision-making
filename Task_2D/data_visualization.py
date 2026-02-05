import numpy as np
import os
from Task_2D.data_handling import load_data_npz
import matplotlib.pyplot as plt

from CONSTANTS import PATH_LENGTH_EASY, PATH_LENGTH_MEDIUM, PATH_LENGTH_DIFFICULT
from matplotlib.ticker import MultipleLocator


def set_ax2_ylim_align(ax1, ax2, y1_target=60, y2_target=1.0, y2_min=0.0, pad=1.05):
    y1_min, y1_max = ax1.get_ylim()
    frac = (y1_target - y1_min) / (y1_max - y1_min)

    # Solve y2_max so that y2_target aligns with y1_target
    y2_max = y2_min + (y2_target - y2_min) / frac

    # Make sure we still include all data
    # (ax2 data limits are known after plotting)
    data_max = ax2.dataLim.y1
    y2_max = max(y2_max, data_max * pad)

    ax2.set_ylim(y2_min, y2_max)

    # Keep "nice" ticks like 1.0, 1.2, 1.4 ...
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))


def create_figure(max_nodes, prm_pct, grid_pct, prm_length, grid_length, prm_yerr, opt_path_length, file_path=None):
    
    x = np.arange(len(max_nodes))
    


    prm_arr  = np.asarray(prm_length, dtype=float)
    grid_arr = np.asarray(grid_length, dtype=float)

    max_len = np.nanmax(np.concatenate([prm_arr, grid_arr]))
    if not np.isfinite(max_len) or max_len == 0:
        max_len = 1.0

    max_y = (max_len / opt_path_length) * 1.05

    prm_len_norm  = prm_arr  / opt_path_length
    grid_len_norm = grid_arr / opt_path_length
      

    fig, ax1 = plt.subplots()
    # Bars: checkpoints reached (%)
    width = 0.25
    ax1.bar(x - width/2, prm_pct,  width, yerr=prm_yerr, capsize=3,
             edgecolor="black", linewidth=1, alpha=0.4, label="PRM checkpoints",
             error_kw={"ecolor": "grey", "elinewidth": 1.0, "capthick": 1.0})
    ax1.bar(x + width/2, grid_pct, width, edgecolor="black", linewidth=1, alpha=0.4, label="Grid checkpoints")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in max_nodes])
    ax1.set_xlabel("Nodes")
    ax1.set_ylabel("Checkpoints reached (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", alpha=0.3)

    # Line plot on second axis: normalized path length
    ax2 = ax1.twinx()
    ax2.plot(x, prm_len_norm, marker="o", label="PRM path length")
    ax2.plot(x, grid_len_norm, marker="o", label="Grid path length")
    ax2.set_ylabel("Path length (normalized)")
    #ax2.set_ylim(0, max_y)

    set_ax2_ylim_align(ax1, ax2, y1_target=60, y2_target=1.0)
    # Combined legend (from both axes)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    #plt.title(f"Graph Creation: {env}")
    if file_path is None:
        plt.show()
    else:
        fig.savefig(file_path, bbox_inches="tight")



base_path = os.path.join("Task_2D", "exp_data", "graph_creation")
# env = "easy"
# save_path = os.path.join(base_path, env)
# file_name = "data_graph_easy.npz"
# file_path = os.path.join(save_path, file_name)

# chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes = load_data_npz(file_path)
# chpts_prm_mins = np.array([np.min(run_prm) for run_prm in chpts_prm_raw])
# chpts_prm_maxs = np.array([np.max(run_prm) for run_prm in chpts_prm_raw])
# prm_yerr = np.vstack([chpts_prm_avgs - chpts_prm_mins, chpts_prm_maxs - chpts_prm_avgs])
# plot_dir = "Task_2D/plots"
# opt_path_length = PATH_LENGTH_EASY
# plot_path = os.path.join(plot_dir, "Graph_Creation_Easy_normalized.pdf")
# create_figure(max_nodes, chpts_prm_avgs, chpts_grid_avgs, path_prm_avgs, path_grid_avgs, prm_yerr, opt_path_length, file_path=plot_path)
# print(chpts_prm_raw)


env = "medium"
save_path = os.path.join(base_path, env)
file_name = "data_graph_medium_v2.npz"
file_path = os.path.join(save_path, file_name)

chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes = load_data_npz(file_path)
chpts_prm_mins = np.array([np.min(run_prm) for run_prm in chpts_prm_raw])
chpts_prm_maxs = np.array([np.max(run_prm) for run_prm in chpts_prm_raw])
prm_yerr = np.vstack([chpts_prm_avgs - chpts_prm_mins, chpts_prm_maxs - chpts_prm_avgs])
plot_dir = "Task_2D/plots"
plot_path = os.path.join(plot_dir, "Graph_Creation_Medium_normalized.pdf")
opt_path_length = PATH_LENGTH_MEDIUM
create_figure(max_nodes, chpts_prm_avgs, chpts_grid_avgs, path_prm_avgs, path_grid_avgs, prm_yerr, opt_path_length, file_path=plot_path)
print(chpts_prm_raw)


# env = "difficult"
# save_path = os.path.join(base_path, env)
# file_name = "data_graph_difficult_v2.npz"
# file_path = os.path.join(save_path, file_name)

# chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes = load_data_npz(file_path)
# chpts_prm_mins = np.array([np.min(run_prm) for run_prm in chpts_prm_raw])
# chpts_prm_maxs = np.array([np.max(run_prm) for run_prm in chpts_prm_raw])
# prm_yerr = np.vstack([chpts_prm_avgs - chpts_prm_mins, chpts_prm_maxs - chpts_prm_avgs])
# plot_dir = "Task_2D/plots"
# plot_path = os.path.join(plot_dir, "Graph_Creation_Difficult_normalized.pdf")
# opt_path_length = PATH_LENGTH_DIFFICULT
# create_figure(max_nodes, chpts_prm_avgs, chpts_grid_avgs, path_prm_avgs, path_grid_avgs, prm_yerr, opt_path_length, file_path=plot_path)
# print(chpts_prm_raw)



