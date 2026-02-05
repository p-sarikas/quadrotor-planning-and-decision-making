import numpy as np
from prm.prm_algorithm_v2 import add_checkpoint
from aStar.aStar import aStarAlgo
from environmentBuilder.getWalls import getLayout
import matplotlib.pyplot as plt
from CONSTANTS import EASY_LOCATIONS, MEDIUM_LOCATIONS, HARD_LOCATIONS
from Task_2D.task_wrapper import Task, draw_tool
from mppi_2d.mppi_main import run_mppi
from Task_2D.data_handling import save_mppi_results_npz
# task
list_of_checkpoints=HARD_LOCATIONS
task_mppi_v1 = Task(list_of_checkpoints=list_of_checkpoints)
task_mppi_v1.list_of_checkpoints = list_of_checkpoints #list of position with position as np.array(3,)
task_mppi_v1.graph_creator = "Grid" #"PRM"Grid
task_mppi_v1.path_finder = "aStar"
task_mppi_v1.global_layout_id = "difficult" #"easy" #"difficult"
task_mppi_v1.local_layout_id = "sparse" #"sparse" #"crowded"
visualize=False

seed_list = [10, 14] #[4, 7, 10, 14, 17]
N_list = [11, 17] #[15, 30]
K_list = [20, 50] # [50, 200, 500, 1000, 2000] or [200, 500, 1000, 2000]]


# ============================================================================================
# ============================================================================================


def compute_path_nodes(nodes_list, checkpoints, walls, graph_search_type="aStar", r=1.0):
    """
    Connect checkpoints to a graph and compute a global path.
    Returns: path_nodes: List of Node objects forming the full global path
    """
    path_nodes = []
    start_position = checkpoints[0]
    nodes_list, connected, start_id = add_checkpoint(nodes_list,
        checkpoint_position=start_position, walls=walls, r=r)

    if not connected:
        print("Couldn't connect start node")

    # Iterate over checkpoint segments ---
    for i in range(len(checkpoints) - 1):

        goal_position = checkpoints[i + 1]

        # Add checkpoint: next goal
        nodes_list, connected, goal_id = add_checkpoint(nodes_list,
        checkpoint_position=goal_position,  walls=walls, r=r)

        if not connected:
            print("Couldn't connect next checkpoint")
            break

        ## Get path for next segment (segment meaning between two checkpoints)
        new_path_nodes, _ = aStarAlgo(start_id,
            goal_id, nodes_list, graph_search_type=graph_search_type)
        if i > 0:
            new_path_nodes = new_path_nodes[1:]
        path_nodes.extend(new_path_nodes)
        # update start for next segment
        start_id = goal_id

    return path_nodes


def count_checkpoints_reached(positions, checkpoints, tol=0.5):
    reached = 0
    for cp in checkpoints:
        d = np.linalg.norm(positions[:, :3] - cp[:3], axis=1)
        if np.any(d < tol):
            reached += 1

    return reached



def mppi_2d_eval(path_nodes, seeds, N_list, K_list, visualize=True):
    """
    Evaluate MPPI performance for combinations of:
      - random seeds
      - horizon lengths N
      - rollout counts K

    Returns:
    # 1 dict -> keys=len(s)*len(n)*len(k) -> 1 dict -> 6 elements/metrics
    Dict[
        Tuple[int, int, int],   # (seed, N, K)
        Dict[str, float]        # metrics
        ]
    """
    results = {}

    for s in seeds:

        np.random.seed(s)
        walls_local = getLayout(layout_id=task_mppi_v1.global_layout_id,
                    random_obstacles=task_mppi_v1.local_layout_id, bounding_walls=True,
                    blockingPositions=task_mppi_v1.list_of_checkpoints)
        
        for N in N_list:
            for K in K_list:
                print(f"\nRunning MPPI for Seed={s}, Horizon={N}, Rollouts={K}")
                
                # (positions, velocities, angles, angular_vel, 
                # distance, velocity_xyz, acceleration_xyz, 
                # hits, clearance, error_global_local_path,
                # clock_time, computational_time, mppi_frequency_capability) = run_mppi(
                # path_nodes, walls_local, seed=s, N=N, K=K, visualize=visualize)
                result = run_mppi_segmented(path_nodes=path_nodes, checkpoints=task_mppi_v1.list_of_checkpoints,
                                            walls_local=walls_local, seed=s, N=N, K=K, visualize=visualize)

                if visualize and result is not None and result["positions"] is not None:
                    draw_tool(nodes_list=None,
                        goal_list=task_mppi_v1.list_of_checkpoints,
                        path_nodes=path_nodes,
                        walls=walls_local,
                        positions=result["positions"].tolist())

                key = (s, N, K)

                # ---------------- FAILURE ----------------
                if result is None or result["clock_time"] is None:
                    reached = 0
                    success = 0.0
                    results[key] = {
                        "tracking_error": np.nan,
                        "success_pct": success,
                        "max_acceleration": np.nan,
                        "computational_time": np.nan,
                        "clock_time": np.nan,
                        "mppi_frequency_capability": np.nan}
                    continue

                # ---------------- SUCCESS ----------------
                reached = count_checkpoints_reached(result["positions"], task_mppi_v1.list_of_checkpoints)

                if result["hits"] > 0:
                    success = 0.0
                else:
                    success = 100.0 * (reached - 1) / (len(task_mppi_v1.list_of_checkpoints) - 1)

                results[key] = { "tracking_error": float(result["tracking_error"]),
                    "success_pct": success,
                    "max_acceleration": float(result["max_acceleration"]),
                    "computational_time": float(result["computational_time"]),
                    "clock_time": float(result["clock_time"]),
                    "mppi_frequency_capability": float(result["mppi_frequency_capability"])}

    return results 

def run_mppi_segmented(path_nodes, checkpoints, walls_local, seed, N, K, visualize=False, tol=0.3
):
    """
    Run MPPI sequentially between checkpoints
    Returns aggregated metrics for the whole run
    """
    all_positions = []; all_acc = []; all_tracking_errors = []
    any_hit = False
    clock_time_total = 0.0
    comp_time_total = 0.0
    start_idx = 0
    MPPI_freq_cap_last = np.nan

    for i in range(len(checkpoints) - 1):
        goal = checkpoints[i + 1]

        # extract path segment up to next checkpoint 
        segment_nodes = []
        for n in path_nodes[start_idx:]:
            segment_nodes.append(n)
            if np.linalg.norm(n.position - goal) < tol:
                break

        # run MPPI for this segment
        (   positions, velocities, angles, angular_vel,
            distance, velocity_xyz, acceleration_xyz,
            hits, clearance, error_global_local_path,
            clock_time, computational_time, MPPI_freq_cap
        ) = run_mppi( segment_nodes, walls_local, seed=seed,N=N, K=K, visualize=visualize)

        if positions is None or len(positions) == 0:
            break

        # accumulate
        segment_tracking_error = np.linalg.norm(error_global_local_path, axis=1)
        all_tracking_errors.extend(segment_tracking_error)
        all_positions.extend(positions)
        all_acc.extend(acceleration_xyz)

        clock_time_total += 0 if clock_time is None else clock_time
        comp_time_total += computational_time
        any_hit |= hits > 0
        MPPI_freq_cap_last = MPPI_freq_cap

        # --- checkpoint reached? ---
        if np.linalg.norm(positions[-1][:2] - goal[:2]) > tol:
            break

        # update start index for next segment
        start_idx += len(segment_nodes) - 1

    return {"positions": np.array(all_positions),
        "tracking_error": float(np.mean(all_tracking_errors)) if len(all_tracking_errors) > 0 else np.nan,
        "max_acceleration": np.max(all_acc) if len(all_acc) > 0 else np.nan,
        "hits": int(any_hit),
        "clock_time": clock_time_total if len(all_tracking_errors) > 0 else None,
        "computational_time": comp_time_total,
        "mppi_frequency_capability": MPPI_freq_cap_last}


def aggregate_performance(results, N, K_list):
    summary = {}

    for K in K_list:
        summary[K] = {"success_pct": [],
                      "tracking_error": [],
                      "max_acceleration": []}

    for (seed, N_i, K), metrics in results.items():
        if metrics is None or N_i != N:
            continue
        summary[K]["success_pct"].append(metrics["success_pct"])
        summary[K]["tracking_error"].append(metrics["tracking_error"])
        summary[K]["max_acceleration"].append(metrics["max_acceleration"])

    # convert lists to mean/std
    for K in summary:
        for m in summary[K]:
            no_nan = np.asarray(summary[K][m], dtype=float)

            if np.any(~np.isnan(no_nan)):
                mean_value = float(np.nanmean(no_nan))
                std_value  = float(np.nanstd(no_nan))
            else:
                mean_value = np.nan
                std_value  = np.nan

            summary[K][m] = {"mean": mean_value,
                             "std": std_value}
            
    return summary

def aggregate_time_table(results, N_list, K_list):
    """
    Returns:
      table[N][K][metric] = {"mean": x, "std": y}
    """
    table = {}

    for N in N_list:
        table[N] = {}
        for K in K_list:
            table[N][K] = {
                "computational_time": [],
                "clock_time": [],
                "mppi_frequency_capability": [],
            }

    for (seed, N, K), metrics in results.items():
        if metrics is None:
            continue
        table[N][K]["computational_time"].append(metrics["computational_time"])
        table[N][K]["clock_time"].append(metrics["clock_time"])
        table[N][K]["mppi_frequency_capability"].append(metrics["mppi_frequency_capability"])

    # convert lists to mean/std
    for N in N_list:
        for K in K_list:
            for m in table[N][K]:
                vals = np.asarray(table[N][K][m], dtype=float)

                if np.any(~np.isnan(vals)):
                    table[N][K][m] = {"mean": float(np.nanmean(vals)),
                                      "std":  float(np.nanstd(vals))}
                else:
                    table[N][K][m] = {"mean": np.nan,
                                      "std":  np.nan}

    return table

def plot_mppi_evaluation(summary_15, summary_30, K_list):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(K_list))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # =====================================================
    # LEFT AXIS: Success rate [%]  (bars)
    # =====================================================
    ax1.bar(x - width/2,
        [summary_15[K]["success_pct"]["mean"] for K in K_list],
        width,
        yerr=[summary_15[K]["success_pct"]["std"] for K in K_list],
        label="Success (N=15)",
        color="tab:blue",
        alpha=0.4,
        edgecolor="black",
        linewidth=1,
        capsize=3,
        error_kw={"ecolor": "grey", "elinewidth": 1.0, "capthick": 1.0})

    ax1.bar(x + width/2,
        [summary_30[K]["success_pct"]["mean"] for K in K_list],
        width,
        yerr=[summary_30[K]["success_pct"]["std"] for K in K_list],
        label="Success (N=30)",
        color="tab:orange",
        alpha=0.4,
        edgecolor="black",
        linewidth=1,
        capsize=3,
        error_kw={"ecolor": "grey", "elinewidth": 1.0, "capthick": 1.0})
    

    ax1.set_xlabel("Rollouts K")
    ax1.set_ylabel("Percentage of Success [%]")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(K_list)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # =====================================================
    # RIGHT AXIS: Tracking Error [m] (solid lines)
    # =====================================================
    ax2 = ax1.twinx()
    ax2.errorbar(x,
        [summary_15[K]["tracking_error"]["mean"] for K in K_list],
        #yerr=[summary_15[K]["tracking_error"]["std"] for K in K_list],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="tab:blue",
        label="Track Error[-] (N=15)")

    ax2.errorbar(x,
        [summary_30[K]["tracking_error"]["mean"] for K in K_list],
        #yerr=[summary_30[K]["tracking_error"]["std"] for K in K_list],
        marker="s",
        linestyle="-",
        linewidth=2,
        color="tab:orange",
        label="Track Error[-] (N=30)")

    ax2.set_ylabel("Mean tracking error [m]")

    # =====================================================
    # THIRD AXIS: Max acceleration [m/s²] (dashed lines)
    # =====================================================
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))  # shift third axis

    ax3.plot(x,
        [summary_15[K]["max_acceleration"]["mean"] for K in K_list],
        linestyle="--",
        marker="^",
        linewidth=2,
        color="tab:blue",
        label="Max Acc[--] (N=15)")

    ax3.plot(x,
        [summary_30[K]["max_acceleration"]["mean"] for K in K_list],
        linestyle="--",
        marker="v",
        linewidth=2,
        color="tab:orange",
        label="Max Acc[--] (N=30)")

    ax3.set_ylabel("Max acceleration [m/s²]")
    ax3.set_ylim(0, 6)   # important: explicit scale

    # =====================================================
    # LEGEND 
    # =====================================================
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()

    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="lower left", fontsize=8)

    plt.title("MPPI Performance vs Rollouts")
    plt.tight_layout()
    plt.show()


def print_time_table(table, N_list, K_list):
    print("\n=== MPPI Timing Table ===\n")

    header = "N \\ K"
    for K in K_list:
        header += f" | {K}"
    print(header)
    print("-" * len(header) * 2)

    for N in N_list:
        row = f"{N}"
        for K in K_list:
            c = table[N][K]["computational_time"]
            m = table[N][K]["mppi_frequency_capability"]
            s = table[N][K]["clock_time"]

            cell = (f"\nComp: {c['mean']:.1f}±{c['std']:.1f}s"
                    f"\nMPPI Step: {m['mean']:.1f}±{m['std']:.1f}Hz"
                    f"\nSim: {s['mean']:.1f}±{s['std']:.1f}s")
            row += f" | {cell}"
        print(row)
        print("-" * len(header) * 2)


# ============================================================================================
# ============================================================================================

# walls
walls_global = getLayout(task_mppi_v1.global_layout_id)
# graph
nodes_list = task_mppi_v1.get_graph(walls=walls_global, max_nodes=5000)
# path of nodes
path_nodes = compute_path_nodes(nodes_list, checkpoints=task_mppi_v1.list_of_checkpoints,
                                walls=walls_global, graph_search_type="aStar", r=1.0)

results = mppi_2d_eval(path_nodes, seeds=seed_list, N_list=N_list, K_list=K_list, visualize=visualize)

# plot
summary_15 = aggregate_performance(results, N=N_list[0], K_list=K_list)
summary_30 = aggregate_performance(results, N=N_list[1], K_list=K_list)
plot_mppi_evaluation(summary_15, summary_30, K_list)

# table
time_table = aggregate_time_table(results, N_list, K_list)
print_time_table(time_table, N_list, K_list)

save_mppi_results_npz(
    file_path="mppi_eval_results.npz", summary_N1=summary_15, summary_N2=summary_30, K_list=K_list, N_list=N_list, time_table=time_table)