import os
import numpy as np

def save_data_npz(
        file_path,
        chpts_prm_avgs, 
        chpts_prm_raw, 
        path_prm_avgs, 
        path_prm_raw, 
        t_graph_prm_avgs, 
        chpts_grid_avgs, 
        path_grid_avgs, 
        t_graph_grid_avgs,
        max_nodes
        ):
    
    chpts_prm_raw_arr = np.array(chpts_prm_raw, dtype=object)
    path_prm_raw_arr = np.array(path_prm_raw, dtype=object)
    
    np.savez(
        file_path, 
        chpts_prm_raw_arr=chpts_prm_raw_arr,
        path_prm_raw_arr=path_prm_raw_arr,
        chpts_prm_avgs=np.array(chpts_prm_avgs),
        path_prm_avgs=np.array(path_prm_avgs),
        t_graph_prm_avgs=np.array(t_graph_prm_avgs),
        chpts_grid_avgs=np.array(chpts_grid_avgs),
        path_grid_avgs=np.array(path_grid_avgs),
        t_graph_grid_avgs=np.array(t_graph_grid_avgs),
        max_nodes=np.array(max_nodes)
        )
    
    return


def load_data_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    chpts_prm_raw = data["chpts_prm_raw_arr"].tolist()
    path_prm_raw = data["path_prm_raw_arr"].tolist()
    chpts_prm_avgs = data["chpts_prm_avgs"]
    path_prm_avgs = data["path_prm_avgs"]
    t_graph_prm_avgs = data["t_graph_prm_avgs"]
    chpts_grid_avgs = data["chpts_grid_avgs"]
    path_grid_avgs = data["path_grid_avgs"]
    t_graph_grid_avgs = data["t_graph_grid_avgs"]
    max_nodes = data["max_nodes"]

    return chpts_prm_avgs, chpts_prm_raw, path_prm_avgs, path_prm_raw, t_graph_prm_avgs, chpts_grid_avgs, path_grid_avgs, t_graph_grid_avgs, max_nodes


def save_time_data_npz(
        file_path,
        t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw
        ):
    
    t_prm_raw_arr = np.array(t_prm_raw, dtype=object)
    t_grid_raw_arr = np.array(t_grid_raw, dtype=object)
    t_aStar_raw_arr = np.array(t_aStar_raw, dtype=object)
    t_dijkstra_raw_arr = np.array(t_dijkstra_raw, dtype=object)
    
    np.savez(
        file_path, 
        t_prm_raw_arr=t_prm_raw_arr,
        t_grid_raw_arr=t_grid_raw_arr,
        t_aStar_raw_arr=t_aStar_raw_arr,
        t_dijkstra_raw_arr=t_dijkstra_raw_arr,
        )
    
    return

def load_time_data_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    t_prm_raw = data["t_prm_raw_arr"].tolist()
    t_grid_raw = data["t_grid_raw_arr"].tolist()
    t_aStar_raw = data["t_aStar_raw_arr"].tolist()
    t_dijkstra_raw = data["t_dijkstra_raw_arr"].tolist()

    return t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw


def save_mppi_results_npz(file_path, summary_N1, summary_N2, K_list, N_list, time_table=None):

    # Convert summaries into arrays (ordered by K)
    def unpack_summary(summary, metric):
        mean = [summary[K][metric]["mean"] for K in K_list]
        std  = [summary[K][metric]["std"]  for K in K_list]
        return np.array(mean), np.array(std)

    data = {
        "K_list": np.array(K_list),
        "N_list": np.array(N_list)}
    
    #  N=15
    data["success_mean_N1"], data["success_std_N1"] = unpack_summary(summary_N1, "success_pct")
    data["tracking_mean_N1"], data["tracking_std_N1"] = unpack_summary(summary_N1, "tracking_error")
    data["acc_mean_N1"], data["acc_std_N1"] = unpack_summary(summary_N1, "max_acceleration")

    #  N=30
    data["success_mean_N2"], data["success_std_N2"] = unpack_summary(summary_N2, "success_pct")
    data["tracking_mean_N2"], data["tracking_std_N2"] = unpack_summary(summary_N2, "tracking_error")
    data["acc_mean_N2"], data["acc_std_N2"] = unpack_summary(summary_N2, "max_acceleration")

    # Timing table
    if time_table is not None:
        data["time_table"] = np.array(time_table, dtype=object)

    np.savez(file_path, **data)