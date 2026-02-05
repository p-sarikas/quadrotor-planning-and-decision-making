from Task_2D.data_handling import load_time_data_npz
import os
import pandas as pd
import numpy as np

save_path = os.path.join("Task_2D", "exp_data", "compute_time")
file_name = "time_data.npz"
file_path = os.path.join(save_path, file_name)

t_prm_raw, t_grid_raw, t_aStar_raw, t_dijkstra_raw = load_time_data_npz(file_path)
print(t_prm_raw)


t_avg = [np.mean(x) for x in t_dijkstra_raw]
print([f"{v:.2f}" for v in t_avg])

