import torch
import torch_geometric
from torch_geometric.data import Data
import pandas as pd
import numpy as np


def read_training_test_data(list_of_folders_train, list_of_folders_val, max_time_steps, characteristic_lenght, characteristic_pressure, delat_t):
    train_dataset = []
    val_dataset = []
    for folder in list_of_folders_train:
        last_three_chars = folder[-3:]
        enforced_pressure = float(last_three_chars)
        for time_step in range(0, max_time_steps):
            cloud_path_t = "/Users/mahdi.rajabi/Documents/MLforPointCloud/data/Pointcloud/64/" + folder +"/64_" + str(time_step) + ".csv"
            cloud_path_t_plus_1 = "/Users/mahdi.rajabi/Documents/MLforPointCloud/data/Pointcloud/64/" + folder +"/64_" + str(time_step + 1) + ".csv"

            df_t_init = pd.read_csv(cloud_path_t)
            df_t = df_t_init[df_t_init['03-Chi'] == 0] # Filter out points that are in the solid phase

            df_t_plus_1_init = pd.read_csv(cloud_path_t_plus_1)
            df_t_plus_1 = df_t_plus_1_init[df_t_plus_1_init['03-Chi'] == 0]

            pos = torch.tensor(df_t[['Points:0', 'Points:1', 'Points:2']].to_numpy())
            x = np.zeros((df_t.shape[0], df_t.shape[1]+2))
            new_column_pressure_imposed = np.full((x.shape[0]), enforced_pressure / characteristic_pressure)
            new_column_time = np.full((x.shape[0]), time_step * delat_t)
            new_column_time_plus_delat_t = np.full((x.shape[0]), (time_step+1) * delat_t)

            x[:, 0:8] = df_t[
                ['Points:0', 'Points:1', 'Points:2', '01-U:0', '01-U:1', '01-U:2', '02-P', 'SubsetNumber']].to_numpy()
            x[:, 8] = new_column_pressure_imposed
            x[:, 9] = new_column_time
            x[:, 10] = new_column_time_plus_delat_t

            x[:, 3:6] /= characteristic_lenght
            x[:, 6] /= characteristic_pressure

            x = torch.tensor(x, dtype=torch.float, requires_grad=True)

            y = df_t_plus_1[['01-U:0', '01-U:1', '01-U:2', '02-P']].to_numpy()
            y[:, 0:3] /= characteristic_lenght
            y[:, 3] /= characteristic_pressure


            y = torch.tensor(y, dtype=torch.float, requires_grad=True)
            edge_index = torch_geometric.nn.knn_graph(pos, k=24, loop=False)
            edge_weight = torch.ones(edge_index.size(1))
            data = Data(x=x, y=y, pos=pos, edge_index=edge_index, edge_weight=edge_weight)
            train_dataset.append(data)

    for folder in list_of_folders_val:
        last_three_chars = folder[-3:]
        enforced_pressure = float(last_three_chars)
        for time_step in range(0, max_time_steps - 4):  # 327 max_time_steps
            cloud_path_t = "/Users/mahdi.rajabi/Documents/MLforPointCloud/data/Pointcloud/64/" + folder + "/64_" + str(
                time_step) + ".csv"
            cloud_path_t_plus_1 = "/Users/mahdi.rajabi/Documents/MLforPointCloud/data/Pointcloud/64/" + folder + "/64_" + str(
                time_step + 1) + ".csv"

            df_t_init = pd.read_csv(cloud_path_t)
            df_t = df_t_init[df_t_init['03-Chi'] == 0]  # Filter out points that are in the solid phase

            pos = torch.tensor(df_t[['Points:0', 'Points:1', 'Points:2']].to_numpy())
            x = np.zeros((df_t.shape[0], df_t.shape[1]+2))
            new_column_pressure_imposed = np.full((x.shape[0]), enforced_pressure / characteristic_pressure)
            new_column_time = np.full((x.shape[0]), time_step * delat_t)
            new_column_time_plus_delat_t = np.full((x.shape[0]), (time_step+1) * delat_t)

            x[:, 0:8] = df_t[
                ['Points:0', 'Points:1', 'Points:2', '01-U:0', '01-U:1', '01-U:2', '02-P', 'SubsetNumber']].to_numpy()
            x[:, 8] = new_column_pressure_imposed
            x[:, 9] = new_column_time
            x[:, 10] = new_column_time_plus_delat_t

            x[:, 3:6] /= characteristic_lenght
            x[:, 6] /= characteristic_pressure

            x = torch.tensor(x, dtype=torch.float, requires_grad=True)

            y = df_t_plus_1[['01-U:0', '01-U:1', '01-U:2', '02-P']].to_numpy()
            y[:, 0:3] /= characteristic_lenght
            y[:, 3] /= characteristic_pressure



            y = torch.tensor(y, dtype=torch.float, requires_grad=True)
            edge_index = torch_geometric.nn.knn_graph(pos, k=24, loop=False)
            edge_weight = torch.ones(edge_index.size(1))
            data = Data(x=x, y=y, pos=pos, edge_index=edge_index, edge_weight=edge_weight)
            val_dataset.append(data)

    return train_dataset, val_dataset