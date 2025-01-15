import torch
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from Physical_Loss_3 import boundary_loss, continuity_loss, momentum_loss

def loss_data_driven(data, y_hat):
    n_nodes = data.x[:, 0].shape[0]

    loss_data_driven = 0
    for i in range(0, n_nodes):
        loss_data_driven += (y_hat[i, 0] - data.y[i, 0] +
                       y_hat[i, 1] - data.y[i, 1] +
                       y_hat[i, 2] - data.y[i, 2] +
                       y_hat[i, 3] - data.y[i, 3]) ** 2
    loss_data_driven /= n_nodes

    return loss_data_driven



def model_training(model, train_dataset, val_dataset,  num_epochs, characteristic_lenght, delta_t):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train_loss_history = np.zeros(num_epochs)
    validation_loss_history = np.zeros(num_epochs)
    epoch_history = np.zeros(num_epochs)
    counter = 0


    for epoch in range(num_epochs):
        model.train()
        counter = 0
        for data in DataLoader(train_dataset, batch_size=1, shuffle=False):
            y_hat = model(data.x, data.edge_index, data.batch)
            #cost_data_driven = loss_data_driven(data, y_hat) #torch.mean((y_hat - data.y) ** 2)
            #print("cost_data_driven1", cost_data_driven)

            cost_data_driven = torch.mean((y_hat - data.y) ** 2)
            #print("cost_data_driven2", cost_data_driven)
            #cost_physics_informed = boundary_loss(data, y_hat) + continuity_loss(data, y_hat, characteristic_lenght) #+ (data, y_hat)
            loss = cost_data_driven #+ cost_physics_informed
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()
        train_loss_history[epoch] += cost_data_driven

        counter += 1
        scheduler.step()

        train_loss_history[epoch] /= counter
        counter = 0
        model.eval()
        with torch.no_grad():
            for data in DataLoader(val_dataset, batch_size=1, shuffle=False):
                out = model(data.x, data.edge_index, data.batch)#, data.batch)
                cost_data_driven = torch.mean((out - data.y) ** 2)
                #cost_physics_informed = navier_stokes_losses(data, data.y)
                val_loss = cost_data_driven #+ cost_physics_informed
                validation_loss_history[epoch] += val_loss
                counter += 1

        validation_loss_history[epoch] /= counter
        print('Epoch: {:03d}, Loss_train: {:.5f}, Loss_validate: {:.5f}'.format(epoch, train_loss_history[epoch], validation_loss_history[epoch]))
        epoch_history[epoch] = epoch

    torch.save(model.state_dict(), "/Users/mahdi.rajabi/Documents/MLforPointCloud/Graph_NN.h5")
    return model

