import torch
import torch.nn as nn
from scipy.interpolate import Rbf
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def boundary_loss(data, y_hat):
    n_nodes = data.x[:, 0].shape[0]
    print("n_nodes", n_nodes)
    r_boundary = 0
    for i in range (0,n_nodes):
        subset_number = data.x[i,7]
        if subset_number != 0:
            r_boundary += (y_hat[i, 0] - data.x[i, 3] +
                            y_hat[i, 1] - data.x[i, 4] +
                            y_hat[i, 2] - data.x[i, 5] +
                            y_hat[i, 3] - data.x[i, 6]) ** 2
    r_boundary /= n_nodes
    print("r_boundary", r_boundary)
    return r_boundary



def auto_set_rbf_epsilon(x, y, z):
    # Compute average distance between neighboring points
    dist = cdist(np.stack((x, y, z), axis=1), np.stack((x, y, z), axis=1))
    avg_dist = np.max(np.sort(dist, axis=1)[:,1])
    print("avg_dist", avg_dist)
    # Set RBF epsilon as a fraction of the average distance
    rbf_epsilon = 0.5 * avg_dist
    print("rbf_epsilon", rbf_epsilon)
    return rbf_epsilon


class RBFFD():
    def __init__(self, x, y, z, epsilon):
        self.x = x
        self.y = y
        self.z = z
        self.epsilon = epsilon
        self.rbf = Rbf(x, y, z,  epsilon=self.epsilon, smooth=0.001)# function='gaussian',
        # Compute nearest neighbors using sklearn
        self.nbrs = NearestNeighbors(n_neighbors=16, algorithm='kd_tree').fit(np.stack((x, y, z), axis=1))
        self.neighbor_lists = self.nbrs.kneighbors(np.stack((x, y, z), axis=1), return_distance=False)

    def compute_derivative(self, f, axis, deriv_order=1):
        # Compute derivative using RBF-FD method
        weights = self.rbf.nodes[:, axis] - self.rbf.nodes[self.neighbor_lists[axis], axis][:, np.newaxis]
        weights = np.where(weights == 0, 1e-16, weights)
        weights = np.power(np.abs(weights), deriv_order + 1) * np.sign(weights) / np.math.factorial(deriv_order)
        return np.dot(weights, f[self.neighbor_lists[axis]])

    def compute_mixed_derivative(self, f1, x1, f2, x2):
        # Compute mixed derivative using RBF-FD method
        weights_x = self.rbf.nodes[:, 0] - self.rbf.nodes[self.neighbor_lists[0], 0][:, np.newaxis]
        weights_x = np.where(weights_x == 0, 1e-16, weights_x)
        weights_x = -self.rbf(x1, self.rbf.nodes[:, 1], self.rbf.nodes[:, 2]) * np.power(np.abs(weights_x), 2) / self.epsilon**2 * np.sign(weights_x)
        dx1 = np.dot(weights_x, f1[self.neighbor_lists[0]])
        dx2 = np.dot(weights_x, f2[self.neighbor_lists[0]])

        weights_y = self.rbf.nodes[:, 1] - self.rbf.nodes[self.neighbor_lists[1], 1][:, np.newaxis]
        weights_y = np.where(weights_y == 0, 1e-16, weights_y)
        weights_y = -self.rbf(self.rbf.nodes[:, 0], x2, self.rbf.nodes[:, 2]) * np.power(np.abs(weights_y), 2) / self.epsilon**2 * np.sign(weights_y)
        dy1 = np.dot(weights_y, f1[self.neighbor_lists[1]])
        dy2 = np.dot(weights_y, f2[self.neighbor_lists[1]])

        weights_z = self.rbf.nodes[:, 2] - self.rbf.nodes[self.neighbor_lists[2], 2][:, np.newaxis]
        weights_z = np.where(weights_z == 0, 1e-16, weights_z)
        weights_z = -self.rbf(self.rbf.nodes[:, 0], self.rbf.nodes[:, 1], x2) * np.power(np.abs(weights_z), 2) / self.epsilon**2 * np.sign(weights_z)
        dz1 = np.dot(weights_z, f1[self.neighbor_lists[2]])
        dz2 = np.dot(weights_z, f2[self.neighbor_lists[2]])

        return (dx1*dy2 - dx2*dy1) + (dx1*dz2 - dx2*dz1) + (dy1*dz2 - dy2*dz1)


def continuity_loss(data, y_hat, characteristic_lenght): #(u, x, y, z):
    x = data.x[:, 0]
    y = data.x[:, 1]
    z = data.x[:, 2]
    u = y_hat[:,0:3]

    x_dis = x.detach().numpy()
    y_dis = x.detach().numpy()
    z_dis = x.detach().numpy()

    rbf_epsilon = 0.1 * characteristic_lenght #auto_set_rbf_epsilon(x_dis, y_dis, z_dis)
    print("x_dis", x_dis)
    rbf = RBFFD(x_dis, y_dis, z_dis, epsilon=rbf_epsilon)######
    ux = rbf.compute_derivative(u[:,0], x, deriv_order=1)
    vy = rbf.compute_derivative(u[:,1], y, deriv_order=1)
    wz = rbf.compute_derivative(u[:,2], z, deriv_order=1)
    div_u = ux + vy + wz
    continuity_loss_eval = torch.mean(div_u.pow(2))
    print("continuity_loss_eval", continuity_loss_eval)
    return continuity_loss_eval



def momentum_loss(data, y_hat, dt, Re, characteristic_lenght):#(u, x, y, z, t, rho, mu, Re, f, rbf):
    x = data.x[:, 0]
    y = data.x[:, 1]
    z = data.x[:, 2]
    u = y_hat[:, 0]
    v = y_hat[:, 1]
    w = y_hat[:, 2]
    #p = y_hat[:, 3] #########

    rbf_epsilon = 0.1 * characteristic_lenght #auto_set_rbf_epsilon(x, y, z)
    rbf = RBFFD(x, y, z, epsilon=rbf_epsilon)

    du_dx = rbf.compute_derivative(u[:,0], x, deriv_order=1)
    du_dy = rbf.compute_derivative(u[:,0], y, deriv_order=1)
    du_dz = rbf.compute_derivative(u[:,0], z, deriv_order=1)
    dv_dx = rbf.compute_derivative(u[:,1], x, deriv_order=1)
    dv_dy = rbf.compute_derivative(u[:,1], y, deriv_order=1)
    dv_dz = rbf.compute_derivative(u[:,1], z, deriv_order=1)
    dw_dx = rbf.compute_derivative(u[:,2], x, deriv_order=1)
    dw_dy = rbf.compute_derivative(u[:,2], y, deriv_order=1)
    dw_dz = rbf.compute_derivative(u[:,2], z, deriv_order=1)
    dp_dx = rbf.compute_derivative(u[:,3], x, deriv_order=1)
    dp_dy = rbf.compute_derivative(u[:,3], y, deriv_order=1)
    dp_dz = rbf.compute_derivative(u[:,3], z, deriv_order=1)

    #du_dt = torch.autograd.grad(u[:,0], t, create_graph=True)[0]
    #dv_dt = torch.autograd.grad(u[:,1], t, create_graph=True)[0]
    #dw_dt = torch.autograd.grad(u[:,2], t, create_graph=True)[0]
    #dp_dt = torch.autograd.grad(u[:,3], t, create_graph=True)[0]


    #dt = t[1] - t[0]  # assuming t is a uniformly spaced array
    du = u[1:, 0] - u[:-1, 0]  # compute the difference of u along the time dimension
    du_dt = torch.cat((du / dt, torch.zeros((1,))), dim=0)  # pad the last entry with zeros to match the shape of u

    dv = v[1:, 0] - v[:-1, 0]
    dv_dt = torch.cat((dv / dt, torch.zeros((1,))), dim=0)

    dw = w[1:, 0] - w[:-1, 0]
    dw_dt = torch.cat((dw / dt, torch.zeros((1,))), dim=0)

    d2u_dx2 = rbf.compute_derivative(u[:,0], x, deriv_order=2)
    d2u_dy2 = rbf.compute_derivative(u[:, 0], y, deriv_order=2)
    d2u_dz2 = rbf.compute_derivative(u[:, 0], z, deriv_order=2)

    d2v_dx2 = rbf.compute_derivative(u[:,0], x, deriv_order=2)
    d2v_dy2 = rbf.compute_derivative(u[:, 0], y, deriv_order=2)
    d2v_dz2 = rbf.compute_derivative(u[:, 0], z, deriv_order=2)

    d2w_dx2 = rbf.compute_derivative(u[:,0], x, deriv_order=2)
    d2w_dy2 = rbf.compute_derivative(u[:, 0], y, deriv_order=2)
    d2w_dz2 = rbf.compute_derivative(u[:, 0], z, deriv_order=2)

    momentum_res = torch.zeros_like(u[:, :3])

    momentum_res[:, 0] = du_dt + u*du_dx+ v*du_dy+ w*du_dz + dp_dx -(1/Re) *(d2u_dx2+d2u_dy2+d2u_dz2)
    momentum_res[:, 1] = dv_dt + u * dv_dx + v * dv_dy + w * dv_dz + dp_dy - (1 / Re) * (d2v_dx2 + d2v_dy2 + d2v_dz2)
    momentum_res[:, 2] = dw_dt + u * dw_dx + v * dw_dy + w * dw_dz + dp_dz - (1 / Re) * (d2w_dx2 + d2w_dy2 + d2w_dz2)

    momentum_loss_eval = torch.mean(momentum_res[:, 0].pow(2)+momentum_res[:, 1].pow(2)+momentum_res[:, 2].pow(2))#momentum_res.norm(dim=1).mean()
    return momentum_loss_eval
