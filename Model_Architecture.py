import torch
from torch_geometric.nn import ChebConv


class DGCNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ChebConv(in_channels, 64, K=3) # K is the order of the chebyshev polynomial
        self.conv2 = ChebConv(64, 128, K=3)
        self.conv3 = ChebConv(128, out_channels, K=3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x