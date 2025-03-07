import torch
from torch_geometric.nn import GIN, global_add_pool, global_mean_pool


class GINModel(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5
    ):
        super(GINModel, self).__init__()
        self.gin = GIN(in_channels, hidden_channels, num_layers, out_channels, dropout)
        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        x = x.float()
        x = self.gin(x, edge_index)
        x = global_add_pool(x, data.batch)
        x = self.fc(x)
        return x
