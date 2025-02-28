import torch
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        x = x.float()

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)

        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x
