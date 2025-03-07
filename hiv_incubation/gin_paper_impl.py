import torch
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, LeakyReLU
import torch.nn.functional as F


class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                LeakyReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = edge_index.long()
        x = x.float()
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            # x_res = x
            x = F.leaky_relu(batch_norm(conv(x, edge_index)))
            # Add residual connection
            # x = x + x_res  # Add the residual connection (skip connection)
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)
        return x
