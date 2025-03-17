import torch
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear, Dropout, BatchNorm1d
import torch.nn.functional as F


class GCNModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super(GCNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, improved=True))
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels, improved=True))

        self.fc = Linear(out_channels, 1)
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = edge_index.long()
        x = x.float()

        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms[:-1]):
            x_residual = x
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x) + x_residual  # Residual connection
            x = self.dropout(x)

        # Final GCN layer (no ReLU, no residual)
        x = self.convs[-1](x, edge_index)

        # Global add pooling
        x = global_add_pool(x, batch)

        # Fully connected layer
        x = self.fc(x)
        return x
