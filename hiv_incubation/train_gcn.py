from ogb_helper import load_data, split_data, evaluate
from simple_gcn import GCN
import torch.optim as optim
import torch.nn as nn
import torch


def train(loader):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, data.y.view(-1, 1).float())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader)


@torch.no_grad()
def test(loader):
    model.eval()

    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        y_true.append(data.y.view(-1, 1).detach().cpu())
        y_pred.append(out.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    result = evaluate(y_true, y_pred)
    return result


if __name__ == "__main__":
    dataset = load_data()
    train_loader, valid_loader, test_loader = split_data(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(in_channels=dataset.num_features, hidden_channels=128, out_channels=64)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    print(f"Decreased lr to 0.0005 from 0.001")
    for epoch in range(1, 101):
        loss = train(train_loader)
        if epoch % 10 == 0:
            val_acc = test(valid_loader)
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                f"Val ROC-AUC: {val_acc['rocauc']:.4f}"
            )
            scheduler.step(val_acc["rocauc"])

    test_acc = test(test_loader)
    print(f"Training has been finished. Testing ROC-AUC: {test_acc['rocauc']:.4f}")


# def train():
#     dataset = load_data()
#     train_loader, valid_loader, test_loader = split_data(dataset)

#     # Initialize the model, optimizer, and loss function
#     model = GCN(in_channels=dataset.num_features, hidden_channels=64, out_channels=64)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.BCEWithLogitsLoss()

#     # Train the model
#     for epoch in range(1, 101):  # Train for 100 epochs
#         model.train()
#         for data in train_loader:
#             optimizer.zero_grad()
#             out = model(data)
#             loss = criterion(
#                 out, data.y.view(-1, 1).float()
#             )  # Assuming binary classification
#             loss.backward()
#             optimizer.step()

#         # Validation
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for data in valid_loader:
#                 out = model(data)
#                 y_true.append(data.y.view(-1, 1).cpu())
#                 y_pred.append(out.cpu())

#         y_true = torch.cat(y_true, dim=0)
#         y_pred = torch.cat(y_pred, dim=0)

#         # Evaluate
#         result = evaluate(y_true, y_pred)
#         if epoch % 10 == 0:
#             print(
#                 f"Epoch {epoch}: Loss={loss.item():.4f}, ROC-AUC={result['rocauc']:.4f}"
#             )


# if __name__ == "__main__":
#     train()
