from ogb_helper import load_data, split_data, evaluate
from gin_pyg_class import GINModel
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
    model = GINModel(
        in_channels=dataset.num_features,
        hidden_channels=128,
        out_channels=64,
        num_layers=2,
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_func = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    print(f"Increased learning rate from 0.001 to 0.005")
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
