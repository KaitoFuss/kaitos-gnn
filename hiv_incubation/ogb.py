from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from ogb.graphproppred import Evaluator


# isjdfjksfsjk
def load_data():
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    return dataset


def split_data(dataset):
    split_idx = dataset.get_idx_split()

    train_dataset = Subset(dataset, split_idx["train"])
    valid_dataset = Subset(dataset, split_idx["valid"])
    test_dataset = Subset(dataset, split_idx["test"])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, valid_loader, test_loader


def evaluate(y_true, y_pred):
    evaluator = Evaluator(name="ogbg-molhiv")
    # You can learn the input and output format specification of the evaluator as follows.
    # print(evaluator.expected_output_format)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)  # E.g., {'rocauc': 0.7321}
    return result_dict
