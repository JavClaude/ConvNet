import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.Model.baseline import ConvNet
from src.Training.train_eval import train_model, eval_model

device = "cpu"


class FakeDataSet(Dataset):
    def __init__(self, n_samples: int, in_channels: int, input_dim: int) -> None:
        self.data = torch.randn(size=(n_samples, in_channels, input_dim, input_dim), device=device, dtype=torch.float)
        self.target = torch.randint(0, 9, size=(n_samples, ), device=device, dtype=torch.long)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.target[index]


def test_train_model():
    test_dataset = FakeDataSet(32, 1, 28)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    test_model = ConvNet(1, 9)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(test_model.parameters(), lr=0.0001)

    train_model(test_model, test_dataloader, criterion, optimizer)

def test_eval_model():
    test_dataset = FakeDataSet(32, 1, 28)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    test_model = ConvNet(1, 9)
    criterion = nn.CrossEntropyLoss()

    eval_model(test_model, test_dataloader, criterion)
