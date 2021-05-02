import argparse
import json
import logging

import tqdm
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from Model.baseline import ConvNet
from Training.train_eval import train_model, eval_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(**kwargs) -> None:

    train_dataset = MNIST(
        root=".",
        train = True,
        download=True, 
        transform = Compose(
           [
               ToTensor(),
               Normalize((0.1307,), (0.3081,))
               
           ]
        )
    )
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, kwargs["batch_size"], sampler = train_sampler)


    test_dataset = MNIST(
        root=".",
        train = False,
        download=True,
        transform = Compose(
           [
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
           ]
        )
    )
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, kwargs["batch_size"], sampler = test_sampler)

    model = ConvNet(**kwargs)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = kwargs["learning_rate"])

    for _ in tqdm.tqdm(range(kwargs["epochs"])):
        train_model(model, train_loader, criterion, optimizer)
        eval_model(model, test_loader, criterion)

    with open("config_file.json", "w") as file:
        json.dump(kwargs, file)

    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--in_channels", type=int, required=False, default=1)
    argument_parser.add_argument("--n_classes", type=int, required=False, default=10)
    argument_parser.add_argument("--epochs", type=int, required=False, default=2)
    argument_parser.add_argument("--batch_size", type=int, required=False, default=128)
    argument_parser.add_argument("--learning_rate", type=float, required=False, default=0.0001)

    arguments = argument_parser.parse_args()

    main(**vars(arguments))