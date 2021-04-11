import logging

import tqdm
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model: Module, dataloader: DataLoader, criterion: Module, optimizer: Optimizer) -> None:
    model.zero_grad()
    model.train()
    model.to(device)

    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        batch = tuple(tensor.to(device) for tensor in batch)

        _, predictions = model(batch[0])

        loss = criterion(predictions, batch[1])
        loss.backward()

        if i % 300 == 0:
            logger.warning("Loss: {}, at iteration: {}".format(loss.item(), i))

        optimizer.step()
        model.zero_grad()

def eval_model(model: Module, dataloader: DataLoader, criterion: Module) -> None:
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            batch = tuple(tensor.to(device) for tensor in batch)

            _, predictions = model(batch[0])

            loss = criterion(predictions, batch[1])

            if i % 100 == 0:
                logger.warning("Loss: {}, at iteration: {}".format(loss.item(), i))
