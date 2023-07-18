import os
import random

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = lossfun(out, y)
            _, pred = torch.max(out, 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y.data)

        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = total_acc / len(val_loader.dataset)

        return avg_acc, avg_loss