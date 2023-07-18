import os
import random

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    for x, _ in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = x.to(device)
            y = pred_fun(model(x))
        y = y.cpu().numpy()
        preds.append(y)
    preds = np.concatenate(preds)
    return preds