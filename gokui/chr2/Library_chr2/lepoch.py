import os
from tqdm import tqdm
import torch

def train_lepoch(model, train_loader, lossfun, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)

        loss = lossfun(out, y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item() * x.size(0)

        _, pred = torch.max(out, 1)
        total_acc += torch.sum(pred==y.data)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss

