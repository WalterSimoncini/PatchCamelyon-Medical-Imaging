import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    model.eval()

    test_loss, correct_preds, batches_n = 0, 0, 0

    with torch.no_grad():
        for _, (images, targets) in enumerate(tqdm(test_loader)):
            images, targets = images.to(device), targets.to(device)

            preds = model(images)
            test_loss += loss_fn(preds, targets)

            batches_n += 1
            preds = preds.argmax(dim=1)
            correct_preds += (preds == targets).sum()

    test_loss /= batches_n
    accuracy = correct_preds / len(test_loader.dataset)

    return test_loss, accuracy
