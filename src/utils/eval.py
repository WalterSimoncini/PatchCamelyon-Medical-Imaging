import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.nn.functional import softmax


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    model.eval()

    ground_truth = np.zeros(len(test_loader.dataset))
    true_class_probs = np.zeros(len(test_loader.dataset))

    batch_size = test_loader.batch_size
    test_loss, correct_preds, batches_n = 0, 0, 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            images, targets = images.to(device), targets.to(device)

            preds = model(images)
            test_loss += loss_fn(preds, targets)

            # Update the arrays used to compute the ROC and AUC
            ground_truth[(i * batch_size):((i + 1) * batch_size)] = targets.cpu().numpy()
            true_class_probs[(i * batch_size):((i + 1) * batch_size)] = softmax(preds, dim=1)[:, 1].cpu().numpy()

            batches_n += 1
            preds = preds.argmax(dim=1)
            correct_preds += (preds == targets).sum()

    test_loss /= batches_n
    accuracy = correct_preds / len(test_loader.dataset)

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth,
        true_class_probs,
        pos_label=1
    )

    return test_loss, accuracy, metrics.auc(fpr, tpr)
