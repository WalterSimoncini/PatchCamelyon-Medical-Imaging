import os
import h5py
import torch
import logging
import numpy as np
import argparse

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from typing import Tuple, List
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision.transforms.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights

import wandb

from dataloader import PatchCamelyonDataset
from utils import show


def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module):
    model.train()
    train_loss, correct_preds, batches_n = 0, 0, 0

    for batch_nr, (images, targets) in enumerate(tqdm(train_loader)):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        preds = model(images)

        loss = loss_fn(preds, targets)
        loss.backward()

        # Record the training loss and the number of correct predictions
        batches_n += 1
        train_loss += loss.item()

        preds = preds.argmax(dim=1)
        nr_correct_batch = (preds == targets).sum()
        correct_preds += nr_correct_batch

        # Log batch metrics
        wandb.log(
            {
                "Batch": batch_nr,
                "batch_loss/train": loss.item(),
                "batch_acc/train": nr_correct_batch / len(images),
            }
        )

        optimizer.step()

    train_loss /= batches_n
    accuracy = correct_preds / len(train_loader.dataset)

    return train_loss, accuracy


def test(model: nn.Module, device: torch.device, test_loader: DataLoader, loss_fn: nn.Module):
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


def get_camelyon_dataloader():
    train_dataset = PatchCamelyonDataset(
        data_path="data/camelyonpatch_level_2_split_train_x.h5", targets_path="data/camelyonpatch_level_2_split_train_y.h5", transform=transforms.Resize(224, antialias=True)
    )

    val_dataset = PatchCamelyonDataset(
        data_path="data/camelyonpatch_level_2_split_valid_x.h5", targets_path="data/camelyonpatch_level_2_split_valid_y.h5", transform=transforms.Resize(224, antialias=True)
    )

    test_dataset = PatchCamelyonDataset(
        data_path="data/camelyonpatch_level_2_split_test_x.h5", targets_path="data/camelyonpatch_level_2_split_test_y.h5", transform=transforms.Resize(224, antialias=True)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=True, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, optimizer, loss_fn, writer, run_folder):
    best_val_accuracy = 0
    for epoch in range(args.epochs):
        logging.info(f"starting epoch {epoch}")

        train_loss, train_accuracy = train_epoch(model=model, device=args.device, optimizer=optimizer, loss_fn=loss_fn, train_loader=train_loader)
        val_loss, val_accuracy = test(model=model, device=args.device, test_loader=val_loader)

        # Log epoch metrics for train and validation
        wandb.log(
            {
                "Epoch": epoch,
                "loss/train": train_loss,
                "loss/val": val_loss,
                "acc/train": train_accuracy,
                "acc/val": val_accuracy,
            }
        )

        logging.info(f"the train accuracy was {train_accuracy} (loss: {train_loss})")
        logging.info(f"the validation accuracy was {val_accuracy} (loss: {val_loss})")

        # Log metrics to tensorboard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_accuracy, epoch)
        writer.add_scalar("acc/val", val_accuracy, epoch)

        # Pick the best model according to the validation
        # accuracy score
        if val_accuracy > best_val_accuracy:
            logging.info(f"found new best model at epoch {epoch} with accuracy {val_accuracy} (loss {val_loss})")

            best_val_accuracy = val_accuracy
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy
            wandb.run.summary["best_val_epoch"] = epoch

            # Save the model to disk
            torch.save(model.state_dict(), os.path.join(run_folder, f"model_{epoch}.pt"))
            wandb.run.summary["best_model_path"] = os.path.join(run_folder, f"model_{epoch}.pt")


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    run_folder = os.path.join(args.output_path, run_id)
    os.makedirs(run_folder)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=2)

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.optimizer = optimizer.__class__.__name__  # for wandb logging
    args.loss_fn = loss_fn.__class__.__name__
    args.model = model.__class__.__name__

    run = wandb.init(project="PatchCamelyon", entity="mi_ams", config=args)
    writer = SummaryWriter(run_folder)

    model = model.to(args.device)

    train_loader, val_loader, test_loader = get_camelyon_dataloader()
    class2label = {0: "No Tumor", 1: "Tumor"}

    train(model, train_loader, val_loader, optimizer, loss_fn, writer, run_folder)

    test_loss, test_accuracy = test(model=model, device=args.device, test_loader=test_loader, loss_fn=loss_fn)

    logging.info(f"the test accuracy was {test_accuracy}")
    wandb.run.summary["acc/test"] = test_accuracy

    run.finish()


if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s:%(levelname)s]: %(message)s", level=logging.INFO)
    wandb.login()

    # Training Hyperparameters
    parser = argparse.ArgumentParser(description="MI Project")

    parser.add_argument("--output_path", default="runs", type=str, help="Path to save the model")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for training and validation")
    parser.add_argument("--batch_size_test", default=64, type=int, help="Batch size for testing")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train for")

    args = parser.parse_args()

    main(args)
