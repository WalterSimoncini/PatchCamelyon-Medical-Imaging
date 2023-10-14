import os
import wandb
import torch
import logging

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.eval import evaluate_model

from .eval import evaluate_model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
):
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
        wandb.log({
            "Batch": batch_nr,
            "batch_loss/train": loss.item(),
            "batch_acc/train": nr_correct_batch / len(images),
        })

        optimizer.step()

    train_loss /= batches_n
    accuracy = correct_preds / len(train_loader.dataset)

    return train_loss, accuracy


def train_epoch_stain_ensemble(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
):
    model.train()

    train_loss, correct_preds, batches_n = 0, 0, 0

    # Es are not used for the stain ensemble as they
    # do not yield good representations
    for batch_nr, (images, norms, Hs, Es, targets) in enumerate(tqdm(train_loader)):
        targets = targets.to(device)
        images, norms, Hs = images.to(device), norms.to(device), Hs.to(device)

        optimizer.zero_grad()

        preds = model.forward(images=images, norms=norms, Hs=Hs)
        loss = loss_fn(preds, targets)
        loss.backward()

        # Record the training loss and the number of correct predictions
        batches_n += 1
        train_loss += loss.item()

        preds = preds.argmax(dim=1)
        nr_correct_batch = (preds == targets).sum()
        correct_preds += nr_correct_batch

        # Log batch metrics
        wandb.log({
            "Batch": batch_nr,
            "batch_loss/train": loss.item(),
            "batch_acc/train": nr_correct_batch / len(norms),
        })

        optimizer.step()

    train_loss /= batches_n
    accuracy = correct_preds / len(train_loader.dataset)

    return train_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    run_folder: str,
    epochs: int = 10,
    use_lr_scheduler: bool = False,
    train_function: Callable = train_epoch,
    eval_function: Callable = evaluate_model
):
    best_val_epoch = 0
    best_val_accuracy = 0

    if use_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=2)

    for epoch in range(epochs):
        logging.info(f"starting epoch {epoch}")

        train_loss, train_accuracy = train_function(
            model=model,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader
        )

        val_loss, val_accuracy, val_auc, _ = eval_function(
            model=model,
            test_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        # Log epoch metrics for train and validation
        wandb.log({
            "Epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "acc/train": train_accuracy,
            "acc/val": val_accuracy,
            "auc/val": val_auc
        })

        logging.info(f"the train accuracy was {train_accuracy} (loss: {train_loss})")
        logging.info(f"the validation accuracy was {val_accuracy} (loss: {val_loss})")

        # Pick the best model according to the validation
        # accuracy score
        if val_accuracy > best_val_accuracy:
            logging.info(f"found new best model at epoch {epoch} with accuracy {val_accuracy} (loss {val_loss})")

            best_val_epoch = epoch
            best_val_accuracy = val_accuracy

            wandb.run.summary["best_val_accuracy"] = best_val_accuracy
            wandb.run.summary["best_val_epoch"] = epoch

            # Save the model to disk
            torch.save(model.state_dict(), os.path.join(run_folder, f"model_{epoch}.pt"))
            wandb.run.summary["best_model_path"] = os.path.join(run_folder, f"model_{epoch}.pt")

        # Update the lr scheduler if needed
        if use_lr_scheduler:
            scheduler.step(val_loss)

        # save the optimizer to disk
        torch.save(optimizer.state_dict(), os.path.join(run_folder, f"optimizer.pt"))

    return best_val_epoch
