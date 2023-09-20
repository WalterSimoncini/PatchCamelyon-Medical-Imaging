import os
import torch
import wandb
import logging
import argparse

import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torchvision import transforms

from src.datasets.loaders import get_data_loader
from src.enums import PatchCamelyonSplit, ModelType

from src.models import get_model
from src.utils.train import train
from src.utils.eval import evaluate_model
from src.utils.logging import configure_logging


def main(args):
    # Configure and create output directories
    # if they do not exist
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    run_folder = os.path.join(args.output_path, run_id)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(run_folder)

    loss_fn = nn.CrossEntropyLoss()
    model = get_model(type_=args.model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    run = wandb.init(
        project="PatchCamelyon",
        entity="mi_ams",
        config=vars(args) | {
            "device": device_name,
            "optimizer": optimizer.__class__.__name__,
            "loss_fn": loss_fn.__class__.__name__,
            "model": args.model.value
        }
    )

    model = model.to(device)
    train_loader, val_loader, test_loader = get_data_loaders()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        run_folder=run_folder,
        epochs=args.epochs,
        device=device
    )

    _, test_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    logging.info(f"the test accuracy was {test_accuracy}")
    wandb.run.summary["acc/test"] = test_accuracy

    run.finish()


def get_data_loaders(batch_size: int = 64):
    config = {
        "transform": transforms.Resize(224, antialias=True),
        "batch_size": batch_size
    }

    train_loader = get_data_loader(split=PatchCamelyonSplit.TRAIN, **config)
    val_loader = get_data_loader(split=PatchCamelyonSplit.VALIDATION, **config)
    test_loader = get_data_loader(split=PatchCamelyonSplit.TEST, **config)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    configure_logging()
    wandb.login()

    # Training Hyperparameters
    parser = argparse.ArgumentParser(description="MI Project (Patch Camelyon)")

    parser.add_argument("--output_path", default="runs", type=str, help="Path to save the model")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for training and validation")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train for")
    parser.add_argument('--model', type=ModelType, choices=list(ModelType), required=True)

    args = parser.parse_args()

    main(args)
