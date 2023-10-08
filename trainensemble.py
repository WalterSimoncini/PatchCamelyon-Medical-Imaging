import os
import json
import torch
import wandb
import logging
import argparse

import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from src.datasets.loaders import get_data_loader
from src.enums import PatchCamelyonSplit, ModelType, TransformType

from src.models import get_ensemble
from src.transforms import get_transform

from src.utils.train import train
from src.utils.eval import evaluate_model
from src.utils.seed import seed_everything
from src.utils.logging import configure_logging


def main(args):
    # Configure and create output directories
    # if they do not exist
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    run_folder = os.path.join(args.output_path, run_id)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(run_folder)

    loss_fn = nn.CrossEntropyLoss()
    with open(args.config, 'r') as file:
        config = json.load(file)
    model, sizes = get_ensemble(config=config)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_transform = get_transform(type_=args.transform, input_size=sizes[0]) # TODO make adaptive for mulitple input sizes
    test_transform = get_transform(type_=TransformType.EVALUATION, input_size=sizes[0])

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    run_config = vars(args) | {
        "device": device_name,
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "model": f"ensemble of {c['type_'] for c in config}"
    }

    # Convert the enum to a string so it can be serialized
    run_config["transform"] = run_config["transform"].value

    # Save the run configuration to a config.json file
    # in the run folder
    with open(os.path.join(run_folder, "config.json"), "w") as config_file:
        config_file.write(json.dumps(run_config))

    run = wandb.init(
        project="PatchCamelyon",
        entity="mi_ams",
        config=run_config
    )

    model = model.to(device)
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        data_dir=args.data_dir,
        data_key=args.data_key
    )

    logging.info(f"training model with weight decay of {args.wd}")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        run_folder=run_folder,
        epochs=args.epochs,
        device=device,
        use_lr_scheduler=args.lr_scheduler
    )

    _, test_accuracy, _ = evaluate_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    logging.info(f"the test accuracy was {test_accuracy}")
    wandb.run.summary["acc/test"] = test_accuracy

    run.finish()


def get_data_loaders(
    train_transform: nn.Module,
    test_transform: nn.Module,
    batch_size: int = 64,
    data_dir: str = "data",
    data_key: str = "x"
):
    train_loader = get_data_loader(
        split=PatchCamelyonSplit.TRAIN,
        batch_size=batch_size,
        transform=train_transform,
        data_dir=data_dir,
        data_key=data_key
    )

    val_loader = get_data_loader(
        split=PatchCamelyonSplit.VALIDATION,
        batch_size=batch_size,
        transform=test_transform,
        data_dir=data_dir,
        data_key=data_key
    )

    test_loader = get_data_loader(
        split=PatchCamelyonSplit.TEST,
        batch_size=batch_size,
        transform=test_transform,
        data_dir=data_dir,
        data_key=data_key
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    configure_logging()

    wandb.login()

    # Training Hyperparameters
    parser = argparse.ArgumentParser(description="Patch Camelyon Training")

    parser.add_argument("--output-path", default="runs", type=str, help="Path to save the model")
    parser.add_argument("--seed", default=42, type=int, help="The seed for random number generators")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size for training and validation")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--wd", default=0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train for")
    parser.add_argument("--config", default="configs/connected_ensemble/config.json", type=str, help="The configuration file for the ensemble")
    parser.add_argument("--transform", type=TransformType, choices=list(TransformType), required=True, help="The transform pipeline to be used for training")
    parser.add_argument("--data-dir", default="data", type=str, help="The directory containing the Patch Camelyon data")
    parser.add_argument("--data-key", default="x", type=str, help="The dataset key which contains the image data. Regular datasets have a single key 'x' and stain-normalized ones have ['norm', 'E', 'H']")
    parser.add_argument("--lr-scheduler", action=argparse.BooleanOptionalAction, help="Whether to use a learning rate scheduler")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
