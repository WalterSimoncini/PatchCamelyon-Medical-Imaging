"""
    This script is used to train a model ensemble
    composed of feature extractors for regular images,
    stain-normalized images and the Hematoxylin component.
    
    The ensemble extracts the features and aggregates them
    using an MLP as the fusion module.

    We provide a separate training script as there are some
    small adjustments to be made for the fusion module, e.g.
    loading a dataset with data (image, norm, H, E) and targets,
    contained in a single HDF5 file.
"""
import os
import json
import torch
import wandb
import logging
import argparse

import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from src.datasets.loaders import get_num_workers
from src.enums import PatchCamelyonSplit, TransformType

from torch.utils.data import DataLoader
from src.transforms import get_transform
from src.models import get_stain_fusion_model

from src.utils.seed import seed_everything
from src.utils.logging import configure_logging
from src.utils.eval import evaluate_model_ensemble
from src.utils.train import train, train_epoch_stain_ensemble
from src.datasets import PatchCamelyonStainNormalizedDataset


def main(args):
    # Configure and create output directories
    # if they do not exist
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    run_folder = os.path.join(args.output_path, run_id)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(run_folder)

    loss_fn = nn.CrossEntropyLoss()
    model, input_size = get_stain_fusion_model(
        config=json.loads(open(args.config).read())
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_transform = get_transform(type_=args.transform, input_size=input_size)
    test_transform = get_transform(type_=TransformType.EVALUATION, input_size=input_size)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    run_config = vars(args) | {
        "device": device_name,
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "model": "stain-ensemble"
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
        data_dir=args.data_dir
    )

    logging.info(f"training model with weight decay of {args.wd}")

    best_val_epoch = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        run_folder=run_folder,
        epochs=args.epochs,
        device=device,
        use_lr_scheduler=args.lr_scheduler,
        train_function=train_epoch_stain_ensemble,
        eval_function=evaluate_model_ensemble
    )

    # Load the best model to evaluate it on the test set
    model.load_state_dict(
        torch.load(os.path.join(run_folder, f"model_{best_val_epoch}.pt"))
    )

    _, test_accuracy, _ = evaluate_model_ensemble(
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
    data_dir: str = "data"
):
    train_dataset = dataset_for_split(split=PatchCamelyonSplit.TRAIN, transform=train_transform, data_dir=data_dir)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    val_dataset = dataset_for_split(split=PatchCamelyonSplit.VALIDATION, transform=test_transform, data_dir=data_dir)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    test_dataset = dataset_for_split(split=PatchCamelyonSplit.TEST, transform=test_transform, data_dir=data_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def dataset_for_split(split: PatchCamelyonSplit, data_dir: str, transform: nn.Module):
    # The datasets assumes that both data and targets are in the same HDF5 file
    return PatchCamelyonStainNormalizedDataset(
        data_path=os.path.join(data_dir, f"camelyonpatch_level_2_split_{split.value}_x.h5"),
        transform=transform
    )


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
    parser.add_argument("--config", type=str, required=True, help="Path to the json config for the stain models")
    parser.add_argument("--transform", type=TransformType, choices=list(TransformType), required=True, help="The transform pipeline to be used for training")
    parser.add_argument("--data-dir", default="data/stain", type=str, help="The directory containing the Patch Camelyon data")
    parser.add_argument("--lr-scheduler", action=argparse.BooleanOptionalAction, help="Whether to use a learning rate scheduler")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
