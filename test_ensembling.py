import logging
import argparse
from argparse import Namespace
import pickle, os
from pathlib import Path
import numpy as np
from sklearn import metrics

from src.utils.logging import configure_logging
from src.enums import EnsembleStrategy, PatchCamelyonSplit, ModelType, TransformType
from src.datasets.loaders import get_data_loader

from eval import main as eval


def str2bool(v):
    """Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):
    eval_args = [
        Namespace(
            model_path=args.model_path[i],
            batch_size=args.batch_size[i],
            model=args.model[i],
            tta=args.tta[i],
            tta_transform=args.tta_transform[i],
            tta_samples=args.tta_samples[i],
            tta_original_weight=args.tta_original_weight[i],
            split=args.split[i],
        )
        for i in range(len(args.model_path))
    ]

    model_predictions, ground_truth = [], []
    for i in range(len(args.model_path)):
        predictions = eval(args=eval_args[i])
        model_predictions.append(predictions)

    logging.info(f"Nr of models in the ensemble: {len(model_predictions)}")

    assert len(set([len(p) for p in model_predictions])) == 1

    systems = np.array(model_predictions)  # shape: systems, test samples

    # get predictions as array
    if args.ensemble == EnsembleStrategy.MAJORITY:
        votes_for_true = (systems >= 0.5).sum(axis=0)
        # in case of a tie, we predict 0
        predictions = (votes_for_true >= len(systems) // 2 + 1).astype(np.int32)

    elif args.ensemble == EnsembleStrategy.AVERAGE:
        predictions = np.mean(systems, axis=0)
        predictions = (predictions >= 0.5).astype(np.int32)

    # get test labels as array
    test_loader = get_data_loader(split=PatchCamelyonSplit.TEST, batch_size=64, transform=None)
    test_labels = []
    for _, targets in test_loader:
        test_labels.extend(targets)
    targets = np.array(test_labels, dtype=np.int32)

    correct = (predictions == targets).sum()
    accuracy = correct / len(targets)
    fpr, tpr, _ = metrics.roc_curve(targets, predictions, pos_label=1)

    logging.info(f"the test accuracy of {args.ensemble} was {accuracy}")
    logging.info(f"the test auc with of {args.ensemble} was {metrics.auc(fpr, tpr)}")


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")

    parser.add_argument("--ensemble", type=EnsembleStrategy, choices=list(EnsembleStrategy), required=True, help="The type of ensembling to use")
    parser.add_argument("--model-path", nargs="+", type=str, required=True, help="Paths to the model weights")
    parser.add_argument("--batch-size", nargs="+", type=int, required=True, help="Batch size for training and validation")  # default=64
    parser.add_argument("--model", nargs="+", type=ModelType, choices=list(ModelType), required=True, help="The type of model to train/evaluate")
    parser.add_argument("--tta", nargs="+", type=str2bool, required=True, help="Whether to use test-time augmentation")
    parser.add_argument("--tta-transform", nargs="+", type=TransformType, choices=list(TransformType), required=True, help="The transform used for TTA")
    parser.add_argument("--tta-samples", nargs="+", type=int, required=True, help="The number of TTA samples to be used, default=5")
    parser.add_argument(
        "--tta-original-weight",
        nargs="+",
        type=float,
        required=True,
        help="The weight [0, 1] given to the original image for weighted TTA. If not specified all images are weighted equally, default=None",
    )
    parser.add_argument("--split", nargs="+", type=PatchCamelyonSplit, choices=list(PatchCamelyonSplit), required=True, help="The dataset split to test on, default test")

    args = parser.parse_args()

    assert len(set([len(args) for args in vars(args).values()])) == 1, "Number of arguments must be the same for all arguments"

    main(args)
