import json
import torch
import logging
import argparse

from sklearn import metrics
from argparse import Namespace

from src.utils.logging import configure_logging
from src.datasets.loaders import get_data_loader
from src.enums import EnsembleStrategy, PatchCamelyonSplit, ModelType

from eval import main as evaluate_model


def main(args):
    positive_probs = []
    ensemble_model_configs = json.loads(open(args.ensemble_config).read())

    logging.info(f"number of models in the ensemble: {len(ensemble_model_configs)}")

    for config in ensemble_model_configs:
        logging.info(f"evaluating {config['model_path']} ({config['model']})")

        config = config | {
            "split": args.split,
            "tta_original_weight": None,
            "model": ModelType(config["model"])
        }

        positive_probs.append(torch.tensor(
            evaluate_model(args=Namespace(**config))
        ).unsqueeze(dim=0))

    # systems x targets tensor with positive probabilities 
    positive_probs = torch.cat(positive_probs, dim=0)

    if args.ensemble_strategy == EnsembleStrategy.MAJORITY:
        votes_for_true = (positive_probs >= 0.5).sum(dim=0)

        # In case of a tie, we predict 0
        predictions = (votes_for_true >= positive_probs.shape[0] // 2 + 1).to(torch.uint8)

        # Get the class probabilities for the AUC calculation
        pred_probs = votes_for_true / positive_probs.shape[0]
    elif args.ensemble_strategy == EnsembleStrategy.AVERAGE:
        predictions = positive_probs.mean(dim=0)
        predictions = (predictions >= 0.5).to(torch.uint8)

        # Get the class probabilities for the AUC calculation
        pred_probs = positive_probs.mean(dim=0)
    else:
        raise ValueError(f"invalid ensemble strategy {args.ensemble_strategy}")

    # Get the test labels
    test_loader = get_data_loader(split=PatchCamelyonSplit.TEST, batch_size=64, transform=None, shuffle=False)
    targets = torch.tensor(test_loader.dataset.targets[:].squeeze())

    correct_preds = (predictions == targets).sum()
    accuracy = correct_preds / len(targets)

    fpr, tpr, _ = metrics.roc_curve(targets, pred_probs, pos_label=1)

    logging.info(f"the test accuracy of the ensemble was {accuracy}")
    logging.info(f"the test auc with of the ensemble was {metrics.auc(fpr, tpr)}")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Ensemble Evaluation")

    parser.add_argument("--ensemble-strategy", type=EnsembleStrategy, choices=list(EnsembleStrategy), required=True, help="The type of ensembling to use")
    parser.add_argument("--ensemble-config", type=str, required=True, help="The ensemble json configuration file")
    parser.add_argument("--split", type=PatchCamelyonSplit, choices=list(PatchCamelyonSplit), required=True, help="The dataset split to test on, default test")

    main(parser.parse_args())
