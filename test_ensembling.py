import logging
import argparse
import pickle, os
from pathlib import Path
import numpy as np

from src.utils.logging import configure_logging
from src.enums import EnsembleType

from src.models import get_model
from src.transforms import get_transform
from src.datasets.loaders import get_data_loader
from src.utils.eval import evaluate_model, evaluate_model_tta

from sklearn import metrics


def main(args):
    args.prediction_folder = Path(args.prediction_folder)
    prediction_files = list(args.prediction_folder.iterdir())

    systems = [pickle.load(open(f, "rb")) for f in prediction_files]
    assert len(set([len(p) for p in systems])) == 1

    # get predictions as array
    if args.ensemble == EnsembleType.MAJORITY_VOTE:
        predictions = [s.argmax(axis=0) for s in systems]
        predictions = np.stack(predictions, axis=1)
        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)  # TODO: needs to be tested

    elif args.ensemble == EnsembleType.AVERAGE:
        predictions = np.stack(systems, axis=1)
        predictions = np.mean(predictions, axis=1)
        # predictions = predictions.argmax(axis=1)

    # get test labels as array
    test_loader = get_data_loader(split=args.split, batch_size=args.batch_size, transform=None)
    test_labels = []
    for _, (_, targets) in test_loader:
        test_labels.extend(targets)
    targets = np.array(test_labels, dtype=np.int32)

    correct = (predictions == targets).sum()
    accuracy = correct / len(targets)
    fpr, tpr, _ = metrics.roc_curve(targets, predictions, pos_label=1)

    logging.info(f"the test accuracy with ensembling was {accuracy}")
    logging.info(f"the test auc with ensembling was {metrics.auc(fpr, tpr)}")


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")

    parser.add_argument("--prediction-folder", default="runs", type=str, help="Path to the folder containing the predictions", required=True)
    parser.add_argument("--ensemble", type=EnsembleType, choices=list(EnsembleType), required=True, help="The type of ensembling to use")

    args = parser.parse_args()

    main(args)
