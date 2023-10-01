import logging
import argparse
import pickle, os
from pathlib import Path
import numpy as np
from sklearn import metrics

from src.utils.logging import configure_logging
from src.enums import EnsembleType, PatchCamelyonSplit
from src.datasets.loaders import get_data_loader


def main(args):
    args.prediction_folder = Path(args.prediction_folder)
    prediction_files = list(args.prediction_folder.iterdir())

    print(f'Nr of predictions {len(prediction_files)}')
    
    systems = [pickle.load(open(f, "rb")) for f in prediction_files]
    assert len(set([len(p) for p in systems])) == 1
    
    systems = np.array(systems)  # shape: systems, test samples
    
    print(f'first system: predictions: {systems[0][:10]}')    

    # get predictions as array
    if args.ensemble == EnsembleType.MAJORITY:
        votes_for_true = (systems >= 0.5).sum(axis=0)
        # in case of a tie, we predict 0
        predictions = (votes_for_true >= len(systems) // 2 + 1).astype(np.int32)  

    elif args.ensemble == EnsembleType.AVERAGE:
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

    logging.info(f"the test accuracy {args.ensemble} was {accuracy}")
    logging.info(f"the test auc with {args.ensemble} was {metrics.auc(fpr, tpr)}")
    
    
if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")

    parser.add_argument("--prediction-folder", default="predictions", type=str, help="Path to the folder containing the predictions", required=False)
    parser.add_argument("--ensemble", type=EnsembleType, choices=list(EnsembleType), required=True, help="The type of ensembling to use")

    args = parser.parse_args()

    main(args)
