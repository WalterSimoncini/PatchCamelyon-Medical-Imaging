import torch
import logging
import argparse
import pickle, os

import torch.nn as nn

from src.utils.logging import configure_logging
from src.enums import PatchCamelyonSplit, ModelType, TransformType

from src.models import get_model
from src.transforms import get_transform
from src.datasets.loaders import get_data_loader
from src.utils.eval import evaluate_model, evaluate_model_tta


def main(args):
    # Verify that a transform was provided if TTA was chosen
    if args.tta and args.tta_transform is None:
        raise ValueError(f"No transform was provided for TTA")

    if args.tta:
        logging.info(f"running TTA with {args.tta_samples} samples")

    if args.tta and args.tta_original_weight:
        logging.info(f"running weighted TTA with weight: {args.tta_original_weight}")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    loss_fn = nn.CrossEntropyLoss()
    model, input_size = get_model(type_=args.model, weights_path=args.model_path)

    if args.tta:
        test_transform = None
    else:
        test_transform = get_transform(type_=TransformType.EVALUATION, input_size=input_size)

    test_loader = get_data_loader(split=args.split, batch_size=args.batch_size, transform=test_transform)

    model = model.to(device)

    if args.tta:
        test_loss, test_accuracy, test_auc, prediction_list = evaluate_model_tta(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            transform=get_transform(type_=args.tta_transform, input_size=input_size),
            default_transform=get_transform(type_=TransformType.EVALUATION, input_size=input_size),
            n_samples=args.tta_samples,
            original_image_weight=args.tta_original_weight,
        )
    else:
        test_loss, test_accuracy, test_auc, prediction_list = evaluate_model(model=model, test_loader=test_loader, loss_fn=loss_fn, device=device)

    logging.info(f"the test accuracy was {test_accuracy} (loss: {test_loss})")
    logging.info(f"the test auc was {test_auc}")

    # Save the predictions to a new file
    os.makedirs(args.save_predictions_path, exist_ok=True)
    c = 0
    f = os.path.join(args.save_predictions_path, f"test_predictions_{c}.pkl")
    if os.file.exists(f):
        while os.file.exists(f) and c < 1000:
            c += 1
            save_path = os.path.join(args.save_predictions_path, f"test_predictions_{c}.pkl")

    pickle.dump(prediction_list, open(save_path, "wb"))
    logging.info(f"saved predictions to {save_path}")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Patch Camelyon Evaluation")

    parser.add_argument("--model-path", default="runs", type=str, help="Path to the model weights", required=True)
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size for training and validation")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="The type of model to train/evaluate")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, help="Whether to use test-time augmentation")
    parser.add_argument("--tta-transform", type=TransformType, choices=list(TransformType), help="The transform used for TTA")
    parser.add_argument("--tta-samples", type=int, default=5, help="The number of TTA samples to be used")
    parser.add_argument("--tta-original-weight", type=float, default=None, help="The weight [0, 1] given to the original image for weighted TTA. If not specified all images are weighted equally")
    parser.add_argument("--split", type=PatchCamelyonSplit, choices=list(PatchCamelyonSplit), default=PatchCamelyonSplit.TEST, help="The dataset split to test on")
    parser.add_argument("--save-predictions-path", default="predictions", help="Save the predictions to this folder - needed for test set ensembling")
    args = parser.parse_args()

    main(args)
