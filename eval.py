import logging
import argparse

import torch.nn as nn

from src.utils.logging import configure_logging
from src.enums import PatchCamelyonSplit, ModelType, TransformType

from src.models import get_model
from src.utils.misc import get_device
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

    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    model, input_size = get_model(type_=args.model, weights_path=args.model_path)

    if args.tta:
        test_transform = None
    else:
        test_transform = get_transform(
            type_=TransformType.EVALUATION,
            input_size=input_size
        )

    test_loader = get_data_loader(
        split=args.split,
        batch_size=args.batch_size,
        transform=test_transform,
        data_dir=args.data_dir,
        data_key=args.data_key
    )

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
            original_image_weight=args.tta_original_weight
        )
    else:
        test_loss, test_accuracy, test_auc, prediction_list = evaluate_model(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device
        )

    logging.info(f"the test accuracy was {test_accuracy} (loss: {test_loss})")
    logging.info(f"the test auc was {test_auc}")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Patch Camelyon Evaluation")

    parser.add_argument("--model-path", default="runs", type=str, help="Path to the model weights", required=True)
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size for training and validation")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="The type of model to evaluate")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, help="Whether to use test-time augmentation")
    parser.add_argument("--tta-transform", type=TransformType, choices=list(TransformType), help="The transform used for TTA")
    parser.add_argument("--tta-samples", type=int, default=5, help="The number of TTA samples to be used")
    parser.add_argument("--tta-original-weight", type=float, default=None, help="The weight [0, 1] given to the original image for weighted TTA. If not specified all images are weighted equally")
    parser.add_argument("--split", type=PatchCamelyonSplit, choices=list(PatchCamelyonSplit), default=PatchCamelyonSplit.TEST, help="The dataset split to test on")
    parser.add_argument("--data-dir", default="data", type=str, help="The directory containing the Patch Camelyon data")
    parser.add_argument("--data-key", default="x", type=str, help="The dataset key which contains the image data. Regular datasets have a single key 'x' and stain-normalized ones have ['norm', 'E', 'H']")

    args = parser.parse_args()

    main(args)
