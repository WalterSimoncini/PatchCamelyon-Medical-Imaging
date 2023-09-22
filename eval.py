import torch
import logging
import argparse

import torch.nn as nn

from src.utils.logging import configure_logging
from src.enums import PatchCamelyonSplit, ModelType, TransformType

from src.models import get_model
from src.transforms import get_transform
from src.utils.eval import evaluate_model
from src.datasets.loaders import get_data_loader


def main(args):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    loss_fn = nn.CrossEntropyLoss()
    model, input_size = get_model(type_=args.model, weights_path=args.model_path)

    test_loader = get_data_loader(
        split=PatchCamelyonSplit.TEST,
        batch_size=args.batch_size,
        transform=get_transform(
            type_=TransformType.EVALUATION,
            input_size=input_size
        )
    )

    model = model.to(device)

    test_loss, test_accuracy, test_auc = evaluate_model(
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
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="The type of model to train/evaluate")

    args = parser.parse_args()

    main(args)
