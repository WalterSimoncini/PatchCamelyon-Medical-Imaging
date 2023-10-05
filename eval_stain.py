import logging
import argparse
import torch.nn as nn

from torch.utils.data import DataLoader

from src.models import get_model
from src.utils.misc import get_device
from src.transforms import get_transform
from src.enums import ModelType, TransformType
from src.utils.logging import configure_logging
from src.datasets.loaders import get_num_workers
from src.utils.eval import evaluate_model_stain_ensemble
from src.datasets import PatchCamelyonStainNormalizedDataset


def main(args):
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()

    # Here we assume that the four models all belong to the same model class/type
    image_model, input_size = get_model(type_=args.image_model, weights_path=args.image_model_path)
    norm_model, _ = get_model(type_=args.norm_model, weights_path=args.norm_model_path)
    H_model, _ = get_model(type_=args.H_model, weights_path=args.H_model_path)
    E_model, _ = get_model(type_=args.E_model, weights_path=args.E_model_path)

    # Move the models to the appropriate device
    norm_model = norm_model.to(device)
    image_model = image_model.to(device)
    H_model = H_model.to(device)
    E_model = E_model.to(device)

    dataset = PatchCamelyonStainNormalizedDataset(
        data_path=args.dataset_path,
        transform=get_transform(
            type_=TransformType.EVALUATION,
            input_size=input_size
        )
    )

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=get_num_workers()
    )

    test_loss, test_accuracy, test_auc = evaluate_model_stain_ensemble(
        image_model=image_model,
        norm_model=norm_model,
        H_model=H_model,
        E_model=E_model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device
    )

    logging.info(f"the test accuracy was {test_accuracy} (loss: {test_loss})")
    logging.info(f"the test auc was {test_auc}")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Patch Camelyon Evaluation (Stain Normalized Ensemble)")

    parser.add_argument("--dataset-path", type=str, help="Path to the test dataset", required=True)

    parser.add_argument("--image-model", type=ModelType, choices=list(ModelType), required=True, help="The image model type")
    parser.add_argument("--image-model-path", type=str, help="Path to the (regular) model weights", required=True)

    parser.add_argument("--norm-model", type=ModelType, choices=list(ModelType), required=True, help="The normalized model type")
    parser.add_argument("--norm-model-path", type=str, help="Path to the (normalized) model weights", required=True)

    parser.add_argument("--H-model", type=ModelType, choices=list(ModelType), required=True, help="The H model type")
    parser.add_argument("--H-model-path", type=str, help="Path to the (H) model weights", required=True)

    parser.add_argument("--E-model", type=ModelType, choices=list(ModelType), required=True, help="The E model type")
    parser.add_argument("--E-model-path", type=str, help="Path to the (E) model weights", required=True)

    main(parser.parse_args())
