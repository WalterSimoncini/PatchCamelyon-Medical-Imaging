import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn import metrics

from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from src.enums import EnsembleStrategy


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    model.eval()

    ground_truth = np.zeros(len(test_loader.dataset))
    true_class_probs = np.zeros(len(test_loader.dataset))

    batch_size = test_loader.batch_size
    test_loss, correct_preds, batches_n = 0, 0, 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            images, targets = images.to(device), targets.to(device)

            preds = model(images)
            test_loss += loss_fn(preds, targets)

            # Update the arrays used to compute the ROC and AUC
            ground_truth[(i * batch_size):((i + 1) * batch_size)] = targets.cpu().numpy()
            true_class_probs[(i * batch_size):((i + 1) * batch_size)] = softmax(preds, dim=1)[:, 1].cpu().numpy()

            batches_n += 1
            preds = preds.argmax(dim=1)
            correct_preds += (preds == targets).sum()

    test_loss /= batches_n
    accuracy = correct_preds / len(test_loader.dataset)

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth,
        true_class_probs,
        pos_label=1
    )

    return test_loss, accuracy, metrics.auc(fpr, tpr), true_class_probs


def evaluate_model_ensemble(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    model.eval()

    ground_truth = np.zeros(len(test_loader.dataset))
    true_class_probs = np.zeros(len(test_loader.dataset))

    batch_size = test_loader.batch_size
    test_loss, correct_preds, batches_n = 0, 0, 0

    with torch.no_grad():
        # The E images are not used as they do not yield good representations
        for i, (images, norms, Hs, Es, targets) in enumerate(tqdm(test_loader)):
            targets = targets.to(device)
            images, norms, Hs = images.to(device), norms.to(device), Hs.to(device)

            preds = model.forward(images=images, norms=norms, Hs=Hs)
            test_loss += loss_fn(preds, targets)

            # Update the arrays used to compute the ROC and AUC
            ground_truth[(i * batch_size):((i + 1) * batch_size)] = targets.cpu().numpy()
            true_class_probs[(i * batch_size):((i + 1) * batch_size)] = softmax(preds, dim=1)[:, 1].cpu().numpy()

            batches_n += 1
            preds = preds.argmax(dim=1)
            correct_preds += (preds == targets).sum()

    test_loss /= batches_n
    accuracy = correct_preds / len(test_loader.dataset)

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth,
        true_class_probs,
        pos_label=1
    )

    return test_loss, accuracy, metrics.auc(fpr, tpr), true_class_probs


def evaluate_model_stain_ensemble(
    image_model: nn.Module,
    norm_model: nn.Module,
    H_model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    strategy: EnsembleStrategy = EnsembleStrategy.AVERAGE
):
    # Set models in evaluation mode
    for model in [image_model, norm_model, H_model]:
        model.eval()

    test_loss, correct_preds = 0, 0
    ground_truth = np.zeros(len(test_loader.dataset))
    true_class_probs = np.zeros(len(test_loader.dataset))

    with torch.no_grad():
        for i, (images, norms, Hs, Es, targets) in enumerate(tqdm(test_loader)):
            targets = targets.to(device)

            # FIXME: We can probably remove the Es
            # as they are not being used
            Hs, Es = Hs.to(device), Es.to(device)
            images, norms = images.to(device), norms.to(device)

            if norms.sum() == 0:
                # If the normalization failed the norm, H and E
                # images will be zero-valued tensors, thus we
                # fall back to using only the original image
                preds = image_model(images)
                test_loss += loss_fn(preds, targets)
                preds = softmax(preds, dim=1)
            else:
                # Run a forward pass with all models
                preds = torch.cat([
                    norm_model(norms),
                    H_model(Hs),
                    image_model(images)
                ], dim=0)

                if strategy == EnsembleStrategy.AVERAGE:
                    # Average out the predictions
                    preds = softmax(preds, dim=1).mean(dim=0).unsqueeze(dim=0)
                elif strategy == EnsembleStrategy.MAJORITY:
                    # Take the argmaxes, count the number of positive predictions
                    # and create a tensor [1 - pos_preds/3, pos_preds/3]
                    pos_prob = preds.argmax(dim=1).sum() / 3
                    preds = torch.tensor([1 - pos_prob, pos_prob], device=device).unsqueeze(dim=0)
                else:
                    raise ValueError(f"Ensembling strategy {strategy} is not supported")

                test_loss += loss_fn(preds, targets)

            test_loss += loss_fn(preds, targets)

            ground_truth[i] = targets.cpu()[0].item()
            true_class_probs[i] = preds[:, 1].cpu().item()

            preds = preds.argmax(dim=1)
            correct_preds += (preds == targets).sum()

    test_loss /= len(test_loader.dataset)
    accuracy = correct_preds / len(test_loader.dataset)

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth,
        true_class_probs,
        pos_label=1
    )

    return test_loss, accuracy, metrics.auc(fpr, tpr), true_class_probs


def evaluate_model_tta(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    transform: nn.Module,
    default_transform: nn.Module,
    n_samples: int = 5,
    original_image_weight: float = None
):
    """
        Evaluates the model using TTA (Test-Time Augmentation).

        For each image this evaluator generates n_samples using the
        given transform and averages the model prediction over them

        TODO: allow the user to specify multiple transformation

        :param transform: the transform used for TTA
        :param default_transform: the default transformation applied to
                                test samples if TTA was not to be used
    """
    batch_size = test_loader.batch_size

    if batch_size != 1:
        raise ValueError(f"the data loader batch size must be 1: {batch_size} given")

    model.eval()

    test_loss, correct_preds = 0, 0
    ground_truth = np.zeros(len(test_loader.dataset))
    true_class_probs = np.zeros(len(test_loader.dataset))

    with torch.no_grad():
        for i, (image, target) in enumerate(tqdm(test_loader)):
            image, target = image.to(device), target.to(device)

            if image.mean() > 0.85:
                # For very bright images (e.g. white or almost) skip
                # augmentations as they may transform images such that
                # they generate NaN values when normalized
                transformed_targets = target
                transformed_images = image
            else:
                # Generate n transformations of the given image, plus the
                # original image with the default transformation
                transformed_targets = target.repeat(n_samples + 1)
                transformed_images = torch.cat(
                    [default_transform(image)] + [transform(image) for _ in range(n_samples)],
                    dim=0
                ).to(device)

            # Predict the labels for all transformed
            # images and compute the mean prediction
            preds = model(transformed_images)
            test_loss += loss_fn(preds, transformed_targets)
            preds = softmax(preds, dim=1)

            if original_image_weight is None:
                preds = preds.mean(dim=0)
            else:
                transformed_image_weight = (1 - original_image_weight) / n_samples

                image_weights = torch.zeros(n_samples + 1) + transformed_image_weight
                image_weights[0] = original_image_weight
                image_weights = image_weights.repeat(2, 1).T

                preds = preds.sum(dim=0)

            ground_truth[i] = target.cpu().item()
            # Get the positive class probability for the AUC computation
            true_class_probs[i] = preds[1].cpu().item()

            # Add one to the correct preds if the predicted
            # label and the target match
            target_label = target.squeeze()
            predicted_label = preds.argmax(dim=0)

            if predicted_label == target_label:
                correct_preds += 1

    test_loss /= len(test_loader.dataset)
    accuracy = correct_preds / len(test_loader.dataset)

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth,
        true_class_probs,
        pos_label=1
    )

    return test_loss, accuracy, metrics.auc(fpr, tpr), true_class_probs
