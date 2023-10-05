import os
import torch
import argparse
import numpy as np

from typing import List
from torchvision import transforms
from datasets import load_from_disk, load_metric
from transformers import Trainer, TrainingArguments
from transformers import ViTImageProcessor, ViTForImageClassification


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset preprocessor
    processor = ViTImageProcessor.from_pretrained(args.model_path)

    # Load the dataset
    dataset = load_from_disk(args.dataset_path)

    # Load the model
    model = ViTForImageClassification.from_pretrained(
        args.model_path,
        num_labels=2
    )

    # Configure the training process and start it!
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        fp16=False,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=10,
        learning_rate=args.lr,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True
    )

    train_dataset = dataset["train"]
    train_dataset.set_transform(
        get_transform(processor=processor, include_augmentations=True)
    )

    eval_dataset = dataset["valid"]
    eval_dataset.set_transform(get_transform(processor=processor))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=get_metrics_func(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=processor
    )

    train_results = trainer.train()

    # Save the best model (since we load the best one
    # at the end of training)
    trainer.save_model()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    trainer.save_state()

    # Evaluate the model on the validation set
    valid_metrics = trainer.evaluate(dataset["valid"])

    trainer.log_metrics("valid", valid_metrics)
    trainer.save_metrics("valid", valid_metrics)

    # Evaluate the model on the test set
    test_metrics = trainer.evaluate(dataset["test"])

    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)


def get_transform(processor: ViTImageProcessor, include_augmentations: bool = False):
    # No resizing as that is done by the processor itself
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
        transforms.RandomRotation(degrees=360)
    ])

    def transform(batch: dict):
        if include_augmentations:
            images = [augmentation_transform(x) for x in batch["image"]]
        else:
            images = [x for x in batch["image"]]

        inputs = processor(images, return_tensors="pt")
        inputs["label"] = batch["label"]

        return inputs

    return transform


def collate_fn(batch: List[dict]):
  return {
      "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
      "labels": torch.tensor([x["label"] for x in batch])
  }


def get_metrics_func():
    accuracy_metric = load_metric("accuracy")

    def metrics_func(pred):
        return accuracy_metric.compute(
            predictions=np.argmax(pred.predictions, axis=1),
            references=pred.label_ids
        )

    return metrics_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a ViT on the PCam dataset")

    parser.add_argument("--dataset-path", type=str, help="Path to PCam dataset in arrow format")
    parser.add_argument("--model-path", type=str, help="Path to the MAE checkpoint")
    parser.add_argument("--output-dir", type=str, help="Path to where models should be stored")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size for training and validation")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs to train for")
    parser.add_argument("--save-steps", default=500, type=int, help="Save the model every n steps")
    parser.add_argument("--eval-steps", default=500, type=int, help="Evaluate the model every n steps")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate")

    main(parser.parse_args())
