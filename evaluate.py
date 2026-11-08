import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train import make_datasets
from vit import TinyViT


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/vit_baseline")
    args = parser.parse_args()

    run_dir = PROJECT_DIR / args.run_dir
    checkpoint_path = run_dir / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # load the saved parameters rather than training again
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint["config"]["model"] != "tiny_vit_v1":
        raise ValueError("This evaluator expects the tiny ViT baseline.")

    model = TinyViT().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # recreate the same validation split and preprocessing
    _, val_dataset = make_datasets(
        checkpoint["config"]["train_size"]
    )

    class_names = val_dataset.dataset.classes
    num_classes = len(class_names)

    loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    example_images = []
    example_labels = []
    example_predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()

            # rows represent true classes and columns predictions
            pair_indices = labels * num_classes + predictions
            confusion += torch.bincount(
                pair_indices,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

            # retain the first twelve examples for inspection
            if len(example_images) < 12:
                remaining = 12 - len(example_images)

                example_images.extend(images[:remaining])
                example_labels.extend(labels[:remaining].tolist())
                example_predictions.extend(
                    predictions[:remaining].tolist()
                )

    accuracy = (
        confusion.diag().sum().item() / confusion.sum().item()
    )

    print("Device:", device)
    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print(f"Validation accuracy: {accuracy:.2%}")

    print("\nPer-class validation accuracy:")
    for index, name in enumerate(class_names):
        total = confusion[index].sum().item()
        correct = confusion[index, index].item()
        class_accuracy = correct / total if total else float("nan")
        print(f"  {name:12s}: {class_accuracy:.2%} ({total} images)")

    # show how errors are distributed across classes
    fig, ax = plt.subplots(figsize=(9, 8))
    matrix = ax.imshow(confusion.numpy(), cmap="Blues")

    ax.set_xticks(range(num_classes), class_names)
    ax.set_yticks(range(num_classes), class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Validation confusion matrix")

    fig.colorbar(matrix, ax=ax, label="Number of images")
    fig.tight_layout()
    fig.savefig(run_dir / "validation_confusion.png", dpi=200)

    # reverse training normalization for display
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))

    for ax, image, label, prediction in zip(
        axes.flat,
        example_images,
        example_labels,
        example_predictions,
    ):
        display_image = (image * 0.5 + 0.5).clamp(0, 1)
        ax.imshow(display_image.permute(1, 2, 0))

        correct = prediction == label
        ax.set_title(
            f"True: {class_names[label]}\n"
            f"Predicted: {class_names[prediction]}",
            color="green" if correct else "crimson",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle("First twelve validation examples")
    fig.tight_layout()
    fig.savefig(run_dir / "validation_predictions.png", dpi=200)

    print("\nSaved validation figures in:", run_dir)
    plt.show()


if __name__ == "__main__":
    main()