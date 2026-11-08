import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional

from train import PROJECT_DIR, build_model


CHECKPOINT_PATHS = {
    "vit": PROJECT_DIR / "runs/vit_baseline/best.pt",
    "hybrid": PROJECT_DIR / "runs/hybrid_baseline/best.pt",
}
MODEL_NAMES = {
    "tiny_vit_v1": "vit",
    "hybrid_vit_v1": "hybrid",
}
EXPECTED_SPLIT_SEED = 42
EXPECTED_NORMALIZATION = "mean_0.5_std_0.5"
EXPECTED_CLEAN_ACCURACY = {
    "vit": 0.5422,
    "hybrid": 0.5148,
}
BLUR_SIGMAS = (0.0, 0.5, 1.0, 1.5)
BLUR_KERNEL_SIZE = 9


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model_name = MODEL_NAMES.get(config["model"])

    if model_name is None:
        raise ValueError(f"Unsupported checkpoint model: {config['model']}")

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return model_name, model, checkpoint, parameter_count


def validate_configs(checkpoints):
    train_sizes = set()

    for model_name, checkpoint in checkpoints.items():
        config = checkpoint["config"]

        if config.get("split_seed") != EXPECTED_SPLIT_SEED:
            raise ValueError(
                f"{model_name} uses split seed {config.get('split_seed')}; "
                f"expected {EXPECTED_SPLIT_SEED}."
            )

        if config.get("normalization") != EXPECTED_NORMALIZATION:
            raise ValueError(
                f"{model_name} uses normalization "
                f"{config.get('normalization')!r}; "
                f"expected {EXPECTED_NORMALIZATION!r}."
            )

        train_sizes.add(config["train_size"])

    if len(train_sizes) != 1:
        raise ValueError("The checkpoints do not use the same training split.")

    return train_sizes.pop()


def make_unnormalized_validation_dataset(train_size):
    # this repeats train.make_datasets' fixed permutation but keeps pixels in [0, 1].
    dataset = datasets.CIFAR10(
        root=str(PROJECT_DIR / "data"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    generator = torch.Generator().manual_seed(EXPECTED_SPLIT_SEED)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    return Subset(dataset, indices[-5000:])


def apply_blur(images, sigma):
    if sigma == 0:
        return images

    return transform_functional.gaussian_blur(
        images,
        kernel_size=[BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
        sigma=[sigma, sigma],
    )


def normalize(images):
    return (images - 0.5) / 0.5


def evaluate_condition(models, loader, sigma, device):
    correct = {model_name: 0 for model_name in models}
    image_count = 0

    with torch.inference_mode():
        for images, labels in loader:
            # blur operates on the unnormalized [0, 1] tensors shared by both models.
            model_images = normalize(apply_blur(images, sigma)).to(device)
            labels = labels.to(device)

            for model_name, model in models.items():
                predictions = model(model_images).argmax(dim=1)
                correct[model_name] += (predictions == labels).sum().item()

            image_count += labels.size(0)

    return image_count, {
        model_name: count / image_count
        for model_name, count in correct.items()
    }


def save_results_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "checkpoint_epoch",
                "sigma",
                "images_evaluated",
                "accuracy",
                "accuracy_percent",
                "accuracy_drop_percentage_points",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "accuracy": f"{row['accuracy']:.6f}",
                    "accuracy_percent": f"{row['accuracy_percent']:.2f}",
                    "accuracy_drop_percentage_points": (
                        f"{row['accuracy_drop_percentage_points']:.2f}"
                    ),
                }
            )


def save_accuracy_plots(rows, report_dir):
    for metric, ylabel, filename in [
        ("accuracy_percent", "Validation accuracy (%)", "accuracy_vs_blur.png"),
        (
            "accuracy_drop_percentage_points",
            "Accuracy drop from clean (percentage points)",
            "accuracy_drop_vs_blur.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for model_name in MODEL_NAMES.values():
            model_rows = [row for row in rows if row["model"] == model_name]
            ax.plot(
                [row["sigma"] for row in model_rows],
                [row[metric] for row in model_rows],
                marker="o",
                linewidth=2,
                label=model_name,
            )

        ax.set_xlabel("Gaussian blur sigma (9x9 kernel)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(BLUR_SIGMAS)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Model")
        fig.tight_layout()
        fig.savefig(report_dir / filename, dpi=200)
        plt.close(fig)


def save_example_images(dataset, report_dir):
    images, labels = next(
        iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0))
    )
    class_names = dataset.dataset.classes
    fig, axes = plt.subplots(len(BLUR_SIGMAS), len(images), figsize=(10, 9))

    for row_index, sigma in enumerate(BLUR_SIGMAS):
        blurred_images = apply_blur(images, sigma)

        for column_index, (image, label) in enumerate(
            zip(blurred_images, labels)
        ):
            ax = axes[row_index, column_index]
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")

            if row_index == 0:
                ax.set_title(class_names[label.item()])
            if column_index == 0:
                ax.set_ylabel(f"sigma {sigma:g}", rotation=0, labelpad=40)

    fig.suptitle("Same validation images at each Gaussian blur strength")
    fig.tight_layout()
    fig.savefig(report_dir / "example_blurred_images.png", dpi=200)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [
        load_checkpoint_model(path, device)
        for path in CHECKPOINT_PATHS.values()
    ]
    models = {model_name: model for model_name, model, _, _ in loaded}
    checkpoints = {
        model_name: checkpoint
        for model_name, _, checkpoint, _ in loaded
    }

    train_size = validate_configs(checkpoints)
    validation_dataset = make_unnormalized_validation_dataset(train_size)
    loader = DataLoader(
        validation_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    report_dir = PROJECT_DIR / "reports/blur_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Validation images:", len(validation_dataset))
    for model_name, _, checkpoint, parameter_count in loaded:
        print(
            f"{model_name}: checkpoint epoch {checkpoint['epoch']}, "
            f"trainable parameters {parameter_count:,}"
        )

    rows = []
    clean_accuracies = {}

    for sigma in BLUR_SIGMAS:
        image_count, accuracies = evaluate_condition(
            models, loader, sigma, device
        )

        for model_name, accuracy in accuracies.items():
            if sigma == 0:
                clean_accuracies[model_name] = accuracy

            accuracy_drop = 100 * (clean_accuracies[model_name] - accuracy)
            row = {
                "model": model_name,
                "checkpoint_epoch": checkpoints[model_name]["epoch"],
                "sigma": sigma,
                "images_evaluated": image_count,
                "accuracy": accuracy,
                "accuracy_percent": 100 * accuracy,
                "accuracy_drop_percentage_points": accuracy_drop,
            }
            rows.append(row)
            print(
                f"{model_name:6s} | sigma {sigma:3.1f} | "
                f"images {image_count} | accuracy {accuracy:.2%} | "
                f"drop {accuracy_drop:.2f} pp"
            )

    print("\nClean accuracy check:")
    for model_name, accuracy in clean_accuracies.items():
        expected = EXPECTED_CLEAN_ACCURACY[model_name]
        discrepancy = 100 * (accuracy - expected)
        status = "matches approximately" if abs(discrepancy) <= 0.1 else "DISCREPANCY"
        print(
            f"{model_name}: {accuracy:.2%} vs expected {expected:.2%} "
            f"({discrepancy:+.2f} pp, {status})"
        )

    save_results_csv(rows, report_dir / "results.csv")
    save_accuracy_plots(rows, report_dir)
    save_example_images(validation_dataset, report_dir)
    print("\nSaved reports in:", report_dir)


if __name__ == "__main__":
    main()
