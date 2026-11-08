import csv
import hashlib
import json

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from compare_blur import (
    BLUR_KERNEL_SIZE,
    BLUR_SIGMAS,
    CHECKPOINT_PATHS,
    apply_blur,
    load_checkpoint_model,
    normalize,
    validate_configs,
)
from train import PROJECT_DIR


REPORT_DIR = PROJECT_DIR / "reports/test_comparison"
EXPECTED_EPOCHS = {"vit": 17, "hybrid": 19}
REQUIRED_OUTPUTS = (
    "protocol.json",
    "results.csv",
    "per_image_predictions.csv",
    "paired_outcomes.csv",
    "per_class_recall.csv",
    "accuracy_vs_blur.png",
    "accuracy_drop_vs_blur.png",
)


class IndexedTestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_protocol(loaded):
    checkpoint_details = {}

    for model_name, _, checkpoint, parameter_count in loaded:
        path = CHECKPOINT_PATHS[model_name]

        if checkpoint["epoch"] != EXPECTED_EPOCHS[model_name]:
            raise ValueError(
                f"{model_name} checkpoint epoch is {checkpoint['epoch']}; "
                f"expected {EXPECTED_EPOCHS[model_name]}."
            )

        checkpoint_details[model_name] = {
            "path": str(path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "sha256": sha256(path),
            "epoch": checkpoint["epoch"],
            "model_config": checkpoint["config"]["model"],
            "trainable_parameters": parameter_count,
        }

    return {
        "protocol_version": 1,
        "checkpoints": checkpoint_details,
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "train": False,
            "images": 10000,
            "indices": "ordered integers 0 through 9999",
        },
        "preprocessing": {
            "tensor_conversion": "torchvision.transforms.ToTensor",
            "blur_input_range": "float tensors in [0, 1]",
            "normalization": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "formula": "(image - 0.5) / 0.5",
                "applications": 1,
            },
        },
        "blur": {
            "sigmas": list(BLUR_SIGMAS),
            "kernel_size": [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
            "sigma_zero": "identity operation",
        },
        "evaluation": {
            "batch_size": 128,
            "shuffle": False,
            "model_mode": "eval",
            "gradient_mode": "torch.inference_mode",
            "shared_model_input_tensor": True,
        },
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }


def validate_checkpoint_identifiers(loaded):
    if len(loaded) != len(EXPECTED_EPOCHS):
        raise RuntimeError("Did not load exactly one checkpoint for each model.")

    for expected_model_name, loaded_checkpoint in zip(
        CHECKPOINT_PATHS,
        loaded,
    ):
        model_name, _, _, _ = loaded_checkpoint

        if model_name != expected_model_name:
            raise ValueError(
                f"{CHECKPOINT_PATHS[expected_model_name]} identifies as "
                f"{model_name}, not {expected_model_name}."
            )


def print_saved_results():
    print("A complete matching test report already exists; reusing it.")

    with (REPORT_DIR / "results.csv").open(encoding="utf-8") as file:
        results = list(csv.DictReader(file))

    with (REPORT_DIR / "paired_outcomes.csv").open(encoding="utf-8") as file:
        paired_outcomes = list(csv.DictReader(file))

    verify_paired_outcomes(paired_outcomes, results)

    for row in results:
        print(
            f"{row['model']:6s} | sigma {float(row['sigma']):3.1f} | "
            f"accuracy {row['accuracy_percent']}% | "
            f"drop {row['accuracy_drop_percentage_points']} pp"
        )


def prepare_report_directory(protocol):
    output_exists = REPORT_DIR.exists() and any(REPORT_DIR.iterdir())

    if output_exists:
        completed = all((REPORT_DIR / name).exists() for name in REQUIRED_OUTPUTS)

        if completed:
            saved_protocol = json.loads(
                (REPORT_DIR / "protocol.json").read_text(encoding="utf-8")
            )

            if saved_protocol != protocol:
                raise RuntimeError(
                    "A completed test report exists but its protocol differs. "
                    "Refusing to overwrite it."
                )

            print_saved_results()
            return False

        raise RuntimeError(
            "An incomplete test report directory exists. Refusing to overwrite it."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Wrote frozen protocol:", REPORT_DIR / "protocol.json")
    return True


def collect_predictions(models, loader, sigma, device, class_names):
    records = []

    with torch.inference_mode():
        for images, labels, test_indices in loader:
            # blur and normalization happen once before both models use this batch.
            model_images = normalize(apply_blur(images, sigma)).to(device)
            predictions = {
                model_name: model(model_images).argmax(dim=1).cpu()
                for model_name, model in models.items()
            }

            for position in range(labels.size(0)):
                label = labels[position].item()
                vit_prediction = predictions["vit"][position].item()
                hybrid_prediction = predictions["hybrid"][position].item()
                records.append(
                    {
                        "test_index": test_indices[position].item(),
                        "true_label": label,
                        "true_class": class_names[label],
                        "sigma": sigma,
                        "vit_prediction": vit_prediction,
                        "hybrid_prediction": hybrid_prediction,
                        "vit_correct": vit_prediction == label,
                        "hybrid_correct": hybrid_prediction == label,
                    }
                )

    indices = {record["test_index"] for record in records}
    expected_indices = set(range(10000))

    if len(records) != 10000 or indices != expected_indices:
        raise RuntimeError(
            f"Sigma {sigma} did not contain exactly the 10,000 unique test indices."
        )

    return records


def accuracy(records, model_name):
    return sum(record[f"{model_name}_correct"] for record in records) / len(records)


def build_paired_outcomes(records_by_sigma):
    rows = []

    for sigma, records in records_by_sigma.items():
        counts = {
            "both_correct": 0,
            "only_vit_correct": 0,
            "only_hybrid_correct": 0,
            "both_incorrect": 0,
        }

        for record in records:
            if record["vit_correct"] and record["hybrid_correct"]:
                counts["both_correct"] += 1
            elif record["vit_correct"]:
                counts["only_vit_correct"] += 1
            elif record["hybrid_correct"]:
                counts["only_hybrid_correct"] += 1
            else:
                counts["both_incorrect"] += 1

        total = sum(counts.values())
        if total != 10000:
            raise RuntimeError(f"Paired counts at sigma {sigma} sum to {total}.")

        rows.append({"sigma": sigma, **counts, "total_images": total})

    return rows


def verify_paired_outcomes(paired_outcomes, results):
    result_lookup = {
        (row["model"], float(row["sigma"])): row
        for row in results
    }

    for row in paired_outcomes:
        sigma = float(row["sigma"])
        both_correct = int(row["both_correct"])
        vit_correct = both_correct + int(row["only_vit_correct"])
        hybrid_correct = both_correct + int(row["only_hybrid_correct"])
        total = int(row["total_images"])
        paired_total = (
            both_correct
            + int(row["only_vit_correct"])
            + int(row["only_hybrid_correct"])
            + int(row["both_incorrect"])
        )

        if paired_total != total or total != 10000:
            raise RuntimeError(
                f"Paired counts at sigma {sigma} sum to {paired_total}, "
                f"not {total}."
            )

        for model_name, correct in (("vit", vit_correct), ("hybrid", hybrid_correct)):
            result = result_lookup[model_name, sigma]
            reported_correct = round(float(result["accuracy"]) * total)

            if correct != reported_correct:
                raise RuntimeError(
                    f"Paired outcomes and {model_name} accuracy disagree at "
                    f"sigma {sigma}."
                )


def build_per_class_recall(records_by_sigma, class_names):
    rows = []

    for sigma, records in records_by_sigma.items():
        for model_name in ("vit", "hybrid"):
            for class_index, class_name in enumerate(class_names):
                class_records = [
                    record
                    for record in records
                    if record["true_label"] == class_index
                ]
                correct = sum(
                    record[f"{model_name}_correct"]
                    for record in class_records
                )
                recall = correct / len(class_records)
                rows.append(
                    {
                        "model": model_name,
                        "sigma": sigma,
                        "class_index": class_index,
                        "class_name": class_name,
                        "images_in_class": len(class_records),
                        "correct_predictions": correct,
                        "recall": recall,
                        "recall_percent": 100 * recall,
                    }
                )

    return rows


def build_results(records_by_sigma, checkpoints):
    rows = []
    clean_accuracies = {
        model_name: accuracy(records_by_sigma[0.0], model_name)
        for model_name in ("vit", "hybrid")
    }

    for sigma, records in records_by_sigma.items():
        for model_name in ("vit", "hybrid"):
            measured_accuracy = accuracy(records, model_name)
            rows.append(
                {
                    "model": model_name,
                    "checkpoint_epoch": checkpoints[model_name]["epoch"],
                    "sigma": sigma,
                    "images_evaluated": len(records),
                    "accuracy": measured_accuracy,
                    "accuracy_percent": 100 * measured_accuracy,
                    "accuracy_drop_percentage_points": 100
                    * (clean_accuracies[model_name] - measured_accuracy),
                }
            )

    return rows


def save_csv(path, rows, fieldnames, decimals):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            formatted = {**row}

            for fieldname, places in decimals.items():
                formatted[fieldname] = f"{row[fieldname]:.{places}f}"

            writer.writerow(formatted)


def save_plots(results):
    plot_settings = [
        (
            "accuracy_percent",
            "Test accuracy (%)",
            "accuracy_vs_blur.png",
        ),
        (
            "accuracy_drop_percentage_points",
            "Accuracy drop from clean (percentage points)",
            "accuracy_drop_vs_blur.png",
        ),
    ]

    for metric, ylabel, filename in plot_settings:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for model_name in ("vit", "hybrid"):
            model_rows = [row for row in results if row["model"] == model_name]
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
        fig.savefig(REPORT_DIR / filename, dpi=200)
        plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [
        load_checkpoint_model(path, device)
        for path in CHECKPOINT_PATHS.values()
    ]
    validate_checkpoint_identifiers(loaded)
    models = {model_name: model for model_name, model, _, _ in loaded}
    checkpoints = {
        model_name: checkpoint
        for model_name, _, checkpoint, _ in loaded
    }

    validate_configs(checkpoints)
    protocol = build_protocol(loaded)

    if not prepare_report_directory(protocol):
        return

    test_dataset = datasets.CIFAR10(
        root=str(PROJECT_DIR / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    indexed_dataset = IndexedTestDataset(test_dataset)

    if len(indexed_dataset) != 10000:
        raise RuntimeError(f"Expected 10,000 test images, found {len(indexed_dataset)}.")

    loader = DataLoader(
        indexed_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    records_by_sigma = {
        sigma: collect_predictions(models, loader, sigma, device, test_dataset.classes)
        for sigma in BLUR_SIGMAS
    }
    results = build_results(records_by_sigma, checkpoints)
    paired_outcomes = build_paired_outcomes(records_by_sigma)
    verify_paired_outcomes(paired_outcomes, results)
    per_class_recall = build_per_class_recall(
        records_by_sigma,
        test_dataset.classes,
    )

    save_csv(
        REPORT_DIR / "results.csv",
        results,
        results[0].keys(),
        {
            "accuracy": 6,
            "accuracy_percent": 2,
            "accuracy_drop_percentage_points": 2,
        },
    )
    save_csv(
        REPORT_DIR / "per_image_predictions.csv",
        [record for records in records_by_sigma.values() for record in records],
        next(iter(records_by_sigma.values()))[0].keys(),
        {},
    )
    save_csv(
        REPORT_DIR / "paired_outcomes.csv",
        paired_outcomes,
        paired_outcomes[0].keys(),
        {},
    )
    save_csv(
        REPORT_DIR / "per_class_recall.csv",
        per_class_recall,
        per_class_recall[0].keys(),
        {"recall": 6, "recall_percent": 2},
    )
    save_plots(results)

    print("Device:", device)
    for row in results:
        print(
            f"{row['model']:6s} | sigma {row['sigma']:3.1f} | "
            f"images {row['images_evaluated']} | "
            f"accuracy {row['accuracy_percent']:.2f}% | "
            f"drop {row['accuracy_drop_percentage_points']:.2f} pp"
        )
    print("Saved test report in:", REPORT_DIR)


if __name__ == "__main__":
    main()
