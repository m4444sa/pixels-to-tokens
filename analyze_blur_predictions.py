import csv

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from compare_blur import (
    BLUR_SIGMAS,
    CHECKPOINT_PATHS,
    apply_blur,
    load_checkpoint_model,
    make_unnormalized_validation_dataset,
    normalize,
    validate_configs,
)
from train import PROJECT_DIR


EXPECTED_ACCURACIES = {
    0.0: {"vit": 0.5422, "hybrid": 0.5148},
    0.5: {"vit": 0.5234, "hybrid": 0.5022},
    1.0: {"vit": 0.4258, "hybrid": 0.4544},
    1.5: {"vit": 0.3780, "hybrid": 0.4062},
}


class IndexedValidationDataset(Dataset):
    def __init__(self, validation_dataset):
        self.validation_dataset = validation_dataset

    def __len__(self):
        return len(self.validation_dataset)

    def __getitem__(self, index):
        image, label = self.validation_dataset[index]
        original_index = self.validation_dataset.indices[index]

        return image, label, original_index


def collect_predictions(models, loader, sigma, device, class_names):
    records = []

    with torch.inference_mode():
        for images, labels, original_indices in loader:
            # both models receive this same blurred, normalized batch.
            model_images = normalize(apply_blur(images, sigma)).to(device)
            predictions = {
                model_name: model(model_images).argmax(dim=1).cpu()
                for model_name, model in models.items()
            }

            for position in range(labels.size(0)):
                true_label = labels[position].item()
                vit_prediction = predictions["vit"][position].item()
                hybrid_prediction = predictions["hybrid"][position].item()
                records.append(
                    {
                        "original_index": original_indices[position].item(),
                        "true_label": true_label,
                        "true_class": class_names[true_label],
                        "sigma": sigma,
                        "vit_prediction": vit_prediction,
                        "hybrid_prediction": hybrid_prediction,
                        "vit_correct": vit_prediction == true_label,
                        "hybrid_correct": hybrid_prediction == true_label,
                    }
                )

    return records


def accuracy(records, model_name):
    return sum(record[f"{model_name}_correct"] for record in records) / len(records)


def verify_aggregate_accuracies(records_by_sigma):
    for sigma, records in records_by_sigma.items():
        for model_name, expected in EXPECTED_ACCURACIES[sigma].items():
            measured = accuracy(records, model_name)
            discrepancy = 100 * (measured - expected)

            if abs(discrepancy) > 0.001:
                raise RuntimeError(
                    f"{model_name} at sigma {sigma} measured {measured:.2%}, "
                    f"expected {expected:.2%} ({discrepancy:+.3f} pp)."
                )

            print(
                f"{model_name:6s} | sigma {sigma:3.1f} | "
                f"accuracy {measured:.2%} | expected {expected:.2%} | "
                f"discrepancy {discrepancy:+.3f} pp"
            )


def save_per_image_predictions(records_by_sigma, report_dir):
    fieldnames = [
        "original_index",
        "true_label",
        "true_class",
        "sigma",
        "vit_prediction",
        "hybrid_prediction",
        "vit_correct",
        "hybrid_correct",
    ]

    with (report_dir / "per_image_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for sigma in BLUR_SIGMAS:
            writer.writerows(records_by_sigma[sigma])


def save_paired_outcomes(records_by_sigma, report_dir):
    rows = []

    for sigma, records in records_by_sigma.items():
        counts = {
            "both_correct": 0,
            "only_vit_correct": 0,
            "only_hybrid_correct": 0,
            "both_incorrect": 0,
        }

        for record in records:
            vit_correct = record["vit_correct"]
            hybrid_correct = record["hybrid_correct"]

            if vit_correct and hybrid_correct:
                counts["both_correct"] += 1
            elif vit_correct:
                counts["only_vit_correct"] += 1
            elif hybrid_correct:
                counts["only_hybrid_correct"] += 1
            else:
                counts["both_incorrect"] += 1

        total = sum(counts.values())
        if total != 5000:
            raise RuntimeError(f"Paired outcomes at sigma {sigma} sum to {total}.")

        rows.append({"sigma": sigma, **counts, "total_images": total})
        print(
            f"sigma {sigma:3.1f} paired outcomes | "
            f"both {counts['both_correct']} | "
            f"ViT only {counts['only_vit_correct']} | "
            f"hybrid only {counts['only_hybrid_correct']} | "
            f"neither {counts['both_incorrect']} | total {total}"
        )

    with (report_dir / "paired_outcomes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def calculate_per_class_recall(records_by_sigma, class_names):
    clean_recall = {}
    rows = []

    for sigma, records in records_by_sigma.items():
        for model_name in ("vit", "hybrid"):
            for class_index, class_name in enumerate(class_names):
                class_records = [
                    record
                    for record in records
                    if record["true_label"] == class_index
                ]
                correct_count = sum(
                    record[f"{model_name}_correct"]
                    for record in class_records
                )
                recall = correct_count / len(class_records)

                if sigma == 0:
                    clean_recall[model_name, class_index] = recall

                rows.append(
                    {
                        "model": model_name,
                        "sigma": sigma,
                        "class_index": class_index,
                        "class_name": class_name,
                        "images_in_class": len(class_records),
                        "correct_predictions": correct_count,
                        "recall": recall,
                        "recall_percent": 100 * recall,
                        "recall_drop_percentage_points": 100
                        * (clean_recall[model_name, class_index] - recall),
                    }
                )

    return rows


def save_per_class_recall(rows, report_dir):
    with (report_dir / "per_class_recall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "recall": f"{row['recall']:.6f}",
                    "recall_percent": f"{row['recall_percent']:.2f}",
                    "recall_drop_percentage_points": (
                        f"{row['recall_drop_percentage_points']:.2f}"
                    ),
                }
            )


def save_per_class_recall_figure(rows, class_names, report_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)

    for ax, sigma in zip(axes.flat, BLUR_SIGMAS):
        sigma_rows = [row for row in rows if row["sigma"] == sigma]
        vit_recall = [
            row["recall_percent"]
            for row in sigma_rows
            if row["model"] == "vit"
        ]
        hybrid_recall = [
            row["recall_percent"]
            for row in sigma_rows
            if row["model"] == "hybrid"
        ]
        positions = list(range(len(class_names)))

        ax.bar(
            [position - 0.2 for position in positions],
            vit_recall,
            width=0.4,
            label="ViT",
        )
        ax.bar(
            [position + 0.2 for position in positions],
            hybrid_recall,
            width=0.4,
            label="Hybrid",
        )
        ax.set_title(f"Sigma {sigma:g}")
        ax.set_xticks(positions, class_names, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Recall (%)")
        ax.grid(axis="y", alpha=0.3)

    axes[0, 0].legend()
    fig.suptitle("Per-class recall on the fixed validation split")
    fig.tight_layout()
    fig.savefig(report_dir / "per_class_recall_comparison.png", dpi=200)
    plt.close(fig)


def save_prediction_stability(records_by_sigma, report_dir):
    clean_records = records_by_sigma[0.0]
    rows = []

    for model_name in ("vit", "hybrid"):
        clean_correct_indices = {
            record["original_index"]
            for record in clean_records
            if record[f"{model_name}_correct"]
        }

        for sigma, records in records_by_sigma.items():
            remained_correct = sum(
                record["original_index"] in clean_correct_indices
                and record[f"{model_name}_correct"]
                for record in records
            )
            fraction = remained_correct / len(clean_correct_indices)
            rows.append(
                {
                    "model": model_name,
                    "sigma": sigma,
                    "clean_correct_images": len(clean_correct_indices),
                    "remained_correct_images": remained_correct,
                    "fraction_remaining_correct": fraction,
                    "percent_remaining_correct": 100 * fraction,
                }
            )
            print(
                f"{model_name:6s} stability | sigma {sigma:3.1f} | "
                f"{remained_correct}/{len(clean_correct_indices)} "
                f"({fraction:.2%})"
            )

    with (report_dir / "prediction_stability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "fraction_remaining_correct": (
                        f"{row['fraction_remaining_correct']:.6f}"
                    ),
                    "percent_remaining_correct": (
                        f"{row['percent_remaining_correct']:.2f}"
                    ),
                }
            )


def save_example_grid(records, sigma, category, validation_dataset, class_names, report_dir):
    if category == "only_hybrid_correct":
        selected = [
            record
            for record in records
            if record["hybrid_correct"] and not record["vit_correct"]
        ]
        category_title = "Only hybrid correct"
    else:
        selected = [
            record
            for record in records
            if record["vit_correct"] and not record["hybrid_correct"]
        ]
        category_title = "Only ViT correct"

    selected = sorted(selected, key=lambda record: record["original_index"])[:6]
    row_count = max(1, len(selected))
    fig, axes = plt.subplots(row_count, 2, figsize=(8, 3 * row_count))
    axes = axes.reshape(row_count, 2)

    if not selected:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "No examples found", ha="center")

    for row_index, record in enumerate(selected):
        clean_image, _ = validation_dataset.dataset[record["original_index"]]
        blurred_image = apply_blur(clean_image.unsqueeze(0), sigma).squeeze(0)
        title = (
            f"index {record['original_index']} | true: {record['true_class']}\n"
            f"ViT: {class_names[record['vit_prediction']]} | "
            f"Hybrid: {class_names[record['hybrid_prediction']]}"
        )

        axes[row_index, 0].imshow(clean_image.permute(1, 2, 0))
        axes[row_index, 0].set_title("Clean\n" + title, fontsize=9)
        axes[row_index, 1].imshow(blurred_image.permute(1, 2, 0))
        axes[row_index, 1].set_title(f"Blurred, sigma {sigma:g}\n" + title, fontsize=9)

        for ax in axes[row_index]:
            ax.axis("off")

    fig.suptitle(
        f"{category_title} at sigma {sigma:g}: illustrative, index-sorted examples"
    )
    fig.tight_layout()
    fig.savefig(
        report_dir / f"examples_sigma_{sigma:.1f}_{category}.png",
        dpi=200,
    )
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
    indexed_dataset = IndexedValidationDataset(validation_dataset)
    loader = DataLoader(
        indexed_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    class_names = validation_dataset.dataset.classes
    report_dir = PROJECT_DIR / "reports/blur_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Validation images:", len(indexed_dataset))
    for model_name, _, checkpoint, parameter_count in loaded:
        print(
            f"{model_name}: checkpoint epoch {checkpoint['epoch']}, "
            f"trainable parameters {parameter_count:,}"
        )

    records_by_sigma = {
        sigma: collect_predictions(models, loader, sigma, device, class_names)
        for sigma in BLUR_SIGMAS
    }

    print("\nAggregate accuracy verification:")
    verify_aggregate_accuracies(records_by_sigma)
    save_per_image_predictions(records_by_sigma, report_dir)

    print("\nPaired outcomes:")
    save_paired_outcomes(records_by_sigma, report_dir)

    recall_rows = calculate_per_class_recall(records_by_sigma, class_names)
    save_per_class_recall(recall_rows, report_dir)
    save_per_class_recall_figure(recall_rows, class_names, report_dir)

    print("\nConditional prediction stability:")
    save_prediction_stability(records_by_sigma, report_dir)

    for sigma in (1.0, 1.5):
        for category in ("only_hybrid_correct", "only_vit_correct"):
            save_example_grid(
                records_by_sigma[sigma],
                sigma,
                category,
                validation_dataset,
                class_names,
                report_dir,
            )

    print("\nSaved prediction analysis in:", report_dir)


if __name__ == "__main__":
    main()
