import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/vit_baseline")
    args = parser.parse_args()

    run_dir = PROJECT_DIR / args.run_dir
    history_path = run_dir / "history.json"

    if not history_path.exists():
        raise FileNotFoundError(
            f"No history found at {history_path}. "
            "Complete at least one training epoch first."
        )

    history = json.loads(history_path.read_text(encoding="utf-8"))

    if not history:
        raise ValueError("Training history is empty.")

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # compare training and validation loss
    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation",
    )
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # display accuracy as a percentage
    axes[1].plot(
        epochs,
        [100 * row["train_accuracy"] for row in history],
        label="Training",
    )
    axes[1].plot(
        epochs,
        [100 * row["val_accuracy"] for row in history],
        label="Validation",
    )
    axes[1].set_title("Classification accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = run_dir / "training_curves.png"
    fig.savefig(output_path, dpi=200)

    best_row = max(history, key=lambda row: row["val_accuracy"])

    print("Completed epochs:", len(history))
    print("Best validation epoch:", best_row["epoch"])
    print(f"Best validation accuracy: {best_row['val_accuracy']:.2%}")
    print("Saved figure:", output_path)

    plt.show()


if __name__ == "__main__":
    main()