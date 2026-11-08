import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from hybrid import HybridViT
from vit import TinyViT


PROJECT_DIR = Path(__file__).resolve().parent


def build_model(model_name):
    if model_name == "vit":
        return TinyViT()

    if model_name == "hybrid":
        return HybridViT()

    raise ValueError(f"Unknown model: {model_name}")


def save_checkpoint(checkpoint, path):
    # replace the previous checkpoint only after writing finishes
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def make_datasets(train_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    dataset = datasets.CIFAR10(
        root=str(PROJECT_DIR / "data"),
        train=True,
        download=True,
        transform=transform,
    )

    # use a separate generator to keep the data split fixed
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    ).tolist()

    # reserve the final 5000 indices for validation
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[-5000:])

    return train_dataset, val_dataset


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # enable gradients only during training
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected.")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            # weight each batch loss by its number of examples
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += batch_size

            if training and (batch_index + 1) % 50 == 0:
                print(
                    f"  batch {batch_index + 1}/{len(loader)}",
                    flush=True,
                )

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def sanity_check(train_dataset, device, model_name):
    # repeatedly train a disposable model on 16 fixed images
    loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    model = build_model(model_name).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    criterion = nn.CrossEntropyLoss()
    model.train()

    print(
        f"Sanity check: {model_name} on 16 fixed images",
        flush=True,
    )

    for step in range(201):
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected.")

        if step % 20 == 0:
            accuracy = (
                logits.argmax(dim=1) == labels
            ).float().mean().item()

            print(
                f"Step {step:3d} | "
                f"loss {loss.item():.4f} | "
                f"accuracy {accuracy:.1%}",
                flush=True,
            )

        if step < 200:
            loss.backward()
            optimizer.step()

    print(
        "\nSanity check finished. No checkpoints were written.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["vit", "hybrid"],
        default="vit",
    )
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-dir", default=None)

    args = parser.parse_args()

    if not 16 <= args.train_size <= 45000:
        parser.error("--train-size must be between 16 and 45000")

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)
    print("Model:", args.model, flush=True)

    train_dataset, val_dataset = make_datasets(args.train_size)

    if args.sanity:
        sanity_check(train_dataset, device, args.model)
        return

    # keep the two model runs in separate folders by default
    default_run_dir = f"runs/{args.model}_baseline"
    run_dir = PROJECT_DIR / (args.run_dir or default_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    config = {
        "model": (
            "tiny_vit_v1"
            if args.model == "vit"
            else "hybrid_vit_v1"
        ),
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "split_seed": 42,
        "normalization": "mean_0.5_std_0.5",
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {num_parameters:,}",
        flush=True,
    )
    print("Run directory:", run_dir, flush=True)

    start_epoch = 0
    best_accuracy = -1.0
    history = []

    if last_path.exists():
        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=True,
        )

        # prevent mixing incompatible settings in the same run
        if checkpoint["config"] != config:
            raise ValueError(
                "Run settings changed. Use a different --run-dir "
                "for a new experiment."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        history = checkpoint["history"]

        # restore random states after constructing the model
        torch.set_rng_state(checkpoint["rng_state"].cpu())

        if (
            device.type == "cuda"
            and checkpoint["cuda_rng"] is not None
        ):
            torch.cuda.set_rng_state_all(
                [
                    state.cpu()
                    for state in checkpoint["cuda_rng"]
                ]
            )

        print(
            f"Resuming after epoch {start_epoch}. "
            f"Best validation accuracy: {best_accuracy:.2%}",
            flush=True,
        )

    print(
        f"Training images: {len(train_dataset)} | "
        f"Validation images: {len(val_dataset)}",
        flush=True,
    )

    if start_epoch >= args.epochs:
        print(
            f"The run already contains {start_epoch} completed epochs. "
            "Increase --epochs to continue training.",
            flush=True,
        )
        return

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        duration = time.perf_counter() - started
        improved = val_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, val_accuracy)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "seconds": duration,
            }
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "config": config,
            "history": history,
            "rng_state": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None
            ),
        }

        if improved:
            save_checkpoint(checkpoint, best_path)

        save_checkpoint(checkpoint, last_path)

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f}, "
            f"acc {train_accuracy:.2%} | "
            f"val loss {val_loss:.4f}, "
            f"acc {val_accuracy:.2%} | "
            f"{duration / 60:.1f} min | saved",
            flush=True,
        )

    print(
        f"\nCompleted epochs: {args.epochs}. "
        f"Best validation accuracy: {best_accuracy:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()