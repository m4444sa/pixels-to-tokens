import torch
from torch import nn

from vit import TinyViT


class CNNTokenizer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                32,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(self, images):
        # features: [batch, embedding dimension, height, width]
        features = self.features(images)

        # combine spatial positions into one axis
        tokens = features.flatten(start_dim=2)

        # tokens: [batch, spatial positions, embedding dimension]
        tokens = tokens.transpose(1, 2)

        return tokens


class HybridViT(TinyViT):
    def __init__(self):
        super().__init__(
            image_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=64,
            num_heads=4,
            depth=2,
            mlp_ratio=4,
        )

        # replace the patch projection with a convolutional tokenizer
        self.patch_embedding = CNNTokenizer(
            in_channels=3,
            embed_dim=64,
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    images = torch.randn(2, 3, 32, 32)

    tokenizer = CNNTokenizer()
    tokens = tokenizer(images)

    print("Input images:", images.shape)
    print("CNN tokens:", tokens.shape)

    assert tokens.shape == (2, 64, 64)

    model = HybridViT()
    logits = model(images)

    print("Class scores:", logits.shape)
    assert logits.shape == (2, 10)

    # verify that gradients reach the convolutional layers
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    first_conv = model.patch_embedding.features[0]
    assert first_conv.weight.grad is not None

    for name, candidate in [
        ("ViT", TinyViT()),
        ("Hybrid", HybridViT()),
    ]:
        count = sum(
            parameter.numel()
            for parameter in candidate.parameters()
            if parameter.requires_grad
        )
        print(f"{name} trainable parameters: {count:,}")

    print("Hybrid checks passed.")