import torch
from torch import nn

from encoder import TransformerEncoderBlock


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()

        self.patch_size = patch_size

        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, images):
        _, _, height, width = images.shape

        assert height % self.patch_size == 0
        assert width % self.patch_size == 0

        patch_list = []

        for top in range(0, height, self.patch_size):
            for left in range(0, width, self.patch_size):
                patch = images[
                    :,
                    :,
                    top : top + self.patch_size,
                    left : left + self.patch_size,
                ]

                patch_list.append(patch.flatten(start_dim=1))

        # patches: [batch, number of patches, flattened patch size]
        patches = torch.stack(patch_list, dim=1)

        return self.projection(patches)


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        depth=2,
        mlp_ratio=4,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.in_channels = in_channels

        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # register position vectors inside the model
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        nn.init.normal_(
            self.position_embeddings,
            mean=0.0,
            std=0.02,
        )

        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        assert images.ndim == 4
        assert images.shape[1] == self.in_channels
        assert images.shape[2:] == (
            self.image_size,
            self.image_size,
        )

        # convert images into tokens and add position information
        tokens = self.patch_embedding(images)
        tokens = tokens + self.position_embeddings

        # process the sequence through the encoder blocks
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # average across tokens to obtain one vector per image
        image_features = tokens.mean(dim=1)

        # produce an unnormalized score for each class
        logits = self.classifier(image_features)

        return logits


if __name__ == "__main__":
    torch.manual_seed(42)

    model = TinyViT()

    # check the model interface before using real training data
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    logits = model(images)

    print("Input shape:", images.shape)
    print("Class scores shape:", logits.shape)

    assert logits.shape == (2, 10)

    # verify the complete path from predictions to gradients
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    loss.backward()

    assert model.patch_embedding.projection.weight.grad is not None
    assert model.position_embeddings.grad is not None
    assert model.classifier.weight.grad is not None

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {num_parameters:,}")
    print(f"Initial demonstration loss: {loss.item():.4f}")
    print("ViT checks passed.")