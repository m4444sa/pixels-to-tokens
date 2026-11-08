from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent

PATCH_SIZE = 4
EMBED_DIM = 64

# make initialization repeatable for this demonstration
torch.manual_seed(42)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()

        self.patch_size = patch_size

        # each flattened patch contains channels × height × width values
        patch_dim = in_channels * patch_size * patch_size

        # use the same learned projection for every patch
        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, images):
        # images: [batch, channels, height, width]
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

                # flatten each patch separately for each image
                patch = patch.flatten(start_dim=1)
                patch_list.append(patch)

        # patches: [batch, number of patches, flattened patch size]
        patches = torch.stack(patch_list, dim=1)

        # embeddings: [batch, number of patches, embedding dimension]
        embeddings = self.projection(patches)

        return embeddings


dataset = datasets.CIFAR10(
    root=str(PROJECT_DIR / "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

image, label = dataset[0]
images = image.unsqueeze(0)

patch_embedding = PatchEmbedding(
    patch_size=PATCH_SIZE,
    in_channels=3,
    embed_dim=EMBED_DIM,
)

embeddings = patch_embedding(images)

num_patches = (image.shape[1] // PATCH_SIZE) * (
    image.shape[2] // PATCH_SIZE
)

# create one trainable position vector for each patch location
position_embeddings = nn.Parameter(
    torch.zeros(1, num_patches, EMBED_DIM)
)

# start with small random values
nn.init.normal_(position_embeddings, mean=0.0, std=0.02)

# add position information to the patch content
tokens = embeddings + position_embeddings

print("Input images:", images.shape)
print("Patch embeddings:", embeddings.shape)
print("Position embeddings:", position_embeddings.shape)
print("Tokens with position:", tokens.shape)

print("\nFirst patch embedding, first five values:")
print(embeddings[0, 0, :5].detach())

print("\nFirst position vector, first five values:")
print(position_embeddings[0, 0, :5].detach())

print("\nTheir sum, first five values:")
print(tokens[0, 0, :5].detach())

assert tokens.shape == (1, num_patches, EMBED_DIM)

# check that position information was added element by element
assert torch.allclose(
    tokens - embeddings,
    position_embeddings,
)