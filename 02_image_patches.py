from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent
PATCH_SIZE = 4
# patch size is the dimension of pixels of each patch
dataset = datasets.CIFAR10(
    root=str(PROJECT_DIR / "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

image, label = dataset[0]
class_name = dataset.classes[label]

channels, height, width = image.shape

assert height % PATCH_SIZE == 0
assert width % PATCH_SIZE == 0

# collect patches from left to right, then top to bottom
patch_list = []

for top in range(0, height, PATCH_SIZE):
    for left in range(0, width, PATCH_SIZE):
        patch = image[
            :,
            top : top + PATCH_SIZE,
            left : left + PATCH_SIZE,
        ]
        patch_list.append(patch)

# add an axis identifying each patch
patches = torch.stack(patch_list, dim=0)

# flatten each patch while keeping patches separate
flat_patches = patches.flatten(start_dim=1)

print("Image shape:", image.shape)
print("Patch tensor shape:", patches.shape)
print("Flattened patch shape:", flat_patches.shape)

# verify that the first patch is the top-left image region
assert torch.equal(
    patches[0],
    image[:, :PATCH_SIZE, :PATCH_SIZE],
)

# verify that flattening preserves the first patch's values
assert torch.equal(
    flat_patches[0].reshape(channels, PATCH_SIZE, PATCH_SIZE),
    patches[0],
)

# display the original image with patch boundaries
fig, ax = plt.subplots(figsize=(5, 5))

ax.imshow(image.permute(1, 2, 0), interpolation="nearest")

for boundary in range(PATCH_SIZE, height, PATCH_SIZE):
    ax.axhline(boundary - 0.5, color="white", linewidth=0.7)

for boundary in range(PATCH_SIZE, width, PATCH_SIZE):
    ax.axvline(boundary - 0.5, color="white", linewidth=0.7)

ax.set_title(f"{class_name}: 64 patches of size 4 × 4")
ax.axis("off")
fig.tight_layout()

# display the patches separately in their original arrangement
grid_rows = height // PATCH_SIZE
grid_cols = width // PATCH_SIZE

fig, axes = plt.subplots(
    grid_rows,
    grid_cols,
    figsize=(8, 8),
)

for index, ax in enumerate(axes.flat):
    ax.imshow(
        patches[index].permute(1, 2, 0),
        interpolation="nearest",
    )
    ax.set_title(str(index), fontsize=8)
    ax.axis("off")


fig.tight_layout()
plt.show()