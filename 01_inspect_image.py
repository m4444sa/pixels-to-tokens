from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent

# convert images to tensors with values in [0, 1]
to_tensor = transforms.ToTensor()

# download the training dataset if needed
dataset = datasets.CIFAR10(
    root=str(PROJECT_DIR / "data"),
    train=True,
    download=True,
    transform=to_tensor,
)

# get one image and its class label
image, label = dataset[0]
class_name = dataset.classes[label]

print("Number of images:", len(dataset))
print("Class:", class_name)
print("Image shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum value:", image.min().item())
print("Maximum value:", image.max().item())

# inspect the rgb values of the top-left pixel
print("Top-left pixel RGB:", image[:, 0, 0])

# add a batch dimension for a single image
batch = image.unsqueeze(0)
print("Batch shape:", batch.shape)

# rearrange [channels, height, width] for matplotlib
display_image = image.permute(1, 2, 0)

plt.imshow(display_image)
plt.title(f"CIFAR-10: {class_name}")
plt.axis("off")
plt.tight_layout()
plt.show()