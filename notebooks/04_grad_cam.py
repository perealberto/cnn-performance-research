# +
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from src import grad, models

datasets_path: Path = Path.cwd() / "data"
if not datasets_path.exists():
    datasets_path.mkdir()
models_path: Path = Path.cwd() / "models"
if not models_path.exists():
    models_path.mkdir()

device: torch.device = models.get_device()
print(f"Using {device} device")

# +
train_dataset = datasets.MNIST(root=datasets_path, transform=ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root=datasets_path, train=False, transform=ToTensor(), download=True
)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# +
simple_cnn = models.SimpleCNN()
checkpoint = torch.load(models_path / "simple_CNN.ckpt", map_location=device)
simple_cnn.load_state_dict(checkpoint)

simple_cnn = simple_cnn.to(device).eval()
batch, _ = next(iter(test_dataloader))
img = batch[6].unsqueeze(0).to(device)

gc = grad.GradCAM(simple_cnn, device=device)
cam = gc.generate(img)
superimposed = grad.compute_superimposed_image(img, heatmap)

img = img.squeeze(0).cpu().detach().numpy()[0]

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

axs[0].imshow(img, cmap="gray")

im1 = axs[1].imshow(heatmap, cmap="jet")
im2 = axs[2].imshow(superimposed)

cbar1 = fig.colorbar(im1, ax=axs[1], orientation="horizontal", fraction=0.05, pad=0.1)
cbar1.ax.tick_params(labelsize=8)

cbar2 = fig.colorbar(im2, ax=axs[2], orientation="horizontal", fraction=0.05, pad=0.1)
cbar2.ax.tick_params(labelsize=8)

plt.show()
# -
