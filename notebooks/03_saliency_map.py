# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from src import metrics, models

datasets_path: Path = Path.cwd() / "data"
if not datasets_path.exists():
    datasets_path.mkdir()
models_path: Path = Path.cwd() / "models"
if not models_path.exists():
    models_path.mkdir()

device: torch.device = models.get_device()
print(f"Using {device} device")

# %%
train_dataset = datasets.MNIST(root=datasets_path, transform=ToTensor())
test_dataset = datasets.MNIST(root=datasets_path, train=False, transform=ToTensor())

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# %% [markdown]
# ## Neural Network


# %%
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


simple_NN = SimpleNN()
trainer = models.Trainer(
    model=simple_NN,
    optimizer=optim.Adam(simple_NN.parameters(), lr=1e-3),
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    device=device,
    save_name="simple_NN",
)
history = trainer.fit()


# %% [markdown]
# ## Convolutional NN


# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


simple_CNN = SimpleCNN()
trainer = models.Trainer(
    model=simple_CNN,
    optimizer=optim.Adam(simple_CNN.parameters(), lr=1e-3),
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    device=device,
    save_name="simple_CNN",
)
history = trainer.fit()


# %% [markdown]
# ## Saliency Map


# %%
def compute_saliency(x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> np.ndarray:
    """
    Compute saliency maps for a batch of images.
    """
    assert x.ndim == 4, "Input tensor must be 4-dimensional (B, C, H, W)"
    assert x.shape[0] == y.shape[0], (
        "Input and target tensors must have the same batch size"
    )
    device = next(model.parameters()).device
    model.eval()

    x = x.to(device).clone().detach().requires_grad_(True)
    y = y.to(device)

    # Forward pass and select target classes
    logits = model(x)
    target = logits.argmax(dim=1) if y is None else y
    score = logits[torch.arange(logits.shape[0]), target].sum()

    # Backprop to get gradients
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    score.backward()
    grad = x.grad.detach().abs()  # type: ignore

    # Normalize and convert to numpy
    saliency = grad.max(dim=1).values.cpu().numpy()
    saliency = (saliency - saliency.min(axis=(1, 2), keepdims=True)) / (
        saliency.max(axis=(1, 2), keepdims=True)
        - saliency.min(axis=(1, 2), keepdims=True)
        + 1e-12
    )

    return saliency


def show_image_and_saliency(
    img: torch.Tensor,
    saliency: np.ndarray,
    title: str = "",
    savepath: str | Path | None = None,
):
    """
    Displays the image and saliency side by side and overlayed.
    """
    img = img.clone().cpu()
    if img.shape[0] == 1:
        img_vis = img[0].cpu().numpy()
        cmap_img = "gray"
    else:
        img_vis = img.permute(1, 2, 0).cpu().numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-12)
        cmap_img = None

    sal = saliency.squeeze(0) if saliency.ndim == 3 else saliency

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_vis, cmap=cmap_img)
    ax[0].axis("off")
    ax[0].set_title("Input Image")

    im1 = ax[1].imshow(sal, cmap="jet")
    ax[1].axis("off")
    ax[1].set_title("Saliency Map")
    cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Saliency Intensity")

    ax[2].imshow(img_vis, cmap=cmap_img)
    ax[2].imshow(sal, cmap="jet", alpha=0.5)
    ax[2].axis("off")
    ax[2].set_title("Overlay")

    if title:
        fig.suptitle(title)
    if savepath:
        plt.savefig(savepath)
    plt.show()


# %%
device = models.get_device()
simple_nn = simple_NN.to(device).eval()

idx = 2
img, label = test_dataset[idx]
x = img.unsqueeze(0)
y = torch.tensor([label])

sal = compute_saliency(x, y, simple_NN)
show_image_and_saliency(img, sal, title=f"NN - Saliency (label={label})")

# %%
device = models.get_device()
simple_cnn = simple_CNN.to(device).eval()

idx = 2
img, label = test_dataset[idx]
x = img.unsqueeze(0)
y = torch.tensor([label])

sal = compute_saliency(x, y, simple_CNN)
show_image_and_saliency(img, sal, title=f"CNN - Saliency (label={label})")

# %% [markdown]
# ## Average Saliency Map

# %%

reps = Counter([x[1] for x in test_dataset])
labels, counts = reps.keys(), reps.values()

fig, ax = plt.subplots()
ax.set_title(f"Label counter - Total = {len(test_dataset):,} imgs")
bars = ax.bar(labels, counts)  # type: ignore
ax.set_xticks(range(10))
ax.bar_label(bars)
plt.show()


# %%
def compute_avg_saliency_by_class(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> dict[int, np.ndarray]:
    """
    Compute average saliency maps for each class in the dataset.
    """
    saliencies_by_class = defaultdict(list)

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        saliency_batch = compute_saliency(X, y, model)
        for i in range(X.shape[0]):
            label = y[i].item()
            saliencies_by_class[label].append(saliency_batch[i])

    avg_saliency_by_class = {
        label: np.mean(saliencies_by_class[label], axis=0)
        for label in sorted(saliencies_by_class.keys())
    }
    return avg_saliency_by_class


def show_avg_saliency_by_class(
    avg_saliency_by_class: dict[int, np.ndarray],
    class_names: dict[int, str] | None = None,
    title: str = "",
    savepath: str | Path | None = None,
):
    """
    Display average saliency maps for each class.
    """
    import matplotlib.pyplot as plt

    n_classes = len(avg_saliency_by_class)
    n_cols = min(5, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, (label, saliency) in enumerate(avg_saliency_by_class.items()):
        ax = axes[i]
        ax.imshow(saliency, cmap="jet")
        ax.axis("off")
        ax_title = f"Class: {label}"
        if class_names and label in class_names:
            ax_title += f": {class_names[label]}"
        ax.set_title(ax_title)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title)
    if savepath:
        plt.savefig(savepath)
    plt.show()


# %%
avg_saliency_map = compute_avg_saliency_by_class(test_dataloader, simple_NN, device)
show_avg_saliency_by_class(
    avg_saliency_map, None, "AVG Saliency Map by Class - Simple NN"
)

# %%
avg_saliency_map = compute_avg_saliency_by_class(test_dataloader, simple_CNN, device)
show_avg_saliency_by_class(
    avg_saliency_map, None, "AVG Saliency Map by Class - Simple CNN"
)

# %% [markdown]
# ## Top 3 most confusing

# %%

y_true_NN, y_pred_NN, *_ = metrics._collect_outputs(test_dataloader, simple_NN, device)
metrics.print_top_k_confusions("Simple NN top 3 confusions", y_true_NN, y_pred_NN)

# %%
y_true_CNN, y_pred_CNN, *_ = metrics._collect_outputs(
    test_dataloader, simple_CNN, device
)
metrics.print_top_k_confusions("Simple CNN top 3 confusions", y_true_CNN, y_pred_CNN)
