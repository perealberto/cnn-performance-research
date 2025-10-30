# +
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
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
images_path: Path = Path.cwd() / "images"
if not images_path.exists():
    images_path.mkdir()

device: torch.device = models.get_device()
print(f"Using {device} device")

# load data
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
# -

# ## Models

logisticReg = models.LogisticRegressionMNIST()
if not (models_path / "logistic_regression.ckpt").exists():
    trainer = models.Trainer(
        model=logisticReg,
        optimizer=optim.Adam(logisticReg.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="logistic_regression",
    )
    history = trainer.fit()
else:
    checkpoint = torch.load(
        models_path / "logistic_regression.ckpt", map_location=device
    )
    logisticReg.load_state_dict(checkpoint)
    logisticReg = logisticReg.to(device).eval()
logisticReg
n_params = sum(p.numel() for p in logisticReg.parameters())
print(f"Total parameters count = {n_params}")

mlp512 = models.MLP512x512()
if not (models_path / "mlp512.ckpt").exists():
    trainer = models.Trainer(
        model=mlp512,
        optimizer=optim.Adam(mlp512.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="mlp512",
    )
    history = trainer.fit()
else:
    checkpoint = torch.load(models_path / "mlp512.ckpt", map_location=device)
    mlp512.load_state_dict(checkpoint)
    mlp512 = mlp512.to(device).eval()
mlp512
n_params = sum(p.numel() for p in mlp512.parameters())
print(f"Total parameters count = {n_params}")

tinyCNN = models.TinyCNNBad()
if not (models_path / "tinyCNN.ckpt").exists():
    with torch.enable_grad():
        trainer = models.Trainer(
            model=tinyCNN,
            optimizer=optim.Adam(tinyCNN.parameters(), lr=1e-3),
            loss_fn=nn.CrossEntropyLoss(),
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            device=device,
            save_name="tinyCNN",
        )
        history = trainer.fit()
else:
    checkpoint = torch.load(models_path / "tinyCNN.ckpt", map_location=device)
    tinyCNN.load_state_dict(checkpoint)
    tinyCNN = tinyCNN.to(device).eval()
tinyCNN
n_params = sum(p.numel() for p in tinyCNN.parameters())
print(f"Total parameters count = {n_params}")

solidCNN = models.SmallCNNSolid()
if not (models_path / "solidCNN.ckpt").exists():
    trainer = models.Trainer(
        model=solidCNN,
        optimizer=optim.Adam(solidCNN.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="solidCNN",
    )
    history = trainer.fit()
else:
    checkpoint = torch.load(models_path / "solidCNN.ckpt", map_location=device)
    solidCNN.load_state_dict(checkpoint)
    solidCNN = solidCNN.to(device).eval()
solidCNN
n_params = sum(p.numel() for p in solidCNN.parameters())
print(f"Total parameters count = {n_params}")

# ## MNIST

# +
images_saved = set()
while len(images_saved) < 10:
    idx = int(torch.randint(len(train_dataset), size=(1,)).item())
    img, label = train_dataset[idx]
    if label in images_saved:
        continue
    plt.imsave(
        images_path / "samples_mnist" / f"mnist_{label}_sample.png",
        img.squeeze(),
        cmap="gray",
    )
    images_saved.add(label)

print(f"The following labels were stored: {images_saved}")
# -

figure = plt.figure(figsize=(8, 4))
cols, rows = 5, 2
for i in range(1, cols * rows + 1):
    sample_idx = int(torch.randint(len(train_dataset), size=(1,)).item())
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.savefig(images_path / "MNIST_examples.png")
plt.show()

# +
import matplotlib.pyplot as plt
from collections import Counter


def count_labels(dataset):
    labels = [y for _, y in dataset]
    counts = Counter(labels)
    return [counts.get(i, 0) for i in range(10)]


train_counts = count_labels(train_subset)
val_counts = count_labels(val_subset)
test_counts = count_labels(test_dataset)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

axes[0].bar(range(10), train_counts, color=colors[0])
axes[0].set_title("Entrenamiento")
axes[0].set_xlabel("Dígito")
axes[0].set_ylabel("Apariciones")
axes[0].set_xticks(range(10))
for i, v in enumerate(train_counts):
    axes[0].text(i, v + 50, str(v), ha="center", fontsize=8)

axes[1].bar(range(10), val_counts, color=colors[1])
axes[1].set_title("Validación")
axes[1].set_xlabel("Dígito")
axes[1].set_xticks(range(10))
for i, v in enumerate(val_counts):
    axes[1].text(i, v + 50, str(v), ha="center", fontsize=8)

axes[2].bar(range(10), test_counts, color=colors[2])
axes[2].set_title("Pruebas")
axes[2].set_xlabel("Dígito")
axes[2].set_xticks(range(10))
for i, v in enumerate(test_counts):
    axes[2].text(i, v + 50, str(v), ha="center", fontsize=8)

# plt.suptitle("Distribution of digits in the MNIST subsets")
plt.tight_layout()
plt.savefig(images_path / "mnist_distribution_datasets.png")
plt.show()

# -

# ## Through traditional models


# ## Through convolutional models

# +
batch, labels = next(iter(train_dataloader))
img, cls = batch[0], labels[0]
figs_path = images_path / "through_convnet"

# original image
plt.imshow(img.squeeze(), cmap="gray")
plt.axis("off")
plt.savefig(figs_path / "original_mnist_number.png")
plt.show()

# first convnet
out1 = solidCNN.conv1(img.to(device))
sample = out1[0].cpu().detach().numpy()
plt.imshow(sample.squeeze())
plt.axis("off")
plt.show()

# +
# kernels conv1
idx = 2
sample = out1[idx].cpu().detach().numpy()
weights = solidCNN.conv1.weight.cpu().detach().numpy()
kernel = weights[idx].squeeze()

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

ax0.set_title(f"Entrada {img.shape[1]}x{img.shape[2]}")
ax0.imshow(img.squeeze(), cmap="gray")
ax0.axis("off")

ax1.set_title(f"K{idx}@{kernel.shape[0]}x{kernel.shape[1]}")
ax1.imshow(kernel, cmap="jet")
for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        text = ax1.text(
            j,
            i,
            round(kernel[i, j], 3),
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
ax1.axis("off")

ax2.set_title(f"FM{idx}@{sample.shape[0]}x{sample.shape[1]}")
ax2.imshow(sample.squeeze())
ax2.axis("off")
plt.savefig(figs_path / "mapa_caracterisicas_con_kernel.png")
plt.show()
# -

# ## Metrics

# ### Saliency Map

# +
idx = 2
batch, labels = next(iter(train_dataloader))
img, cls = batch[idx], labels[idx]
figs_path = images_path / "saliency_map"

x = img.unsqueeze(0)
y = torch.tensor([cls])

saliency = grad.compute_saliency(x, y, solidCNN)
grad.show_image_and_saliency(img, saliency, savepath=figs_path)
# -

# ### Grad CAM


# ## Scripts

# +
import os
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec


def visualize_mnist_forward(
    model: nn.Module,
    x: torch.Tensor,
    save_path: str | Path = "images/mnist_forward.png",
    cols: int = 8,
    dpi: int = 160,
    cmap: str = "viridis",
    device: torch.device | None = None,
):
    """
    Build and save a composite figure with:
      - Input image
      - Feature maps captured after each interesting layer (Conv2d, ReLU, MaxPool2d)
      - Bar chart with softmax probabilities (FC output)

    Returns the captures (layer tag, collection of feats) in tuple format.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep original training/eval state
    was_training = model.training
    model.eval()

    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Ensure input shape [1, C, H, W]
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clone().detach().to(device)
    x.requires_grad_(True)  # needed if the model registers hooks on activations

    # --- Register forward hooks to capture feature maps from Conv2d/ReLU/MaxPool2d ---
    interested = (nn.Conv2d, nn.ReLU, nn.MaxPool2d)
    captures = []  # list of (tag, tensor[B,C,H,W])
    handles = []

    # Counters to build readable tags like conv1/relu1/pool1...
    counts = {nn.Conv2d: 0, nn.ReLU: 0, nn.MaxPool2d: 0}
    type2short = {nn.Conv2d: "conv", nn.ReLU: "relu", nn.MaxPool2d: "pool"}

    def make_hook(m):
        t = type(m)
        counts[t] += 1
        tag = f"{type2short[t]}{counts[t]}"

        def _hook(_m, _inp, out):
            # Only store 4D tensors as feature maps
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                captures.append((tag, out.detach().cpu()))

        return _hook, tag

    # Register hooks in module definition order
    for m in model.modules():
        if isinstance(m, interested):
            h, _ = make_hook(m)
            handles.append(m.register_forward_hook(h))

    # --- Forward pass WITH grad enabled (to let the model register its own hooks safely) ---
    with torch.enable_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        logits = logits.detach().cpu()

    # Remove hooks
    for h in handles:
        h.remove()

    # --- Build figure layout ---
    # Rows: +1 (input) + len(captures) (one block per stage) +1 (bar chart)
    n_stages = len(captures)
    total_rows = 1 + n_stages + 1

    # Figure width grows with columns; height grows with number of rows
    cell_w, cell_h = 2.0, 2.0  # each feature map cell will be roughly this size
    fig_w = cols * cell_w
    fig_h = total_rows * cell_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(total_rows, 1, figure=fig, hspace=0.35)

    row_idx = 0

    # --- (1) Input image ---
    ax_in = fig.add_subplot(gs[row_idx, 0])
    row_idx += 1
    x_show = x[0].detach().cpu()
    if x_show.shape[0] == 1:
        ax_in.imshow(x_show[0], cmap="gray", interpolation="nearest")
    else:
        # Normalize to [0,1] and convert to HxWxC
        img = x_show.numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax_in.imshow(np.transpose(img, (1, 2, 0)), interpolation="nearest")
    ax_in.set_title("Input")
    ax_in.axis("off")

    # --- (2) Feature map stages (conv/relu/pool) ---
    for tag, feat in captures:
        B, C, H, W = feat.shape
        # Show ALL channels (no cap)
        n = C
        _cols = max(1, min(cols, n))
        _rows = math.ceil(n / _cols)

        # Per-stage block: a small title row + a grid of maps
        outer = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[row_idx, 0], height_ratios=[0.3, 4], hspace=0.1
        )
        ax_title = fig.add_subplot(outer[0, 0])
        ax_title.axis("off")
        ax_title.text(
            0.01,
            0.5,
            f"{tag}: {C} maps  ({H}×{W})",
            fontsize=11,
            va="center",
            ha="left",
        )

        inner = gridspec.GridSpecFromSubplotSpec(
            _rows, _cols, subplot_spec=outer[1, 0], wspace=0.05, hspace=0.05
        )
        for i in range(_rows * _cols):
            ax = fig.add_subplot(inner[i // _cols, i % _cols])
            ax.axis("off")
            if i < n:
                m = feat[0, i].numpy()
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                ax.imshow(m, cmap=cmap, interpolation="nearest")

        row_idx += 1

    # --- (3) FC output as a probability bar chart ---
    ax_fc = fig.add_subplot(gs[row_idx, 0])
    row_idx += 1
    classes = np.arange(logits.shape[1])
    pred = int(np.argmax(probs))
    ax_fc.bar(classes, probs, align="center")
    ax_fc.set_xticks(classes)
    ax_fc.set_xlabel("Class")
    ax_fc.set_ylabel("Probability (softmax)")
    ax_fc.set_title(f"FC output (pred = {pred}, p = {probs[pred]:.3f})")
    # Highlight the predicted class
    ax_fc.bar(pred, probs[pred], color="tab:orange")

    fig.suptitle("Forward pass: Input → (Conv/ReLU/Pool)* → FC", y=0.995, fontsize=13)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # Restore original train/eval state
    if was_training:
        model.train()

    print(f"Saved figure at: {save_path}")

    return captures


# +
batch, _ = next(iter(test_dataloader))
x = batch[6].unsqueeze(0).to(device).requires_grad_(True)

captures = visualize_mnist_forward(
    model=solidCNN, x=x, save_path=images_path / "solidCNN_forward.png", cols=8, dpi=250
)
# -
tag, feats = captures[0]
feats = feats.squeeze()

# ## Mathematical representations

# ### Gradient Descent

# +
f_error = lambda x: 1 / 2 * (x) ** 2
df_error = lambda x: x
data = np.linspace(-1, 1, num=1000)
rnd_x = np.random.choice(data, 1)

fig, ax = plt.subplots()
ax.plot(data, f_error(data), label="f_error", color="black")
ax.scatter(rnd_x, f_error(rnd_x), color="red", label="inicio")

change_rate = 0.01
for _ in range(250):
    slope = df_error(rnd_x)
    rnd_x = rnd_x - slope * change_rate
    step = ax.scatter(rnd_x, f_error(rnd_x), marker=".", color="blue")

ax.scatter(rnd_x, f_error(rnd_x), color="green", label="final")

ax.grid(True)
h, l = ax.get_legend_handles_labels()
h.append(step)
l.append("paso")
ax.legend(h, l)
plt.savefig(images_path / "backpropagation" / "gradient_descent_2d.png")
plt.show()

# +
f_error = lambda x, y: 1 / 2 * ((x - 1) ** 2 + y**2)
dfdx_error = lambda x: x - 1
dfdy_error = lambda y: y

data = np.linspace(-10, 10, 1000)
x, y = np.meshgrid(data, data)
rnd_x = np.random.choice(data, 1)
rnd_y = np.random.choice(data, 1)

fig, ax = plt.subplots()
contour = ax.contourf(x, y, f_error(x, y), levels=10)
ax.scatter(rnd_x, rnd_y, color="red", label="inicio")

change_rate = 0.01
for _ in range(250):
    slope_x = dfdx_error(rnd_x)
    slope_y = dfdy_error(rnd_y)
    rnd_x = rnd_x - slope_x * change_rate
    rnd_y = rnd_y - slope_y * change_rate
    step = ax.scatter(rnd_x, rnd_y, color="blue", marker=".")

ax.scatter(rnd_x, rnd_y, color="green", label="final")

ax.grid(True)
h, l = ax.get_legend_handles_labels()
h.append(step), l.append("paso")
ax.legend(h, l)
plt.colorbar(contour, label="f_error")
plt.savefig(images_path / "backpropagation" / "gradient_descent_3d.png")
plt.show()
# -

# ### Convolutions

# +
import numpy as np
import matplotlib.pyplot as plt

f = np.array([0, 0, 1, 1, 0, 0], dtype=float)
g = np.array([-1, 0, 1], dtype=float)
conv = np.convolve(f, g, mode="full")


fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2)


ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[1, :])


ax1.set_title("Señal f(n)")
ax1.stem(range(len(f)), f, basefmt=" ")
ax1.grid(True)


ax2.set_title("Señal g(n)")
ax2.stem(range(len(g)), g, basefmt=" ", linefmt="r-", markerfmt="ro")
ax2.grid(True)


ax3.set_title("Convolución f*g")
ax3.stem(
    range(len(conv)), conv, basefmt=" ", linefmt="C1-", markerfmt="C1o", label="f * g"
)
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig(images_path / "convolucion_f_g_discretas.png")
plt.show()

# +
batch, labels = next(iter(train_dataloader))
img, cls = batch[0], labels[0]
figs_path = images_path / "sobel_filter"

# original image
plt.imshow(img.squeeze(), cmap="gray")
plt.axis("off")
plt.savefig(figs_path / "original_image.png")
plt.show()

# sobel x filter
sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
wh_t = torch.from_numpy(sobel_x).unsqueeze(0).unsqueeze(0)
wv_t = torch.from_numpy(sobel_y).unsqueeze(0).unsqueeze(0)
before_sobel = img.unsqueeze(0)

after_sobel_x = F.relu(F.conv2d(before_sobel, wh_t))
plt.imshow(after_sobel_x.squeeze())
plt.axis("off")
plt.savefig(figs_path / "after_sobel_x.png")
plt.show()
after_sobel_y = F.relu(F.conv2d(before_sobel, wv_t))
plt.imshow(after_sobel_y.squeeze())
plt.axis("off")
plt.savefig(figs_path / "after_sobel_y.png")
plt.show()
after_sobel_xy = torch.sqrt(after_sobel_x.pow(2) + after_sobel_y.pow(2) + 1e-8)
plt.imshow(after_sobel_xy.squeeze())
plt.axis("off")
plt.savefig(figs_path / "after_sobel_xy.png")
plt.show()
# -

pool = F.max_pool2d(after_sobel_x, kernel_size=2)
plt.imshow(pool.squeeze())
plt.axis("off")
plt.savefig(figs_path / "pool_filter.png")
plt.show()
print(after_sobel_x.shape)
print(pool.shape)
