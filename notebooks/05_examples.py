# +
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch import nn
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

simple_cnn
# -

# ## Process of passing image by conv net

# +
import os, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec

def visualize_mnist_forward(
    model: nn.Module,
    x: torch.Tensor,
    save_path: str | Path = "images/mnist_forward.png",
    cols: int = 8,
    dpi: int = 160,
    cmap: str = "viridis",
    device: torch.device | None = None
):
    """
    Build and save a composite figure with:
      - Input image
      - Feature maps captured after each interesting layer (Conv2d, ReLU, MaxPool2d)
      - Bar chart with softmax probabilities (FC output)

    No cap on the number of feature maps: it shows ALL channels in each captured stage.
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
    captures = []   # list of (tag, tensor[B,C,H,W])
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
    cell_w, cell_h = 2.0, 2.0   # each feature map cell will be roughly this size
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
        ax_title.text(0.01, 0.5, f"{tag}: {C} maps  ({H}×{W})", fontsize=11, va="center", ha="left")

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



# +
batch, _ = next(iter(test_dataloader))
x = batch[6].unsqueeze(0).to(device).requires_grad_(True)

visualize_mnist_forward(
    model=simple_cnn,
    x=x,
    save_path=images_path / "mnist_forward_simplecnn.png",
    cols=8,
    dpi=250
)
# -

