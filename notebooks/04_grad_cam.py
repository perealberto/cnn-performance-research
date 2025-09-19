# +
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
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


# -


class GradCAM:
    def __init__(self, model: nn.Module, device: torch.device | None = None):
        self.model = model
        self.target_layer = self._get_last_conv_layer(model)
        self.gradients = None
        self.activations = None
        self.device = device if device else next(model.parameters()).device

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._full_backward_hook)

    def _get_last_conv_layer(self, model: nn.Module) -> nn.Module:
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No convolutional layer found in the model.")

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _full_backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_heatmap(self, x: torch.Tensor) -> tuple[np.ndarray, int, float]:
        """Compute Grad-CAM heatmap for a single input image tensor."""
        self.model.eval()
        x = x.to(self.device).requires_grad(True)  # type: ignore

        logits = self.model(x)
        self.model.zero_grad()
        class_idx = logits.argmax(dim=1).item()

        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        assert self.gradients is not None, "Gradients have not been computed."
        assert self.activations is not None, "Activations have not been recorded."

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)  # ReLU removes negative values
        heatmap /= torch.max(heatmap)  # Normalize to [0, 1]

        probs = torch.softmax(logits, dim=1)
        predicted_prob = probs[0, class_idx].item()

        return heatmap.squeeze().cpu().numpy(), class_idx, predicted_prob


# +
simple_cnn = models.SimpleCNN()
checkpoint = torch.load(models_path / "simple_CNN.ckpt", map_location=device)
simple_cnn.load_state_dict(checkpoint)

simple_cnn = simple_cnn.to(device).eval()
batch, _ = next(iter(test_dataloader))
img = batch[6].unsqueeze(0).to(device)

gc = grad.GradCAM(simple_cnn, device=device)
heatmap, cls, _ = gc.compute_heatmap(img)
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
