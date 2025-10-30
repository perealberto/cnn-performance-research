from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps as cm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def compute_saliency(x: torch.Tensor, y: torch.Tensor, model: nn.Module) -> np.ndarray:
    """
    Computes the saliency map for a given input tensor and model.
    The saliency map highlights the regions of the input that have the greatest influence
    on the model's output. This is achieved by computing the gradient of the model's output
    with respect to the input tensor.
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size,
            C is the number of channels, H is the height, and W is the width.
        y (torch.Tensor): Target tensor of shape (B,) containing the target class indices
            for each input in the batch. If `None`, the predicted class indices are used.
        model (nn.Module): The neural network model for which the saliency map is computed.
    Returns:
        np.ndarray: A numpy array of shape (B, H, W) containing the normalized saliency
        maps for each input in the batch. The values are scaled to the range [0, 1].
    Raises:
        AssertionError: If the input tensor `x` is not 4-dimensional or if the batch sizes
            of `x` and `y` do not match.
    Notes:
        - The input tensor `x` is cloned and detached before computing gradients to ensure
          that the original tensor is not modified.
        - The gradients are normalized to the range [0, 1] for visualization purposes.
        - A small epsilon value (1e-12) is added to the denominator during normalization
          to avoid division by zero.
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
    Displays an input image, its corresponding saliency map, and an overlay of the two.
    Args:
        img (torch.Tensor): The input image tensor. Should be in CHW format (channels, height, width).
                           If the image has a single channel, it will be displayed in grayscale.
        saliency (np.ndarray): The saliency map as a NumPy array. Should have the same spatial dimensions
                               as the input image. If it has 3 dimensions, the first dimension is squeezed.
        title (str, optional): The title for the entire figure. Defaults to an empty string.
        savepath (str | Path | None, optional): The file path to save the figure. If None, the figure is not saved.
                                                Defaults to None.
    Returns:
        None: This function does not return anything. It displays the plots and optionally saves the figure.
    Notes:
        - The function creates three subplots:
            1. The input image.
            2. The saliency map with a color bar indicating intensity.
            3. An overlay of the input image and the saliency map.
        - The saliency map is visualized using the "jet" colormap.
        - If `savepath` is provided, the figure is saved to the specified path.
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


def compute_avg_saliency_by_class(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> dict[int, np.ndarray]:
    """
    Computes the average saliency map for each class in the dataset.
    Args:
        dataloader (DataLoader): A PyTorch DataLoader providing batches of input data and labels.
        model (nn.Module): The neural network model used to compute saliency maps.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.
    Returns:
        dict[int, np.ndarray]: A dictionary where keys are class labels (int) and values are
        the average saliency maps (numpy arrays) for each class.
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
    Visualizes the average saliency maps for each class in a grid layout.
    Args:
        avg_saliency_by_class (dict[int, np.ndarray]):
            A dictionary where keys are class labels (integers) and values are
            the corresponding average saliency maps as NumPy arrays.
        class_names (dict[int, str] | None, optional):
            A dictionary mapping class labels to their corresponding names.
            If provided, class names will be displayed in the titles. Defaults to None.
        title (str, optional):
            The overall title for the visualization. Defaults to an empty string.
        savepath (str | Path | None, optional):
            The file path to save the visualization. If None, the visualization
            will not be saved. Defaults to None.
    Returns:
        None: This function does not return anything. It displays the visualization
        and optionally saves it to the specified path.
    """

    n_classes = len(avg_saliency_by_class)
    n_cols = min(5, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, (label, saliency) in enumerate(avg_saliency_by_class.items()):
        ax = axes[i]
        ax.imshow(saliency, cmap="jet")
        ax.axis("off")
        ax_title = f"Class {label}"
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


class GradCAM:
    def __init__(self, model: nn.Module, device: torch.device | None = None):
        self.model = model
        self.target_layer = self._find_last_conv_layer(model)
        if self.target_layer is None:
            raise ValueError("No convolutional layer found in the model.")

        self._fwd_handle = None
        self._bwd_handle = None
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._orig_training = model.training

        self.device = device if device else next(model.parameters()).device

    # Manage hooks with context manager

    def __enter__(self):
        self.model.eval()
        assert self.target_layer
        self._fwd_handle = self.target_layer.register_forward_hook(self._store_acts)
        self._bwd_handle = self.target_layer.register_full_backward_hook(
            self._store_grads
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fwd_handle:
            self._fwd_handle.remove()
        if self._bwd_handle:
            self._bwd_handle.remove()
        self.model.train(self._orig_training)

    # Hooks

    def _store_acts(self, module, input, output):
        self._acts = output

    def _store_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0]

    @torch.no_grad()
    def _resize_like_input(self, cam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )

    def generate(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
        reduction: Callable = torch.mean,
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmaps for a batch.
        Args:
            x: input tensor (B, C, H, W)
            class_idx: if None, uses argmax over logits per-sample.
            reduction: function to reduce gradients over spatial dims before channel weights.
                       Default is mean, equivalent to GAP on grads.

        Returns:
            cam: (B, 1, H, W) in [0,1], resized to input size.
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        self.model.eval()

        if x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        with torch.enable_grad():
            logits = self.model(x)

        if class_idx is None:
            target_scores = logits.softmax(dim=1).argmax(dim=1)
        else:
            target_scores = torch.as_tensor([class_idx] * x.shape[0], device=device)

        scores = logits.gather(1, target_scores.unsqueeze(1)).squeeze(1)

        self.model.zero_grad(set_to_none=True)
        scores.sum().backward(retain_graph=True)

        if self._acts is None or self._grads is None:
            raise RuntimeError("Activations or gradients have not been recorded.")

        acts = self._acts
        grads = self._grads

        weights = reduction(grads, dim=(2, 3), keepdim=True)
        weights_relu = F.relu(weights)
        cam = (weights_relu * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-12)
        cam = self._resize_like_input(cam, x)

        return cam

    @staticmethod
    def overlay_on_image(
        img: torch.Tensor,
        cam: torch.Tensor,
        alpha: float = 0.65,
        colormap: str = "jet",
    ) -> torch.Tensor:
        """
        Alpha-blend CAM on top of the input image.
        Args:
            img: (B, C, H, W) in [0,1] or normalized; if 1-channel, it will be stacked to 3.
            cam: (B, 1, H, W) in [0,1]
            alpha: blending factor for CAM.
            colormap: Matplotlib colormap name.

        Returns:
            overlay: (B, 3, H, W)
        """

        if img.dim() > 4 or img.dim() < 2 or cam.dim() > 4 or cam.dim() < 4:
            raise ValueError("img and heatmap must be 2D, 3D or 4D tensor")

        # prepare image dims to rgb == [B, 3, W, H]
        if img.dim() < 3:
            img = img.unsqueeze(0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)

        # prepare heat map to rgb == [B, 3, W, H]
        cam_np = cam.squeeze().cpu().numpy()
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(cam_np)[..., :3]
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).to(img.device)

        overlay = (1 - alpha) * img + alpha * heatmap
        overlay = overlay.clamp(0, 1)
        if overlay.dim() < 3:
            overlay = overlay.unsqueeze(0)
        if overlay.dim() == 3:
            overlay = overlay.unsqueeze(0)

        return overlay

    @staticmethod
    def _find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
        last_conv = None
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
        return last_conv
