from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
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
    """
    GradCAM is a class for generating Gradient-weighted Class Activation Maps (Grad-CAM)
    to visualize the regions of an input image that are most relevant for a model's predictions.
    Attributes:
        model (nn.Module): The neural network model for which Grad-CAM is applied.
        target_layer (nn.Module): The last convolutional layer of the model, automatically detected.
        gradients (torch.Tensor | None): Gradients of the target layer, recorded during backpropagation.
        activations (torch.Tensor | None): Activations of the target layer, recorded during the forward pass.
        device (torch.device): The device (CPU or GPU) on which the model and input tensors are processed.
    Methods:
        __init__(model: nn.Module, device: torch.device | None = None):
            Initializes the GradCAM instance with the given model and device.
        _get_last_conv_layer(model: nn.Module) -> nn.Module:
            Identifies and returns the last convolutional layer in the model.
            Raises a ValueError if no convolutional layer is found.
        _forward_hook(module, input, output):
            A forward hook to capture the activations of the target layer during the forward pass.
        _full_backward_hook(module, grad_input, grad_output):
            A backward hook to capture the gradients of the target layer during backpropagation.
        compute_heatmap(x: torch.Tensor) -> tuple[np.ndarray, int, float]:
            Computes the Grad-CAM heatmap for a single input image tensor.
            Args:
                x (torch.Tensor): The input image tensor of shape (1, C, H, W).
            Returns:
                tuple[np.ndarray, int, float]: A tuple containing:
                    - heatmap (np.ndarray): The Grad-CAM heatmap as a 2D NumPy array.
                    - class_idx (int): The predicted class index.
                    - predicted_prob (float): The predicted probability for the class.
    """

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
        x = x.to(self.device).requires_grad_(True)

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


def compute_superimposed_image(
    img: torch.Tensor | np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a heatmap (Grad-CAM or saliency map) on the original image.

    Args:
        img (torch.Tensor | np.ndarray): The original image as a tensor or numpy array.
        heatmap (np.ndarray): The heatmap to overlay (Grad-CAM or saliency map).
        alpha (float): The transparency factor for the heatmap overlay.

    Returns:
        np.ndarray: The normalized superimposed image in RGB format.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    if img.ndim >= 3 and img.shape[0] in {1, 3}:
        img = img.squeeze(0).transpose(1, 2, 0)
    if img.ndim == 2 or img.shape[-1] != 3:
        img = cv2.cvtColor((255 * np.array(img)).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (255 * heatmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # Ensure the output image is in RGB format
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Normalize the superimposed image to the range [0, 1]
    superimposed_img = superimposed_img.astype(np.float32) / 255.0

    return superimposed_img
