from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Determines the appropriate device for PyTorch operations based on availability.

    Args:
        prefer_mps (bool): If True, prefers the Metal Performance Shaders (MPS) backend
                           on macOS when CUDA is not available. Defaults to True.

    Returns:
        torch.device: The selected device, which can be "cuda" (if a CUDA-enabled GPU is available),
                      "mps" (if MPS is available and preferred), or "cpu" (as a fallback).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        prefer_mps
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device) -> Any:
    """
    Moves a given object to the specified PyTorch device (e.g., CPU or GPU).
    This function recursively transfers tensors, dictionaries, lists, and tuples
    to the specified device. Non-tensor objects that are not dictionaries, lists,
    or tuples are returned unchanged.
    Args:
        obj (Any): The object to be moved to the specified device. This can be a
                   tensor, dictionary, list, tuple, or any other type.
        device (torch.device): The target device to which the object should be moved.
    Returns:
        Any: The object moved to the specified device. The type of the returned object
             matches the input type, with tensors moved to the target device.
    """

    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device(o, device) for o in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj


def unpack_batch(batch: Any) -> Tuple[Any, Any]:
    """
    Unpacks a batch of data into input and target components.

    This function supports batches in the following formats:
    - A list or tuple with exactly two elements, where the first element is
      the input data and the second element is the target data.
    - A dictionary containing keys for input and target data. The input data
      can be under the keys 'inputs', 'x', or 'data', and the target data can
      be under the keys 'targets', 'y', or 'labels'.

    Args:
        batch (Any): The batch of data to unpack. Can be a list, tuple, or
                     dictionary.

    Returns:
        Tuple[Any, Any]: A tuple containing the input data and target data.

    Raises:
        ValueError: If the dictionary batch does not contain the required keys
                    for input and target data.
        TypeError: If the batch format is unsupported.
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        x = batch.get("inputs", batch.get("x", batch.get("data")))
        y = batch.get("targets", batch.get("y", batch.get("labels")))
        if x is None or y is None:
            raise ValueError(
                "Batch dict must contain keys 'inputs'/'x'/'data' and 'targets'/'y'/'labels'."
            )
        return x, y
    raise TypeError("Unsupported batch format.")


class Trainer:
    """
    A class for training PyTorch models with support for training and validation loops,
    device management, and model checkpointing.

    Attributes:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (Optimizer): The optimizer used for training.
        loss_fn (nn.Module): The loss function used for training.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset (default: None).
        epochs (int): Number of training epochs (default: 10).
        log_every_n_steps (int): Frequency of logging training progress (default: 1000).
        device (Optional[torch.device]): The device to use for training (default: None).
        prefer_mps (bool): Whether to prefer MPS (Metal Performance Shaders) if available (default: False).
        save_path (Optional[str | Path]): Directory or file path to save the model checkpoint (default: None).
        save_name (Optional[str]): Name of the model checkpoint file (default: None).

    Methods:
        fit() -> Dict[str, list]:
            Trains the model for the specified number of epochs and returns the training history.
            If a validation DataLoader is provided, validation loss is also computed and logged.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        log_every_n_steps: int = 1000,
        device: Optional[torch.device] = None,
        prefer_mps: bool = False,
        save_path: Optional[str | Path] = None,
        save_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.log_every_n_steps = log_every_n_steps
        self.device = device or get_device(prefer_mps=prefer_mps)

        # Move model to device
        self.model.to(self.device)

        # Handle save path and name
        self.save_path: Optional[Path] = None
        if save_path or save_name:
            self.save_dir = Path(save_path) if save_path else Path("models")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_name = save_name or "model.ckpt"
            self.save_path = self.save_dir / self.save_name
            if not self.save_path.suffix:
                self.save_path = self.save_path.with_suffix(".ckpt")

    def fit(self) -> Dict[str, list]:
        # Initialize history dictionary to store training and validation losses
        history: Dict[str, list] = {"train_loss": []}
        if self.val_loader is not None:
            history["val_loss"] = []

        # Loop through each epoch
        for epoch in range(1, self.epochs + 1):
            self.model.train()  # Set model to training mode
            running_loss, n_samples = 0.0, 0

            # Iterate over training batches
            for step, batch in enumerate(self.train_loader, start=1):
                # Unpack and move batch data to the target device
                x, y = unpack_batch(batch)
                x, y = to_device(x, self.device), to_device(y, self.device)

                # Forward pass
                preds = self.model(x)
                loss = self.loss_fn(preds, y)  # Compute loss

                # Backward pass and optimization
                self.optimizer.zero_grad(set_to_none=True)  # Clear gradients
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update model parameters

                # Accumulate loss and sample count for the current batch
                batch_size = (
                    y.shape[0] if hasattr(y, "shape") and len(y.shape) > 0 else 1
                )
                running_loss += float(loss.item()) * batch_size
                n_samples += batch_size

                if step % self.log_every_n_steps == 0:
                    print(
                        f"[Epoch {epoch}/{self.epochs}] step {step}: train_loss={running_loss / max(1, n_samples):.4f}"
                    )

            # Compute and store average training loss for the epoch
            epoch_train_loss = running_loss / max(1, n_samples)
            history["train_loss"].append(epoch_train_loss)

            # Perform validation if a validation DataLoader is provided
            if self.val_loader is not None:
                self.model.eval()  # Set model to evaluation mode
                val_loss_sum, val_n = 0.0, 0
                with torch.no_grad():  # Disable gradient computation for validation
                    for batch in self.val_loader:
                        # Unpack and move batch data to the target device
                        x, y = unpack_batch(batch)
                        x, y = to_device(x, self.device), to_device(y, self.device)

                        # Forward pass and compute loss
                        preds = self.model(x)
                        loss = self.loss_fn(preds, y)

                        # Accumulate validation loss and sample count
                        batch_size = (
                            y.shape[0]
                            if hasattr(y, "shape") and len(y.shape) > 0
                            else 1
                        )
                        val_loss_sum += float(loss.item()) * batch_size
                        val_n += batch_size

                # Compute and store average validation loss for the epoch
                epoch_val_loss = val_loss_sum / max(1, val_n)
                history["val_loss"].append(epoch_val_loss)

                print(
                    f"[Epoch {epoch}] train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f}"
                )
            else:
                print(f"[Epoch {epoch}] train_loss={epoch_train_loss:.4f}")

            # Save the model checkpoint if a save path is specified
            if self.save_path:
                torch.save(
                    self.model.state_dict(),
                    self.save_path,
                )

        return history


# Models already implemented


class SimpleNN(nn.Module):
    """
    SimpleNN is a simple feedforward neural network implemented using PyTorch's `nn.Module`.
    This model is designed for image classification tasks, where the input images are expected
    to have a shape of (28, 28). The network consists of a flattening layer followed by a
    sequential stack of fully connected layers with ReLU activations.
    Attributes:
        flatten (nn.Flatten): A layer that flattens the input tensor into a 1D tensor.
        linear_relu_stack (nn.Sequential): A sequential container of three fully connected
            layers with ReLU activations. The layers are:
            - Linear layer with input size 28*28 and output size 512, followed by ReLU.
            - Linear layer with input size 512 and output size 512, followed by ReLU.
            - Linear layer with input size 512 and output size 10.
    Methods:
        forward(x):
            Defines the forward pass of the network. Takes an input tensor `x`, flattens it,
            and passes it through the sequential stack of layers to produce logits.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 28, 28).
            Returns:
                torch.Tensor: Output logits of shape (batch_size, 10).
    """

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


class SimpleCNN(nn.Module):
    """
    SimpleCNN is a convolutional neural network (CNN) model implemented using PyTorch's `nn.Module`.
    It is designed for image classification tasks and includes methods to retrieve the activations
    and gradients of the last convolutional layer.
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 1 input channel, 32 output channels, and a kernel size of 3.
        conv2 (nn.Conv2d): Second convolutional layer with 32 input channels, 64 output channels, and a kernel size of 3.
        pool (nn.MaxPool2d): Max pooling layer with a kernel size of 2 and stride of 2.
        fc1 (nn.Linear): Fully connected layer with input size 64 * 5 * 5 and output size 120.
        fc2 (nn.Linear): Fully connected layer with input size 120 and output size 84.
        fc3 (nn.Linear): Fully connected layer with input size 84 and output size 10.
        _acts (torch.Tensor): Stores the activations of the last convolutional layer.
        _grads (torch.Tensor): Stores the gradients of the output with respect to the activations of the last convolutional layer.
    Methods:
        forward(x):
            Defines the forward pass of the network. Takes an input tensor `x` and returns the output tensor.
        get_activations():
            Returns the activations of the last convolutional layer.
        get_activations_gradient():
            Returns the gradients of the output with respect to the activations of the last convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self._acts = None  # A_k (activations of the last conv layer)
        self._grads = None  # dY/dA_k (gradients of the output)

    def _save_grad(self, grad):
        self._grads = grad

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = F.relu(self.conv2(x))
        self._acts = x
        self._acts.register_hook(self._save_grad)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_activations(self):
        return self._acts

    def get_activations_gradient(self):
        return self._grads
