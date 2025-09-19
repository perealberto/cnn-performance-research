# +
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

src: Path = Path.cwd().parent
data_dir: Path = src / "data"
models_dir: Path = src / "models"

assert src.exists()
if not data_dir.exists():
    data_dir.mkdir()
if not models_dir.exists():
    models_dir.mkdir()

# ensure using cuda device (install cuda utilities from torch)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# +
training_data: datasets.MNIST = datasets.MNIST(
    root=data_dir, transform=ToTensor(), download=True
)
test_data: datasets.MNIST = datasets.MNIST(
    root=data_dir, train=False, transform=ToTensor(), download=True
)

figure = plt.figure(figsize=(8, 8))
figure.suptitle("MNIST Samples")
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = int(torch.randint(len(training_data), size=(1,)).item())
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# +
# create dataloaders to store de datasets
train_dataloader: DataLoader = DataLoader(dataset=training_data, batch_size=64)
test_dataloader: DataLoader = DataLoader(dataset=test_data, batch_size=64)

# iterate over batches is possible with our DataLoader
train_features, train_labels = next(iter(train_dataloader))
print(f"Features shape: {train_features.size()}")
print(f"Labels shape: {train_labels.size()}")

# each batch has 64 images with shape (1, 28, 28)
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()


# +
class NeuralNetwork(nn.Module):
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


model = NeuralNetwork().to(device)
print(model)


# +
def train_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    size: int = len(dataloader.dataset)  # type: ignore
    model.train()  # set model to training
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * X.size(0)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(
    dataloader: DataLoader, model: NeuralNetwork, loss_fn: nn.CrossEntropyLoss, device
) -> None:
    size: int = len(dataloader.dataset)  # type: ignore
    model.eval()  # set model to predict
    num_batches = len(dataloader)

    test_loss: float = 0.0
    correct: float = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if num_batches > 0:
        test_loss /= num_batches
    if size > 0:
        correct /= size

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# +
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loop(test_dataloader, model, loss_fn, device)
print("Done!")


# +
test_features, test_labels = next(iter(test_dataloader))
sample_idx = torch.randint(len(test_features), size=(1,)).item()
model.eval()

img = test_features[sample_idx].unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(img)
    probs = torch.softmax(logits, dim=1)
    pred_label = int(probs.argmax(1).item())

true_label = test_labels[sample_idx]

plt.figure(1)
plt.title(f"Predicted: {pred_label} - Real: {true_label}")
plt.imshow(img.squeeze(), cmap="gray")
plt.show()
