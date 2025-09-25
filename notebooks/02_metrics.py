# +
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

datasets_path: Path = Path.cwd() / "data"
if not datasets_path.exists():
    datasets_path.mkdir()

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
# -

# ## Load dataloaders

# +
train_dataset = datasets.MNIST(root=datasets_path, transform=ToTensor())
test_dataset = datasets.MNIST(root=datasets_path, train=False, transform=ToTensor())

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


def train_loop(
    dataloader: DataLoader,
    model,
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
    dataloader: DataLoader, model, loss_fn: nn.CrossEntropyLoss, device
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


def train_batch(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int,
    model,
    loss_fn,
    optimizer,
    device,
) -> None:
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")


# -

# ## Load models


# +
# simple neural network from '01_torch.py'
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


simple_nn = NeuralNetwork().to(device)
print(simple_nn)


# +
# simple convolutional neural network
class CNN(nn.Module):
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


simple_cnn = CNN().to(device)
print(simple_cnn)
# -

# ## Train models

# simple neural network
loss_fn = nn.CrossEntropyLoss()
optimizer_nn = torch.optim.SGD(simple_nn.parameters())
train_batch(
    train_dataloader, test_dataloader, 10, simple_nn, loss_fn, optimizer_nn, device
)

# convolutional neural network
loss_fn = nn.CrossEntropyLoss()
optimizer_cnn = torch.optim.SGD(simple_cnn.parameters())
train_batch(
    train_dataloader, test_dataloader, 10, simple_cnn, loss_fn, optimizer_cnn, device
)


# ## Evaluate


# +
@torch.no_grad()
def _collect_outputs(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Returns (y_true, y_pred, y_prob, avg_ce_loss, avg_infer_time_per_sample_ms)."""
    model.eval()
    ce_sum = 0.0
    n_samples = 0
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    start_all = time.perf_counter()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        t0 = time.perf_counter()  # noqa: F841
        logits = model(X)
        _ = logits.argmax(1)  # forces execution to measure time
        t1 = time.perf_counter()  # noqa: F841

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)

        # CE per batch (sum) to average later
        ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n_samples += y.size(0)

        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(preds.detach().cpu().numpy())
        y_prob_list.append(probs.detach().cpu().numpy())

    total_time = time.perf_counter() - start_all
    infer_time_ms = (total_time / n_samples) * 1000 if n_samples > 0 else float("nan")

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list)

    avg_ce = ce_sum / n_samples
    return y_true, y_pred, y_prob, avg_ce, infer_time_ms


def evaluate_model(
    name: str, dataloader: DataLoader, model: nn.Module, device: torch.device
) -> Dict[str, float]:
    """Evaluates the model on the given dataloader and returns the following metrics:
    - model name
    - accuracy
    - loss (cross-entropy)
    - precision (macro)
    - recall (macro)
    - F1 score (macro)
    - F1 score (weighted)
    - ROC AUC (macro)
    - inference time (ms/sample)
    - classification report (per class)
    - confusion matrix (normalized)"""

    y_true, y_pred, y_prob, avg_ce, infer_ms = _collect_outputs(
        dataloader, model, device
    )

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "loss_ce": avg_ce,
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_macro": (
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
        "infer_time_ms_per_sample": infer_ms,
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, normalize="true"),
    }

    return metrics


# -


def mcnemar_test_from_predictions(
    y_true: np.ndarray,
    y_pred_A: np.ndarray,
    y_pred_B: np.ndarray,
    continuity_correction: bool = False,
) -> Dict[str, float]:
    """
    McNemar test to compare two classifiers on paired data.
    Table:
          B correct   B incorrect
    A correct     n00         n01
    A incorrect   n10         n11
    The statistic uses the discordant pairs n01 and n10.
    """
    assert y_true.shape == y_pred_A.shape == y_pred_B.shape
    A_ok = y_pred_A == y_true
    B_ok = y_pred_B == y_true

    n01 = np.sum(A_ok & ~B_ok)  # A correct, B incorrect
    n10 = np.sum(~A_ok & B_ok)  # A incorrect, B correct
    n = n01 + n10

    pvalue = 0.0

    from scipy.stats import binom, chi2

    n_min, n_max = sorted([n01, n10])
    corr = int(continuity_correction)
    if (n_min + n_max) < 25:
        pvalue = 2 * binom.cdf(n_min, n_min + n_max, 0.5) - binom.pmf(
            n_min, n_min + n_max, 0.5
        )
    else:
        chi2_statistic = (abs(n_min - n_max) - corr) ** 2 / (n_min + n_max)
        pvalue = chi2.sf(chi2_statistic, 1)

    return {
        "n01": int(n01),
        "n10": int(n10),
        "n_discordant": int(n),
        "p_value": float(pvalue),
    }


metrics_nn = evaluate_model("MLP simple", test_dataloader, simple_nn, device)
metrics_cnn = evaluate_model("CNN simple", test_dataloader, simple_cnn, device)

y_true_NN, y_pred_NN, _, _, _ = _collect_outputs(test_dataloader, simple_nn, device)
y_true_CNN, y_pred_CNN, _, _, _ = _collect_outputs(test_dataloader, simple_cnn, device)
assert np.array_equal(y_true_NN, y_true_CNN), "y_true must be the same for NN y CNN."
comp = mcnemar_test_from_predictions(y_true_NN, y_pred_NN, y_pred_CNN, True)

print("\n=== Comparation NN vs CNN (McNemar exacto) ===")
print(
    f"Discordants: Success NN / Fails CNN = n01={comp['n01']}, Fails NN / Success CNN = n10={comp['n10']}"
)
print(
    f"Total discordants = {comp['n_discordant']}, p-value = {comp['p_value']:.4f}, p-value < alpha (0.05) = {comp['p_value'] < 0.05}"
)


def _row(m):
    return [
        m["name"],
        f"{m['accuracy'] * 100:.2f}%",
        f"{m['loss_ce']:.4f}",
        f"{m['f1_macro']:.4f}",
        f"{m['precision_macro']:.4f}",
        f"{m['recall_macro']:.4f}",
        f"{m['roc_auc_macro']:.4f}",
        f"{m['infer_time_ms_per_sample']:.3f} ms",
    ]


headers = [
    "Model",
    "Accuracy",
    "CE loss",
    "F1 (macro)",
    "Prec. (macro)",
    "Recall (macro)",
    "ROC-AUC (macro)",
    "Avg pred time",
]
table = [_row(metrics_nn), _row(metrics_cnn)]

colw = [max(len(h), max(len(r[i]) for r in table)) for i, h in enumerate(headers)]
print("\n=== Metrics ===")
print(" | ".join(h.ljust(colw[i]) for i, h in enumerate(headers)))
print("-|-".join("-" * w for w in colw))
for r in table:
    print(" | ".join(r[i].ljust(colw[i]) for i in range(len(headers))))


# ## Top 3 most confusing


# +
def top_k_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 3, top_mis_as: int = 3
) -> List[Tuple[int, float, List[Tuple[int, int, float]]]]:
    """
    Returns a list with the top-k most confusing true classes:
      [(class, error_rate, [(pred_class, count, pct_over_class), ... ]), ...]
    where the sub-list contains the 'top_mis_as' predicted classes most confused with.
    """
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    errors = totals - correct
    with np.errstate(divide="ignore", invalid="ignore"):
        err_rate = np.where(totals > 0, errors / totals, 0.0)

    # Sort classes by descending error rate
    order = np.argsort(err_rate)[::-1]

    results = []
    for idx in order[:k]:
        row = cm[idx].astype(int).copy()
        row[idx] = 0  # ignore correct predictions (diagonal)
        # Top predicted classes mistakenly classified for this true 'idx'
        top_idx = np.argsort(row)[::-1][:top_mis_as]
        conf_list = []
        for j in top_idx:
            if row[j] == 0:  # if no relevant errors remain, stop
                continue
            pct = (row[j] / totals[idx]) * 100 if totals[idx] > 0 else 0.0
            conf_list.append((int(labels[j]), int(row[j]), float(pct)))
        results.append((int(labels[idx]), float(err_rate[idx]), conf_list))

    return results


def print_top_k_confusions(
    title: str, y_true: np.ndarray, y_pred: np.ndarray, k: int = 3, top_mis_as: int = 3
) -> None:
    """
    Prints the top-k most confusing true classes along with their error rates
    and the predicted classes they are most confused with.
    """
    print(f"\n=== Top {k} most confusing — {title} ===")
    results = top_k_confusions(y_true, y_pred, k=k, top_mis_as=top_mis_as)
    for cls, er, confs in results:
        er_pct = er * 100
        if confs:
            conf_str = "; ".join(
                [f"{pred} ({cnt} → {pct:.1f}%)" for pred, cnt, pct in confs]
            )
        else:
            conf_str = "—"
        print(f"Class {cls}: error {er_pct:.2f}%  |  Confused with → {conf_str}")


# -

print_top_k_confusions("MLP simple", y_true_NN, y_pred_NN, k=3, top_mis_as=3)
print_top_k_confusions("CNN simple", y_true_CNN, y_pred_CNN, k=3, top_mis_as=3)
