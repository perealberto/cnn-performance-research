import time
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


@torch.no_grad()
def _collect_outputs(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Collects predictions, probabilities, and true labels from a model over a given dataloader,
    while also calculating the average cross-entropy loss and inference time per sample.
    Args:
        dataloader (DataLoader): The DataLoader providing the input data and labels.
        model (nn.Module): The PyTorch model to evaluate.
        device (torch.device): The device (CPU or GPU) to run the model on.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            - y_true (np.ndarray): Array of true labels.
            - y_pred (np.ndarray): Array of predicted labels.
            - y_prob (np.ndarray): Array of predicted probabilities for each class.
            - avg_ce (float): Average cross-entropy loss over all samples.
            - infer_time_ms (float): Average inference time per sample in milliseconds.
    """
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
    """
    Evaluates a given model on a dataset and computes various performance metrics.
    Args:
        name (str): The name or identifier for the model being evaluated.
        dataloader (DataLoader): A PyTorch DataLoader providing the dataset for evaluation.
        model (nn.Module): The PyTorch model to be evaluated.
        device (torch.device): The device (CPU or GPU) on which the evaluation is performed.
    Returns:
        Dict[str, float]: A dictionary containing the following metrics:
            - "name": The name of the model.
            - "accuracy": The accuracy of the model.
            - "loss_ce": The average cross-entropy loss.
            - "precision_macro": The macro-averaged precision score.
            - "recall_macro": The macro-averaged recall score.
            - "f1_macro": The macro-averaged F1 score.
            - "f1_weighted": The weighted-averaged F1 score.
            - "roc_auc_macro": The macro-averaged ROC AUC score (NaN if only one class is present).
            - "infer_time_ms_per_sample": The average inference time per sample in milliseconds.
            - "classification_report": A detailed classification report as a dictionary.
            - "confusion_matrix": The normalized confusion matrix.
    """

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


def top_k_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 3, top_mis_as: int = 3
) -> List[Tuple[int, float, List[Tuple[int, int, float]]]]:
    """
    Identify the top-k classes with the highest error rates and provide details
    about the most common misclassifications for each of these classes.
    Args:
        y_true (np.ndarray): Ground truth (true class labels).
        y_pred (np.ndarray): Predicted class labels.
        k (int, optional): Number of classes with the highest error rates to return. Defaults to 3.
        top_mis_as (int, optional): Number of top misclassified classes to include for each class. Defaults to 3.
    Returns:
        List[Tuple[int, float, List[Tuple[int, int, float]]]]:
            A list of tuples, where each tuple corresponds to a class with high error rates.
            Each tuple contains:
                - int: The class label.
                - float: The error rate for the class (as a fraction).
                - List[Tuple[int, int, float]]: A list of tuples for the top misclassified classes, where each tuple contains:
                    - int: The misclassified class label.
                    - int: The number of instances misclassified as this class.
                    - float: The percentage of total instances misclassified as this class.
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
    Prints the top-k most confusing classes based on prediction errors.

    This function identifies the classes with the highest error rates and displays
    the most common misclassifications for each of these classes. It provides a
    summary of confusion patterns in the model's predictions.

    Args:
        title (str): A title or description for the confusion analysis.
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.
        k (int, optional): The number of classes with the highest error rates to display. Defaults to 3.
        top_mis_as (int, optional): The number of most common misclassifications to show for each class. Defaults to 3.

    Returns:
        None
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


def mcnemar_test_from_predictions(
    y_true: np.ndarray,
    y_pred_A: np.ndarray,
    y_pred_B: np.ndarray,
    continuity_correction: bool = False,
) -> Dict[str, float]:
    """
    Perform McNemar's test to compare the performance of two classifiers.
    McNemar's test is used to determine whether there is a significant difference
    between the predictions of two classifiers on the same dataset. It is based
    on the number of discordant pairs (instances where one classifier is correct
    and the other is incorrect).
    Args:
        y_true (np.ndarray): Ground truth (correct) labels.
        y_pred_A (np.ndarray): Predictions from classifier A.
        y_pred_B (np.ndarray): Predictions from classifier B.
        continuity_correction (bool, optional): Whether to apply the continuity
            correction for the chi-squared statistic. Defaults to False.
    Returns:
        Dict[str, float]: A dictionary containing the following keys:
            - "n01" (int): Number of instances where classifier A is correct
              and classifier B is incorrect.
            - "n10" (int): Number of instances where classifier A is incorrect
              and classifier B is correct.
            - "n_discordant" (int): Total number of discordant pairs (n01 + n10).
            - "p_value" (float): The p-value of the test, indicating the
              significance of the difference between the classifiers.
    Notes:
        - If the total number of discordant pairs (n01 + n10) is less than 25,
          the binomial distribution is used to compute the p-value.
        - If the total number of discordant pairs is 25 or more, the chi-squared
          distribution is used.
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
