import pytest
import numpy as np
from tqdm import tqdm
import logging

# Set up the logging
logging.basicConfig(level=logging.INFO)

# --- Accuracy Tests ---

# Overall accuracy

def test_overall_accuracy(predictions):
    all_labels, all_predictions = predictions
    acc = (all_predictions == all_labels).sum() / len(all_labels) * 100
    logging.info(f"Overall accuracy of ViT: {acc:.2f}% ")
    assert acc > 90, f"Overall accuracy too low: {acc:.2f}%"

# Per-class accuracy must be greater than 75% for every class
def test_per_class_accuracy(predictions):
    all_labels, all_predictions = predictions
    num_classes = 2
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    failed_classes = []

    for true_label, pred_label in zip(all_labels, all_predictions):
        total[true_label] += 1
        if true_label == pred_label:
            correct[true_label] += 1

    for cls in range(num_classes):
        acc = correct[cls] / total[cls] * 100
        if acc < 75:
            failed_classes.append((cls, acc))
        
    if failed_classes:
        summary = ", ".join([f"{cls} ({acc:.2f}%)" for cls, acc in failed_classes])
        pytest.fail(f"Some classes have low accuracy: {summary}")
