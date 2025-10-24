"""
Metrics and evaluation utilities for sentiment analysis.

This module provides comprehensive metrics calculation including
precision, recall, F1-score, confusion matrix, and classification report.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Dict:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (auto-detected if None)

    Returns:
        Dictionary containing all metrics

    Example:
        >>> y_true = ['positive', 'negative', 'positive']
        >>> y_pred = ['positive', 'positive', 'positive']
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.667
    """
    if labels is None:
        labels = sorted(set(y_true + y_pred))

    logger.info(f"Calculating metrics for {len(y_true)} samples...")

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )

    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Classification report
    class_report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'total_samples': len(y_true),
        'labels': labels
    }

    logger.info(f"Metrics calculated: Accuracy={accuracy:.4f}, F1={f1_macro:.4f}")

    return metrics


def print_metrics_report(metrics: Dict) -> None:
    """
    Print a formatted metrics report.

    Args:
        metrics: Metrics dictionary from calculate_metrics()

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print_metrics_report(metrics)
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS REPORT")
    print("=" * 60)

    print(f"\nTotal Samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for label, class_metrics in metrics['per_class_metrics'].items():
        print(
            f"{label:<12} "
            f"{class_metrics['precision']:<12.4f} "
            f"{class_metrics['recall']:<12.4f} "
            f"{class_metrics['f1_score']:<12.4f}"
        )

    print("\n" + "=" * 60)


def plot_confusion_matrix(
    metrics: Dict,
    normalize: bool = False,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix using matplotlib.

    Args:
        metrics: Metrics dictionary from calculate_metrics()
        normalize: Whether to normalize the confusion matrix
        title: Plot title

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> plot_confusion_matrix(metrics, normalize=True)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(metrics['confusion_matrix'])
        labels = metrics['labels']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    except ImportError:
        logger.warning(
            "matplotlib and/or seaborn not installed. "
            "Install them to visualize confusion matrix: "
            "pip install matplotlib seaborn"
        )


def create_metrics_dataframe(metrics: Dict) -> pd.DataFrame:
    """
    Create a pandas DataFrame from metrics.

    Args:
        metrics: Metrics dictionary from calculate_metrics()

    Returns:
        DataFrame with per-class metrics

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> df = create_metrics_dataframe(metrics)
        >>> print(df)
    """
    data = []
    for label, class_metrics in metrics['per_class_metrics'].items():
        data.append({
            'Class': label,
            'Precision': class_metrics['precision'],
            'Recall': class_metrics['recall'],
            'F1-Score': class_metrics['f1_score']
        })

    df = pd.DataFrame(data)

    # Add macro and weighted averages
    macro_row = {
        'Class': 'macro avg',
        'Precision': metrics['precision_macro'],
        'Recall': metrics['recall_macro'],
        'F1-Score': metrics['f1_macro']
    }

    weighted_row = {
        'Class': 'weighted avg',
        'Precision': metrics['precision_weighted'],
        'Recall': metrics['recall_weighted'],
        'F1-Score': metrics['f1_weighted']
    }

    df = pd.concat([df, pd.DataFrame([macro_row, weighted_row])], ignore_index=True)

    return df


def compare_models(
    models_metrics: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Compare metrics from multiple models.

    Args:
        models_metrics: Dictionary mapping model names to their metrics

    Returns:
        Comparison DataFrame

    Example:
        >>> nb_metrics = calculate_metrics(y_true, y_pred_nb)
        >>> bert_metrics = calculate_metrics(y_true, y_pred_bert)
        >>> comparison = compare_models({
        ...     'Naive Bayes': nb_metrics,
        ...     'BERT': bert_metrics
        ... })
        >>> print(comparison)
    """
    data = []
    for model_name, metrics in models_metrics.items():
        data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision (Macro)': metrics['precision_macro'],
            'Recall (Macro)': metrics['recall_macro'],
            'F1-Score (Macro)': metrics['f1_macro'],
            'F1-Score (Weighted)': metrics['f1_weighted']
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    return df


def calculate_error_analysis(
    y_true: List[str],
    y_pred: List[str],
    texts: List[str]
) -> Dict:
    """
    Analyze prediction errors.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Original text samples

    Returns:
        Dictionary containing error analysis

    Example:
        >>> error_analysis = calculate_error_analysis(y_true, y_pred, texts)
        >>> print(f"Error rate: {error_analysis['error_rate']:.2%}")
    """
    errors = []
    for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
        if true != pred:
            errors.append({
                'index': i,
                'text': text,
                'true_label': true,
                'predicted_label': pred
            })

    error_rate = len(errors) / len(y_true) if y_true else 0

    # Analyze error patterns
    error_patterns = {}
    for error in errors:
        pattern = f"{error['true_label']} -> {error['predicted_label']}"
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

    return {
        'total_errors': len(errors),
        'error_rate': error_rate,
        'errors': errors,
        'error_patterns': error_patterns
    }
