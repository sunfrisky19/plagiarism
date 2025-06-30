"""
Utility functions for classification tasks: class weights, metrics, plotting, and error analysis.
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path

def compute_class_weights(labels):
    """
    Compute class weights for imbalanced datasets.
    Args:
        labels (list or np.ndarray): List of class labels.
    Returns:
        dict: Mapping from class index to weight.
    """
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {i: w for i, w in enumerate(weights)}

def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    """
    Plot and save a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def plot_normalized_confusion_matrix(y_true, y_pred, class_names, filename):
    """
    Plot and save a normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='PuRd', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, class_names, filename):
    """
    Plot and save ROC curves for each class.
    """
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (np.array(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_precision_recall_curves(y_true, y_pred_proba, class_names, filename):
    """
    Plot and save precision-recall curves for each class.
    """
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (np.array(y_true) == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true_binary, y_pred_proba[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.
    Returns a dict with accuracy, f1, precision, recall.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def enhanced_error_analysis(y_true, y_pred, y_pred_proba, texts, class_names, output_dir, id2label):
    """
    Perform detailed error analysis and save results as JSON/CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    detailed_results = []
    for i in range(len(y_true)):
        probs = y_pred_proba[i].tolist() if y_pred_proba is not None else None
        max_prob = probs[y_pred[i]] if probs is not None else None
        detailed_results.append({
            'text': texts[i],
            'true_label': id2label[y_true[i]],
            'true_label_id': int(y_true[i]),
            'predicted_label': id2label[y_pred[i]],
            'predicted_label_id': int(y_pred[i]),
            'prediction_probabilities': probs,
            'max_probability': max_prob,
            'is_correct': y_true[i] == y_pred[i]
        })
    # Save detailed results
    detailed_path = output_dir / 'detailed_predictions.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    # Save misclassified instances
    misclassified_df = pd.DataFrame([r for r in detailed_results if not r['is_correct']])
    misclassified_path = output_dir / 'misclassified_instances.csv'
    misclassified_df.to_csv(misclassified_path, index=False)
    # Per-class error breakdown
    error_analysis = {
        'total_samples': len(y_true),
        'correct_predictions': sum(1 for x in detailed_results if x['is_correct']),
        'incorrect_predictions': sum(1 for x in detailed_results if not x['is_correct']),
        'accuracy': accuracy_score(y_true, y_pred),
        'per_class_error_breakdown': {}
    }
    for i, class_name in enumerate(class_names):
        class_instances = [x for x in detailed_results if x['true_label'] == class_name]
        class_errors = [x for x in class_instances if not x['is_correct']]
        error_analysis['per_class_error_breakdown'][class_name] = {
            'total_instances': len(class_instances),
            'total_errors': len(class_errors),
            'error_rate': len(class_errors) / len(class_instances) if class_instances else 0,
            'common_misclassifications': {}
        }
        misclass_counter = {}
        for error in class_errors:
            misclass_key = error['predicted_label']
            misclass_counter[misclass_key] = misclass_counter.get(misclass_key, 0) + 1
        error_analysis['per_class_error_breakdown'][class_name]['common_misclassifications'] = misclass_counter
    # Save error analysis
    error_analysis_path = output_dir / 'error_analysis.json'
    with open(error_analysis_path, 'w') as f:
        json.dump(error_analysis, f, indent=2, default=str)
    return error_analysis 