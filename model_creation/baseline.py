"""
Baseline script for sentence-level classification using SciBERT.
Performs data loading, preprocessing, model training, evaluation, and error analysis.
"""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
)
import seaborn as sns
from pathlib import Path
from model_creation.utils.classification_utils import (
    compute_class_weights,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    enhanced_error_analysis
)


# Load the dataset from a CSV file
def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['source', 'article_id', 'category']) # 'full_text', 'prompt', 'input_tokens', 'output_tokens',
    df = df.rename(columns={'text_type': 'label'})
    return df

# Create DatasetDict
dataset = DatasetDict({
    'train': Dataset.from_pandas(load_csv('/ukp-storage-1/naguleswaran/scibert_baseline/data/labeled_sentences_train_data.csv')),
    'test': Dataset.from_pandas(load_csv('/ukp-storage-1/naguleswaran/scibert_baseline/data/labeled_sentences_test_data.csv')),
    'validation': Dataset.from_pandas(load_csv('/ukp-storage-1/naguleswaran/scibert_baseline/data/labeled_sentences_val_data.csv'))
})

# # Ultra-short testing mode
# def quick_dev_mode(dataset, num_samples=500):
#     return dataset.select(range(min(num_samples, len(dataset))))

# dataset['train'] = quick_dev_mode(dataset['train'])
# dataset['validation'] = quick_dev_mode(dataset['validation'])
# dataset['test'] = quick_dev_mode(dataset['test'])

label_names = sorted(set(label for label in dataset["train"]["label"]))
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for label, idx in label2id.items()}

def encode_labels(examples):
    examples['label'] = label2id[examples['label']]
    return examples

dataset = dataset.map(encode_labels)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['sentence'],
        padding='max_length',
        truncation=True,
        max_length=256 # 512
    )

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_labels = tokenized_datasets['train']['label']
class_weights_dict = compute_class_weights(train_labels)
print("Class weights:", class_weights_dict)

# Model initialization with explicit label mapping
model = AutoModelForSequenceClassification.from_pretrained(
    'allenai/scibert_scivocab_uncased',
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
    }

    return metrics

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-scibert-abstract-classifier-model-sentence-level-weighted-softmax',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',
    report_to='none'  # Change to 'wandb' if using Weights & Biases
)

# Custom Trainer to handle class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use class weights in loss calculation
        if self.class_weights is not None:
            # Convert weights to float tensor of the correct shape
            weights = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))],
                dtype=torch.float,  # Ensure float type
                device=model.device
            )
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

# Initialize Trainer with class weights
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=class_weights_dict
)

# Train and evaluate
trainer.train()

# Final evaluation on test set
test_results = trainer.evaluate(tokenized_datasets['test'])
print("\nTest Set Performance:")
print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"F1 Score: {test_results['eval_f1']:.4f}")
print(f"Precision: {test_results['eval_precision']:.4f}")
print(f"Recall: {test_results['eval_recall']:.4f}")


# Prepare results directory
results_dir = Path('./results-scibert-abstract-classifier-model-sentence-level-weighted-softmax')
results_dir.mkdir(exist_ok=True)

# Generate classification report with Matrix
def get_classification_report_matrix(dataset_split):
    predictions = trainer.predict(tokenized_datasets[dataset_split])
    preds = np.argmax(predictions.predictions, axis=1)

    # Generate and save both standard and normalized confusion matrices
    cm_standard_path = results_dir / f'confusion_matrix_{dataset_split}-weighted.png'
    cm_normalized_path = results_dir / f'normalized_confusion_matrix_{dataset_split}-weighted.png'
    
    plot_confusion_matrix(
        preds, 
        predictions.label_ids,
        cm_standard_path
    )

    plot_normalized_confusion_matrix(
        preds, 
        predictions.label_ids,
        cm_normalized_path
    )
    
    return classification_report(
        predictions.label_ids,
        preds,
        target_names=label_names
    )

print("\nValidation Set Classification Report & Matrix:")
print(get_classification_report_matrix('validation'))

print("\nTest Set Classification Report & Matrix:")
print(get_classification_report_matrix('test'))

print("\nGenerating detailed evaluation plots...")

print("\nValidation Set Classification Report ROC:")
plot_roc_curves('validation')

print("\nTest Set Classification Report ROC:")
plot_roc_curves('test')

print("\nValidation Set Classification Report Recall:")
plot_precision_recall_curves('validation')

print("\nTest Set Classification Report Recall:")
plot_precision_recall_curves('test')

# After training and initial evaluation, add:
print("\nPerforming Validation Set Error Analysis:")
validation_error_analysis = enhanced_error_analysis(
    tokenized_datasets, 
    trainer, 
    'validation', 
    label_names, 
    label2id, 
    id2label
)

print("\nPerforming Test Set Error Analysis:")
test_error_analysis = enhanced_error_analysis(
    tokenized_datasets, 
    trainer, 
    'test', 
    label_names, 
    label2id, 
    id2label
)


# Save model and tokenizer
model.save_pretrained('scibert-abstract-classifier-model-sentence-level-weighted-softmax', safe_serialization=True)
tokenizer.save_pretrained('scibert-abstract-classifier-model-sentence-level-weighted-softmax', safe_serialization=True)


# Save label mappings
import joblib
joblib.dump(label2id, 'label2id-sentence-level-weighted-softmax.pkl')
joblib.dump(id2label, 'id2label-sentence-level-weighted-softmax.pkl')