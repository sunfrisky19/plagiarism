"""
Fine-tuning script for LLM-based sentence classification with LoRA and class weighting.
Handles data loading, preprocessing, model training, evaluation, and error analysis.
"""
from pathlib import Path
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
torch.cuda.empty_cache()
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    AutoPeftModelForSequenceClassification
)
import numpy as np
from trl import SFTTrainer
from transformers import TrainingArguments 
from transformers import EarlyStoppingCallback
import evaluate
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")  
recall_metric = evaluate.load("recall")
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from model_creation.utils.classification_utils import (
    compute_class_weights,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def create_model_dir(task_name, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, recreate_dir=True):
    # create model dir
    output_dir = Path("./new_results-abstract-sentence-all-trainbatch-32-256-trainepoch-6-grad_acc-8-target_modules-all-linear-metrics-llama3.1-8B-Instruct-Epoch-6-lora_r-64_alpha-64-lora_dropout-0.1-lr-2e-4-new") # maxlength-512-lora-short-batch-4
    if not output_dir.exists():
            output_dir.mkdir()
    output_dir = output_dir/task_name
    if not output_dir.exists():
            output_dir.mkdir()
    if not output_dir.exists():
            output_dir.mkdir()
    model_folder_name = Path(model_path).stem +'_'+ f'_lora-r{lora_r}-a{lora_alpha}-d{lora_dropout}_lr{learning_rate}'
    model_folder_name += f'_bs{per_device_train_batch_size}_ep{train_epochs}_{train_type}_{test_type}_ml{max_length}'
    output_dir = Path(output_dir/model_folder_name)
    if output_dir.exists():
         if recreate_dir:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
    else:
        output_dir.mkdir()
    return output_dir

def collate_fn(examples, device=None):
    for example in examples:
        example['input_ids'] = torch.as_tensor(example['input_ids_text'])
        example['attention_mask'] = torch.as_tensor(example['attention_mask_text'])
        example['label'] = torch.as_tensor(example['label'])
    input_ids = torch.stack([torch.as_tensor(example["input_ids"]) for example in examples])
    attention_masks = torch.stack([torch.as_tensor(example["attention_mask"]) for example in examples])
    input_ids = torch.squeeze(input_ids, dim=1)
    attention_masks = torch.squeeze(attention_masks, dim=1)
    labels = torch.stack([torch.as_tensor(example["label"]) for example in examples])

    if device is not None:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
    return {"input_ids": input_ids, 
            "attention_mask": attention_masks,
            "labels": labels}

def compute_metrics(eval_pred):   
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Enhanced metrics calculation
    metrics = {
        'accuracy': accuracy.compute(predictions=predictions, references=labels),
        'f1': f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1'],
        'precision': precision_metric.compute(predictions=predictions, references=labels, average='weighted')['precision'],
        'recall': recall_metric.compute(predictions=predictions, references=labels, average='weighted')['recall']
    }
    
    return metrics

def print_trainable_parameters(model, use_4bit = False):
    """Prints the number of trainable parameters in the model.
    :param model: PEFT model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2
    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

def create_input_tokenizer_and_label(sample, label2id, tokenizer, max_length):

    label = label2id[sample['label']]
    sample["label"] = label

    sample['input_ids_text'] = tokenizer.encode_plus(sample["text"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    sample['attention_mask_text'] = tokenizer.encode_plus(sample["text"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
    return sample

def preprocess_data(dataset, label2id, tokenizer, max_length=256):
    """
    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param input_type (str): Type of input text
    """
    # perpare input text and label
    print('max_length: ', max_length)
    
    dataset = dataset.rename_column('sentence', 'text')

    dataset = dataset.map(lambda x: create_input_tokenizer_and_label(x, label2id, tokenizer, max_length),keep_in_memory=True)
    # Shuffle dataset
    seed = 42
    dataset = dataset.shuffle(seed = seed)
    return dataset

# Custom loss function class for weighted loss
class CustomTrainer(SFTTrainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Added **kwargs to handle additional arguments like num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            # Convert weights to a tensor
            weights = torch.tensor(self.class_weights, dtype=torch.float, device=model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss
    
def fine_tune(model,
            tokenizer,
            train_ds = None,
            val_ds = None,
            lora_r = 128,
            lora_alpha = 128,
            lora_dropout = 0.1,
            learning_rate = 2e-4,
            per_device_train_batch_size = 32,
            train_epochs = 10,
            output_dir = None,
            bias = 'none',
            target_modules="all-linear",
            task_type = "SEQ_CLS",
            max_seq_length = 4096,
            class_weights = None  # Added class weights parameter
            ):
    print('fine-tuning....')
    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()
    
    # Prepare the model for training 
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    args = TrainingArguments(
                output_dir = output_dir,
                num_train_epochs=train_epochs,
                per_device_train_batch_size = per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps = 8,
                learning_rate = learning_rate, 
                logging_steps=10,
                fp16 = True,
                weight_decay=0.001,
                max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
                max_steps=-1,
                warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
                group_by_length=True,
                lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                report_to="tensorboard",                  # report metrics to tensorboard
                evaluation_strategy="epoch",              # save checkpoint every epoch
                save_strategy="epoch",
                gradient_checkpointing=True,              # use gradient checkpointing to save memory
                optim="paged_adamw_32bit",
                remove_unused_columns=False,
                load_best_model_at_end=True,
                # metric_for_best_model="eval_accuracy",
                metric_for_best_model="eval_f1",          # Changed to f1 for imbalanced data
                label_names = ['labels'],
                )

    # Define callbacks: EarlyStopping and ModelCheckpoint
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),  # Stop if no improvement in 3 evals
    ]

    if class_weights is not None:
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            peft_config=peft_config,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            callbacks=callbacks,
            class_weights=class_weights,
        )
    else:
        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                peft_config=peft_config,
                # dataset_text_field="text",
                tokenizer=tokenizer,
                # packing=False,
                # max_seq_length=max_seq_length,
                data_collator = collate_fn,
                # dataset_kwargs={
                #     "add_special_tokens": False,
                #     "append_concat_token": False,
                # }
                callbacks=callbacks,
            )
    
    model.config.use_cache = False
    do_train = True

    # Launch training and log metrics
    print("Training...")
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    
    return trainer


def evaluate(test_ds, model_dir, eval_batch_size, labels, label2id, id2label, ):
    # Load model and tokenizer with proper configuration
    print("\nLoading model for evaluation...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # This is fine
    tokenizer.padding_side = 'right'
    
    # Load model with explicit number of labels
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        num_labels=len(labels),  # Add this line to match class numbers
        id2label=id2label,
        label2id=label2id
    )

    # IMPORTANT: Make sure the model config knows about the padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # If using a Llama model, you may need to set this attribute explicitly
    if hasattr(model.config, "tokenizer_padding_side"):
        model.config.tokenizer_padding_side = "right"

    # model, tokenizer = merge_model(model_dir, labels, label2id, id2label)
    model.eval()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Rest of the function remains the same...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # model.to(device)  # TODO for large LLM

    # Prepare dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        collate_fn=lambda x: collate_fn(x), #, device=device),  # Use model's device # device=model.device), # TODO for large LLM
        shuffle=False
    )

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Explicitly move all batch elements to the model's device
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits.float(), dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)

    # Convert label indices to names
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_metric = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall =recall_score(all_labels, all_preds, average='weighted')
    precision_recall_fscore_support_results = precision_recall_fscore_support(all_labels, all_preds, average='weighted')


    clf_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_names,
        output_dict=True
    )

    # Save and print results
    results_dir = Path(model_dir)

    # 1. Save classification report as CSV
    report_df = pd.DataFrame(clf_report).transpose().reset_index()
    report_df.columns = ['class'] + list(report_df.columns[1:])
    report_csv_path = results_dir / "classification_report.csv"
    report_df.to_csv(report_csv_path, index=False)
    
    # 2. Save accuracy and metadata in separate CSV
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'f1_metric': [f1_metric],
        'precision': [precision],
        'recall': [recall],
        'precision_recall_fscore_support_results': [precision_recall_fscore_support_results],
        'num_samples': [len(test_ds)],
        'batch_size': [eval_batch_size],
        'model_dir': [str(model_dir)]
    })
    metrics_csv_path = results_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 3. Generate and save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, label_names, results_dir)

    plot_normalized_confusion_matrix(all_labels, all_preds, label_names, results_dir)
    
    # 4. Generate and save ROC curves
    plot_roc_curves(np.array(all_labels), all_probs, label_names, results_dir)
    
    # 5. Generate and save precision-recall curves
    plot_precision_recall_curves(np.array(all_labels), all_probs, label_names, results_dir)


    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nF1 Metric: {f1_metric:.4f}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"\nRecall: {recall:.4f}")
    
    print("\nClassification Report:")
    print(pd.DataFrame(clf_report).transpose())


    # Collect comprehensive prediction details
    detailed_results = []
    for i in range(len(all_preds)):
        detailed_results.append({
            'text': test_ds[i]['text'],  # Original input text
            'true_label': id2label[all_labels[i]],  # True label name
            'true_label_id': int(all_labels[i]),  # True label ID
            'predicted_label': id2label[all_preds[i]],  # Predicted label name
            'predicted_label_id': int(all_preds[i]),  # Predicted label ID
            'prediction_probabilities': all_probs[i].tolist(),  # Probabilities for all classes
            'is_correct': all_labels[i] == all_preds[i]  # Whether prediction was correct
        })

    # Save detailed results as JSON for comprehensive error analysis
    results_dir = Path(model_dir)
    detailed_results_path = results_dir / "detailed_predictions.json"
    with open(detailed_results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)

    # Create a DataFrame for misclassified instances
    misclassified_df = pd.DataFrame([
        result for result in detailed_results if not result['is_correct']
    ])
    misclassified_path = results_dir / "misclassified_instances.csv"
    misclassified_df.to_csv(misclassified_path, index=False)

    # Additional error analysis information
    error_analysis = {
        'total_samples': len(all_preds),
        'correct_predictions': sum(1 for x in detailed_results if x['is_correct']),
        'incorrect_predictions': sum(1 for x in detailed_results if not x['is_correct']),
        'accuracy': accuracy,
        'per_class_error_breakdown': {}
    }

    # Per-class error breakdown - FIX HERE: Use id2label values instead of labels
    for label_id in id2label.keys():
        label_name = id2label[label_id]
        label_instances = [x for x in detailed_results if x['true_label'] == label_name]
        label_errors = [x for x in label_instances if not x['is_correct']]
        
        if label_instances:  # Avoid division by zero
            error_rate = len(label_errors) / len(label_instances)
        else:
            error_rate = 0
            
        error_analysis['per_class_error_breakdown'][label_name] = {
            'total_instances': len(label_instances),
            'total_errors': len(label_errors),
            'error_rate': error_rate,
            'common_misclassifications': {}
        }

        # Find common misclassification patterns
        misclass_counter = {}
        for error in label_errors:
            misclass_key = error['predicted_label']
            misclass_counter[misclass_key] = misclass_counter.get(misclass_key, 0) + 1
        
        error_analysis['per_class_error_breakdown'][label_name]['common_misclassifications'] = misclass_counter


    # Save error analysis JSON
    error_analysis_path = results_dir / "error_analysis.json"
    with open(error_analysis_path, 'w') as f:
        json.dump(error_analysis, f, indent=2, default=str)

    # Print and return existing results
    print("\nError Analysis Saved:")
    print(f"1. Detailed Predictions: {detailed_results_path}")
    print(f"2. Misclassified Instances: {misclassified_path}")
    print(f"3. Comprehensive Error Analysis: {error_analysis_path}")
    
    return {
        "accuracy": accuracy,
        "classification_report": clf_report,
        "label_mapping": id2label
    }

def main():
    ############################################################################
    # basic settings
    # <settings>
    task_name = 'abstract_sentence_classification'
    train_type ='train' 
    val_type = 'val' 
    test_type = 'test' 
    # </settings>
    print('========== Basic settings: ==========')
    print(f'task_name: {task_name}')
    print(f'train_type: {train_type}')
    print(f'val_type: {val_type}')
    print(f'test_type: {test_type}')
    ############################################################################
    # load task data
    task_data_dir = Path('data/Abstract-Sentence')
    data_files = {}
    for i in task_data_dir.iterdir():
        if i.is_file() and i.name.endswith('.csv'):
            data_files[i.stem] = str(i)
    dataset = load_dataset("csv", data_files = data_files, keep_default_na=False)
    # Only keep these columns
    dataset = dataset.remove_columns(['article_id', 'source', 'category', 'sentence_label'])
    labels = sorted(set(label for label in dataset["train"]["label"]))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    train_ds, val_ds, test_ds = dataset[train_type], dataset[val_type], dataset[test_type]
    print('========== 1. Task data loaded: ==========')
    print(f'train_ds: {train_ds}')
    print(f'val_ds: {val_ds}')
    print(f'test_ds: {test_ds}')
    print(f'labels: {labels}')
    print(f'label2id: {label2id}')
    print(f'id2label: {id2label}')

    ############################################################################
    # Calculate class weights for imbalanced dataset
    train_labels = [label2id[label] for label in dataset["train"]["label"]]
    class_weights_dict = compute_class_weights(train_labels)
    print("Class weights:", class_weights_dict)

    ############################################################################
    # load model from path
    # <settings>
    
    # model_path = '/storage/ukp/shared/shared_model_weights/models--llama-3/8B'
    model_path = '/storage/ukp/shared/shared_model_weights/models--llama-3.1/hf/8B-Instruct'
    # model_path = '/storage/ukp/shared/shared_model_weights/models--meta-llama/Llama-3.2-3B'
    
    # </settings>

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True, # Activate 4-bit precision base model loading
        bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
        bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
        bnb_4bit_compute_dtype = torch.bfloat16, # Compute data type for 4-bit base models
    )

    print('Loading model from...', model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                 quantization_config = bnb_config,
                                                 device_map = 'auto', 
                                                 num_labels = len(labels),
                                                 ) 
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = id2label
    model.config.label2id = label2id

    print('========== 2. Model loaded: ==========')
    print(f'model: {model}')
    

    ############################################################################
    # preprocess dataset
    # <settings>
    max_length = 256 # 1024 # TODO Because sentence wont we longer than 512 or even 384
    # </settings>

    train_ds = preprocess_data(train_ds, label2id, tokenizer, max_length=max_length)
    val_ds = preprocess_data(val_ds, label2id, tokenizer, max_length=max_length)
    test_ds = preprocess_data(test_ds, label2id, tokenizer, max_length=max_length)
    print('========== 3. Dataset preprocessed: ==========')
    print('train_ds: ', train_ds)
    print('val_ds: ', val_ds)
    print('test_ds: ', test_ds)


    ############################################################################
    # fine-tune model
    # <settings>
    lora_r = 64 # LoRA rank parameter # 128
    lora_alpha = 64 # Alpha parameter for LoRA scaling # 128
    lora_dropout = 0.1 # Dropout probability for LoRA layers # standard 0.1
    learning_rate = 2e-4 # Learning rate # Standard 2e-4
    per_device_train_batch_size = 32 # Batch size per GPU for training # 4
    train_epochs = 6 # Number of epochs to train # 10
    recreate_dir = True # Create a directory for the model
    eval_batch_size = 8 # 32 TODO (try using 4)

    print('lora_r: ', lora_r)
    print('lora_alpha: ', lora_alpha)
    print('lora_dropout: ', lora_dropout)
    print('learning_rate: ', learning_rate)
    print('per_device_train_batch_size: ', per_device_train_batch_size)
    print('train_epochs: ', train_epochs)
    print('eval_batch_size: ', eval_batch_size)
    print('class_weights: ', class_weights_dict)
    # </settings>

    # create model dir to save the fine-tuned model
    output_dir = create_model_dir(task_name, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, recreate_dir=recreate_dir)
    print('========== 4. Model dir created: ==========')
    print('output_dir: ', output_dir)
    # fine-tune

    fine_tune(model, tokenizer, train_ds = train_ds , val_ds = val_ds,  lora_r = lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout, class_weights=class_weights_dict,
                                   learning_rate = learning_rate, per_device_train_batch_size = per_device_train_batch_size, train_epochs = train_epochs, output_dir = output_dir)
    print('========== 5. Model fine-tuned: ==========')
    print('output_dir: ', output_dir)

    ############################################################################
    # evaluate the fine-tuned model
    print('Evaluating the model...')
    evaluate(test_ds, model_dir=output_dir, eval_batch_size=eval_batch_size, labels=labels, label2id=label2id, id2label=id2label)
    print('========== 6. Model evaluated ==========')
    print('output_dir: ', output_dir)
    print('========== DONE ==========')
    

def main_evaluate_only():
    # Basic settings - these should match your original run
    task_name = 'abstract_sentence_classification'
    train_type = 'train'
    test_type = 'test'

    # Path to your existing results directory

    existing_output_dir = Path("./new_results-abstract-sentence-all-trainbatch-32-256-trainepoch-6-grad_acc-8-target_modules-all-linear-metrics-llama3.1-8B-Instruct-Epoch-6-lora_r_alpha-64/abstract_sentence_classification/8B-Instruct__lora-r64-a64-d0.1_lr0.0002_bs32_ep6_train_test_ml256_None")
    
    # Load task data
    task_data_dir = Path('data/Abstract-Sentence')
    data_files = {}
    for i in task_data_dir.iterdir():
        if i.is_file() and i.name.endswith('.csv'):
            data_files[i.stem] = str(i)
    dataset = load_dataset("csv", data_files=data_files, keep_default_na=False)
    dataset = dataset.remove_columns(['article_id', 'source', 'category', 'sentence_label'])
    # dataset = dataset.rename_column('sentence', 'text')
    labels = sorted(set(label for label in dataset["train"]["label"]))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    test_ds = dataset[test_type]

    # Modify this line in the main function:
    # test_ds = dataset[test_type].select(range(100))  # Only use the first 5 rows
    print('Test dataset loaded:', test_ds)

    # Setup for preprocessing
    # model_path = '/storage/ukp/shared/shared_model_weights/models--llama-2-hf/7B'  # Original model path
    model_path = '/storage/ukp/shared/shared_model_weights/models--llama-3.1/hf/8B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Preprocessing settings
    max_length = 256  # Must match what you used in training

    test_ds = preprocess_data(test_ds, label2id, tokenizer, max_length=max_length)
    print('Test dataset preprocessed')
    print('test_ds: ', test_ds)
    
    # Evaluate existing model
    eval_batch_size=4
    print(f'Evaluating model from: {existing_output_dir}')
    evaluate(
        test_ds, 
        model_dir=existing_output_dir, 
        eval_batch_size=eval_batch_size,
        labels=labels, 
        label2id=label2id, 
        id2label=id2label,
    )
    print('Evaluation completed')


if __name__ == "__main__":
    main()

    # main_evaluate_only()