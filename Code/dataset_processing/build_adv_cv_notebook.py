#!/usr/bin/env python3
"""Generate Proposed_Model_Adversarial_BERT_CV.ipynb"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.split("\n")}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.split("\n")}

cells = []

# ── Cell 1: Title ──
cells.append(md("""# Adversarial BERT — Full Dataset Training with Cross-Validation

This notebook trains the proposed AdversarialBERT model on each of the five
fake-news datasets using 5-fold cross-validation on the full dataset (no subsets).

## Workflow
1. Imports & environment setup.
2. Define dataset paths (original + pre-computed perturbations).
3. Define Dataset classes, model architecture, custom trainer.
4. 85/15 stratified train/test split → 5-fold CV on the 85% training pool.
5. Save the best model (by validation F1) per dataset.
6. Cross-dataset generalisation evaluation (Accuracy, F1, Precision, Recall)."""))

# ── Cell 2: Imports & Setup ──
cells.append(md("# Section 1: Imports & Setup"))
cells.append(code("""import pandas as pd
import numpy as np
import torch
import os
import random
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput

from google.colab import drive
drive.mount('/content/drive')

try:
    import torch_xla as torch_xla_pkg
    import torch_xla.core.xla_model as xm
    if not hasattr(torch, "xla"):
        torch.xla = torch_xla_pkg
    _TORCH_XLA_AVAILABLE = True
except Exception:
    xm = None
    _TORCH_XLA_AVAILABLE = False

# Reproducibility
SEED = random.randint(0, 4294967295)
print(f"Random seed: {SEED}")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)"""))

# ── Cell 3: Dataset Paths ──
cells.append(md("# Section 2: Dataset Paths"))
cells.append(code("""DATASETS = {
    "WELFake": "/content/drive/MyDrive/datasets/WELFake_processed.csv",
    "FakeNewsNet": "/content/drive/MyDrive/datasets/FakeNewsNet_processed.csv",
    "Fake_News_Detection": "/content/drive/MyDrive/datasets/Fake_News_Detection_processed.csv",
    "ISOT": "/content/drive/MyDrive/datasets/ISOT_processed.csv",
    "Fake_News_Classification": "/content/drive/MyDrive/datasets/Fake_News_Classification_processed.csv",
}

PERTURBED_DATASETS = {
    "WELFake": "/content/drive/MyDrive/datasets/perturbed_outputs/WELFake_perturbed.csv",
    "FakeNewsNet": "/content/drive/MyDrive/datasets/perturbed_outputs/FakeNewsNet_perturbed.csv",
    "Fake_News_Detection": "/content/drive/MyDrive/datasets/perturbed_outputs/Fake_News_Detection_perturbed.csv",
    "ISOT": "/content/drive/MyDrive/datasets/perturbed_outputs/ISOT_perturbed.csv",
    "Fake_News_Classification": "/content/drive/MyDrive/datasets/perturbed_outputs/Fake_News_Classification_perturbed.csv",
}"""))

# ── Cell 4: Dataset Classes ──
cells.append(md("# Section 3: Dataset Classes"))
cells.append(code("""class FakeNewsDataset(Dataset):
    \"\"\"Standard dataset for evaluation (no perturbations).\"\"\"
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)


class AdversarialFakeNewsDataset(Dataset):
    \"\"\"Dataset that pairs original texts with their pre-computed perturbations (pre-tokenized).\"\"\"
    def __init__(self, orig_enc, pert_enc, labels):
        self.orig_enc = orig_enc
        self.pert_enc = pert_enc
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.orig_enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['input_ids_pert'] = torch.tensor(self.pert_enc['input_ids'][idx], dtype=torch.long)
        item['attention_mask_pert'] = torch.tensor(self.pert_enc['attention_mask'][idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)"""))

# ── Cell 5: Model & Loss ──
cells.append(md("# Section 4: AdversarialBERT Model & Loss"))
cells.append(code("""class AdversarialBERT(nn.Module):
    def __init__(self, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def gradient_checkpointing_enable(self, **kwargs):
        self.bert.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.bert.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        if token_type_ids is not None:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_emb))
        return logits, cls_emb


def adversarial_loss(logits_orig, logits_pert, labels, cls_orig, cls_pert, lambda_adv=0.5):
    ce = nn.CrossEntropyLoss()
    ce_loss = 0.5 * (ce(logits_orig, labels) + ce(logits_pert, labels))
    # Mathematically equivalent to CosineEmbeddingLoss with target=1, but avoids dynamic tensor instantiation on TPU
    cosine_sim = nn.functional.cosine_similarity(cls_orig, cls_pert, dim=-1)
    adv_loss = (1.0 - cosine_sim).mean()
    return ce_loss + (lambda_adv * adv_loss)"""))

# ── Cell 6: Custom Trainer ──
cells.append(md("# Section 5: Custom Trainer & Callbacks"))
cells.append(code("""class LambdaSchedulerCallback(TrainerCallback):
    def __init__(self, max_lambda=0.5, warmup_ratio=0.1):
        self.max_lambda = max_lambda
        self.warmup_ratio = warmup_ratio
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer') or self.trainer
        if trainer is None:
            return
        if not hasattr(trainer, 'lambda_adv_tensor'):
            return
        total_steps = state.max_steps
        current_step = state.global_step
        if total_steps <= 0:
            return
        warmup_steps = total_steps * self.warmup_ratio
        if warmup_steps <= 0:
            new_lambda = self.max_lambda
        elif current_step < warmup_steps:
            new_lambda = self.max_lambda * (current_step / warmup_steps)
        else:
            new_lambda = self.max_lambda
        
        # Update the tensor in-place on TPU without JIT recompilation
        trainer.lambda_adv_tensor.fill_(new_lambda)


class AdversarialTrainer(Trainer):
    def __init__(self, *args, lambda_adv=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # Store lambda_adv as a device tensor to prevent XLA JIT re-compilations when the float value changes
        self.lambda_adv_tensor = torch.tensor(lambda_adv, device=self.args.device, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        input_ids_pert = inputs.get('input_ids_pert')
        attn_mask_pert = inputs.get('attention_mask_pert')

        combined_input_ids = torch.cat([inputs['input_ids'], input_ids_pert], dim=0)
        combined_attention_mask = torch.cat([inputs['attention_mask'], attn_mask_pert], dim=0)

        combined_token_type_ids = None
        if 'token_type_ids' in inputs:
            combined_token_type_ids = torch.cat([inputs['token_type_ids'], inputs['token_type_ids']], dim=0)

        combined_logits, combined_cls = model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            token_type_ids=combined_token_type_ids
        )

        batch_size = labels.size(0)
        logits_orig, logits_pert = combined_logits[:batch_size], combined_logits[batch_size:]
        cls_orig, cls_pert = combined_cls[:batch_size], combined_cls[batch_size:]

        loss = adversarial_loss(logits_orig, logits_pert, labels, cls_orig, cls_pert, self.lambda_adv_tensor)
        return (loss, (loss, logits_orig)) if return_outputs else loss"""))

# ── Cell 7: Helper Functions ──
cells.append(md("# Section 6: Helper Functions"))
cells.append(code("""def compute_metrics(pred):
    \"\"\"Compute accuracy, precision, recall, F1.\"\"\"
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary',
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def load_device():
    if _TORCH_XLA_AVAILABLE and xm is not None:
        try:
            device = xm.xla_device()
            print(f"✓ Using TPU: {device}")
            return device, True
        except Exception:
            pass
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (Training will be slow!)")
    return device, False


def save_model(model, tokenizer, output_path, use_tpu, device):
    print("\\n" + "=" * 60)
    print("MODEL SAVING")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    if use_tpu:
        model.to("cpu")
        print("Moved model to CPU for saving.")

    torch.save(model.state_dict(), os.path.join(output_path, "adversarial_bert.pt"))
    tokenizer.save_pretrained(output_path)
    print(f"✓ Model saved to {output_path}")

    if use_tpu:
        model.to(device)
        print("Moved model back to TPU.")

    print("\\n" + "=" * 60)
    print("ADVERSARIAL BERT FINE-TUNING COMPLETE! 🎉")
    print("=" * 60)"""))

# ── Cell 8: Cross-Validation Function ──
cells.append(md("# Section 7: Adversarial Cross-Validation"))
cells.append(code("""def cross_validate_adversarial(full_train_texts, full_train_labels,
                               full_train_pert_texts, test_dataset,
                               tokenizer, compute_metrics_fn,
                               use_tpu, device, n_splits=5):
    print("\\n" + "=" * 60)
    print(f"STARTING {n_splits}-FOLD CROSS VALIDATION (Adversarial BERT)")
    print("=" * 60)

    # Pre-tokenize full training set once to save time and RAM
    print("Pre-tokenizing full training pool...")
    full_train_enc = tokenizer(full_train_texts, truncation=True, padding='max_length', max_length=128)
    full_train_pert_enc = tokenizer(full_train_pert_texts, truncation=True, padding='max_length', max_length=128)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(full_train_labels))

    fold_accuracies = []
    fold_f1_scores = []
    fold_test_results = []
    
    # Store only the state dict of the best model on CPU to avoid OOM
    best_f1 = -1.0
    best_model_state = None
    best_fold_idx = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, full_train_labels)):
        print(f"\\n--- FOLD {fold + 1} ---")

        # Slice pre-tokenized inputs for training & validation folds
        train_enc = {k: [v[i] for i in train_idx] for k, v in full_train_enc.items()}
        val_enc = {k: [v[i] for i in val_idx] for k, v in full_train_enc.items()}
        
        train_pert_enc = {k: [v[i] for i in train_idx] for k, v in full_train_pert_enc.items()}
        val_pert_enc = {k: [v[i] for i in val_idx] for k, v in full_train_pert_enc.items()}
        
        train_labels = [full_train_labels[i] for i in train_idx]
        val_labels = [full_train_labels[i] for i in val_idx]

        # Create adversarial datasets using pre-tokenized features
        train_dataset = AdversarialFakeNewsDataset(train_enc, train_pert_enc, train_labels)
        val_dataset = AdversarialFakeNewsDataset(val_enc, val_pert_enc, val_labels)

        # Fresh model each fold
        model = AdversarialBERT()
        if not use_tpu:
            model.to(device)

        training_kwargs = {
            "output_dir": f'./results_adv_fold_{fold+1}',
            "num_train_epochs": 3,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 2,
            "save_strategy": "steps",
            "save_steps": 128,
            "save_total_limit": 1,               # Conserve disk space
            "bf16": use_tpu,
            "gradient_checkpointing": not use_tpu,
            "report_to": "none",
            "optim": "adamw_torch",
            "logging_dir": './logs',
            "logging_steps": 128,
            "metric_for_best_model": "f1",
            "load_best_model_at_end": True,
            "weight_decay": 0.01,
            "remove_unused_columns": False,
            "label_names": ["labels"],
        }
        if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
            training_kwargs["evaluation_strategy"] = "steps"
            training_kwargs["eval_steps"] = 128
        else:
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = 128

        training_args = TrainingArguments(**training_kwargs)

        lambda_cb = LambdaSchedulerCallback(max_lambda=0.5, warmup_ratio=0.1)
        trainer = AdversarialTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                lambda_cb
            ],
            lambda_adv=0.0
        )
        lambda_cb.trainer = trainer

        trainer.train()

        # Validation evaluation
        eval_metrics = trainer.evaluate()
        val_f1 = eval_metrics['eval_f1']
        fold_f1_scores.append(val_f1)
        fold_accuracies.append(eval_metrics['eval_accuracy'])
        print(f"Fold {fold+1} Validation - Accuracy: {eval_metrics['eval_accuracy']:.4f}, F1: {val_f1:.4f}")

        # Save best state dict to CPU
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_fold_idx = fold

        # Test set evaluation using PyTorch native inference to avoid compiling new Trainer graphs
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=32)
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.numpy())
                
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        acc = accuracy_score(all_targets, all_preds)
        test_metrics = {
            'eval_accuracy': acc,
            'eval_f1': f1,
            'eval_precision': precision,
            'eval_recall': recall
        }
        fold_test_results.append(test_metrics)
        print(f"Fold {fold+1} Test     - Accuracy: {test_metrics['eval_accuracy']:.4f}, F1: {test_metrics['eval_f1']:.4f}, Precision: {test_metrics['eval_precision']:.4f}, Recall: {test_metrics['eval_recall']:.4f}")

        # Clean up memory immediately to prevent RAM OOM
        del model, trainer, train_dataset, val_dataset
        import gc
        gc.collect()
        if not use_tpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary ----
    print("\\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Fold':<6} {'Val Acc':<10} {'Val F1':<10} {'Test Acc':<10} {'Test F1':<10} {'Test Prec':<10} {'Test Rec':<10}")
    print("-" * 66)
    for i in range(n_splits):
        t = fold_test_results[i]
        print(f"{i+1:<6} {fold_accuracies[i]:<10.4f} {fold_f1_scores[i]:<10.4f} {t['eval_accuracy']:<10.4f} {t['eval_f1']:<10.4f} {t['eval_precision']:<10.4f} {t['eval_recall']:<10.4f}")

    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    avg_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    print(f"\\nAverage Validation Accuracy: {avg_acc:.4f} (Std Dev: {std_acc:.4f})")
    print(f"Average Validation F1-score: {avg_f1:.4f} (Std Dev: {std_f1:.4f})")

    best_test = fold_test_results[best_fold_idx]
    print(f"\\n★ Best Fold: {best_fold_idx + 1} (Val F1: {fold_f1_scores[best_fold_idx]:.4f})")
    print(f"  Test Results — Accuracy: {best_test['eval_accuracy']:.4f}, F1: {best_test['eval_f1']:.4f}, Precision: {best_test['eval_precision']:.4f}, Recall: {best_test['eval_recall']:.4f}")

    # Re-instantiate the best model on the appropriate device
    best_model = AdversarialBERT()
    best_model.load_state_dict(best_model_state)
    if not use_tpu:
        best_model.to(device)

    return best_model"""))

# ── Cell 9: Cross-Dataset Evaluation ──
cells.append(md("# Section 8: Cross-Dataset Evaluation"))
cells.append(code("""def cross_dataset_evaluation(model, tokenizer, current_dataset_name, all_datasets_paths, compute_metrics_fn):
    \"\"\"Evaluate model on all datasets except the one it was trained on.
    Outputs Accuracy, F1, Precision, and Recall for each dataset.\"\"\"
    device, use_tpu = load_device()

    print("\\n" + "!" * 60)
    print(f"CROSS-DATASET GENERALIZATION: {current_dataset_name}")
    print("!" * 60)

    class ModelWrapper(nn.Module):
        def __init__(self, inner_model):
            super().__init__()
            self.inner_model = inner_model
        def forward(self, input_ids, attention_mask, labels=None, **kwargs):
            outputs = self.inner_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)

    eval_model = ModelWrapper(model)

    results = {}

    for name, path in all_datasets_paths.items():
        if name == current_dataset_name:
            continue

        print(f"\\nTesting on unseen dataset: {name} (Full Dataset)...")
        df = pd.read_csv(path).dropna()

        test_texts = df['combined_text'].tolist()
        test_labels = df['label'].tolist()

        encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
        dataset = FakeNewsDataset(encodings, test_labels)

        eval_trainer = Trainer(
            model=eval_model,
            compute_metrics=compute_metrics_fn,
            args=TrainingArguments(
                output_dir="./temp_eval",
                remove_unused_columns=False,
                label_names=["labels"],
                per_device_eval_batch_size=32,
                report_to="none"
            )
        )

        metrics = eval_trainer.evaluate(eval_dataset=dataset)
        results[name] = metrics

        acc = metrics.get('eval_accuracy', 0)
        f1 = metrics.get('eval_f1', 0)
        prec = metrics.get('eval_precision', 0)
        rec = metrics.get('eval_recall', 0)

        print(f"  -> {name} Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    return results"""))

# ── Cell 10: Main Training Loop ──
cells.append(md("# Section 9: Main Training Loop"))
cells.append(code("""def adversarial_train_loop(dataset_name, output_path, n_splits=5):
    dataset_path = DATASETS[dataset_name]
    perturbed_path = PERTURBED_DATASETS[dataset_name]

    # Data Loading
    print(f"\\nInitialising adversarial experiment on: {dataset_name}")
    df = pd.read_csv(dataset_path).dropna().reset_index(drop=True)
    print(f"✓ Original dataset loaded: {len(df)} rows")
    print(f"Label distribution:\\n{df['label'].value_counts()}")

    # Load pre-computed perturbations
    df_pert = pd.read_csv(perturbed_path).dropna().reset_index(drop=True)
    assert len(df) == len(df_pert), (
        f"Row count mismatch: original={len(df)}, perturbed={len(df_pert)}"
    )
    assert 'combined_text_perturbed' in df_pert.columns, (
        "Perturbed dataset must contain 'combined_text_perturbed' column"
    )
    print(f"✓ Perturbed dataset loaded: {len(df_pert)} rows")

    all_texts = df['combined_text'].tolist()
    all_pert_texts = df_pert['combined_text_perturbed'].tolist()
    all_labels = df['label'].tolist()

    # 85-15 split
    print("\\n" + "=" * 60)
    print("TRAIN / TEST SPLIT (85-15)")
    print("=" * 60)

    indices = list(range(len(df)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=42,
        stratify=all_labels
    )

    train_texts = [all_texts[i] for i in train_idx]
    train_pert_texts = [all_pert_texts[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]

    test_texts = [all_texts[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    print(f"✓ Training pool samples: {len(train_texts)} (85%)")
    print(f"✓ Test set samples:      {len(test_texts)} (15%)")

    # Tokenize test set (standard, no perturbations needed)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("\\nTokenizing held-out test set...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = FakeNewsDataset(test_encodings, test_labels)
    print("✓ Test set tokenized")

    # Cross-Validation
    device, use_tpu = load_device()
    best_model = cross_validate_adversarial(
        train_texts, train_labels, train_pert_texts,
        test_dataset, tokenizer, compute_metrics,
        use_tpu, device, n_splits=n_splits
    )

    # Save best model
    save_model(best_model, tokenizer, output_path, use_tpu, device)

    # Cross-dataset generalisation
    cross_dataset_evaluation(best_model, tokenizer, dataset_name, DATASETS, compute_metrics)"""))

# ── Cells 11-15: Per-dataset execution ──
ds_list = [
    ("WELFake", "WELFake Dataset"),
    ("FakeNewsNet", "FakeNewsNet Dataset"),
    ("Fake_News_Detection", "Fake News Detection Dataset"),
    ("ISOT", "ISOT Dataset"),
    ("Fake_News_Classification", "Fake News Classification Dataset"),
]
for ds_key, ds_title in ds_list:
    cells.append(md(f"# {ds_title}"))
    cells.append(code(f'adversarial_train_loop("{ds_key}", "/content/drive/MyDrive/models/AdversarialBERT_{ds_key}")'))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".",
                   "Proposed_Model_Adversarial_BERT_CV.ipynb")
out = "/Users/jordan/Desktop/Live_Projects/FYP_Documents/Code/dataset_processing/Proposed_Model_Adversarial_BERT_CV.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✓ Wrote {out}")
