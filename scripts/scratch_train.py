import gc
import os

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, login
from kaggle_secrets import UserSecretsClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from transformers import (
    # Loads structural configuration parameters (layers, attention heads, dimensions) without downloading weights.
    AutoConfig,
    # Wraps a transformer backbone with a linear classification/regression head on top to score text sequences.
    AutoModelForSequenceClassification,
    # Automatically selects and loads the matching tokenizer vocabulary and rules for a specified model architecture.
    AutoTokenizer,
    # High-level training utility that executes model training, evaluation, logging, and checkpointing.
    Trainer,
    # Defines hyperparameters, hardware optimizations, and execution settings for the Trainer loop.
    TrainingArguments,
    # Sets random seeds across Python, NumPy, and PyTorch to ensure reproducible execution.
    set_seed,
)

# Function that scans a directory for saved checkpoints to allow resuming interrupted training runs.
from transformers.trainer_utils import get_last_checkpoint

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
wandb_key = user_secrets.get_secret("WANDB_API_KEY")

os.environ["HF_TOKEN"] = hf_token
os.environ["WANDB_API_KEY"] = wandb_key
os.environ.setdefault("WANDB_PROJECT", "mcq-ensemble")
os.environ.setdefault("WANDB_ENTITY", "default")
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

BASE_DIR = "./QSolver_Checkpoints"
os.makedirs(BASE_DIR, exist_ok=True)

login(token=hf_token)
wandb.login(key=wandb_key)

REPO_ID = os.environ.get("MCQ_SCRATCH_REPO", "your-account/mcq-ensemble-scratch")
TOKENIZER_NAME = "microsoft/deberta-v3-xsmall"
SEED = 42
MAX_LENGTH = 256

set_seed(SEED)
create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=hf_token)
api = HfApi(token=hf_token)

# AutoTokenizer handles text tokenization; from_pretrained downloads the vocabulary and subword rules for TOKENIZER_NAME.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=hf_token, use_fast=True)
# pad_token fills shorter sequences to uniform batch length; eos_token marks the end of a text sequence. If pad_token is missing, eos_token is used as padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_col(df, candidates):
    return next((c for c in df.columns if c in candidates), None)

TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
df_train = pd.read_csv(TRAIN_PATH)

q_col = get_col(df_train, ['prompt'])
ans_col = get_col(df_train, ['answer'])

label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

df_train["hard_label"] = df_train[ans_col].apply(lambda x: label_map[str(x).strip().upper()])
df_train["unified_question"] = df_train[q_col].astype(str)

cols_to_keep = ["unified_question", "A", "B", "C", "D", "E", "hard_label"]
df_train = df_train[cols_to_keep]

def explode_dataset(df):
    records = []
    for _, row in df.iterrows():
        q = str(row["unified_question"])
        target_idx = row["hard_label"]
        for i, opt in enumerate(['A', 'B', 'C', 'D', 'E']):
            opt_val = str(row[opt])
            text = f"Question: {q}\nOption: {opt_val}"
            label = 1.0 if i == target_idx else 0.0
            records.append({
                "text": text,
                "label": label
            })
    return pd.DataFrame(records)

def preprocess_function(examples):
    # Converts raw text into numerical token IDs and truncates sequences longer than MAX_LENGTH.
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    # Casts labels to float values
    tokenized["labels"] = [float(l) for l in examples["label"]]
    return tokenized

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    # Flatten collapses multidimensional prediction and label arrays into 1D vectors so option outputs can be grouped and reshaped uniformly for evaluation metrics.
    logits = logits.flatten()
    labels = labels.flatten()
    
    num_groups = len(logits) // 5
    if num_groups == 0:
        return {"eval_map@3": 0.0, "eval_accuracy": 0.0, "eval_precision": 0.0, "eval_recall": 0.0, "eval_f1": 0.0}
        
    reshaped_logits = logits[:num_groups * 5].reshape(num_groups, 5)
    reshaped_labels = labels[:num_groups * 5].reshape(num_groups, 5)
    
    preds = np.argsort(-reshaped_logits, axis=1)[:, :3]
    top1_preds = np.argmax(reshaped_logits, axis=1)
    true_labels = np.argmax(reshaped_labels, axis=1)
    
    map3 = 0.0
    for i, pred in enumerate(preds):
        if true_labels[i] in pred:
            rank = np.where(pred == true_labels[i])[0][0] + 1
            map3 += 1.0 / rank
    map3 /= num_groups
    
    acc = accuracy_score(true_labels, top1_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, top1_preds, average='macro', zero_division=0)
    
    return {"eval_map@3": map3, "eval_accuracy": acc, "eval_precision": precision, "eval_recall": recall, "eval_f1": f1}

# Loads architecture structural settings (vocabulary size, embeddings) from the base model without downloading trained weights.
my_config = AutoConfig.from_pretrained(TOKENIZER_NAME)
# Sets the number of transformer encoder layers to 4 (down from 12) to build a lightweight model.
my_config.num_hidden_layers = 4
# Sets the vector dimensionality for token embeddings and hidden states across layers.
my_config.hidden_size = 256
# Sets the number of parallel self-attention heads per layer to 4.
my_config.num_attention_heads = 4
# Sets the internal expansion dimension of the feed-forward network (FFN) layers to 1024.
my_config.intermediate_size = 1024
# Sets the sequence pooler layer output dimension to 256 to match hidden_size and prevent tensor dimension mismatch errors.
my_config.pooler_hidden_size = 256
# Sets the classification head to output 1 score logit per input sequence for binary prediction.
my_config.num_labels = 1
# Sets pad_token_id in the model config so self-attention layers ignore padding tokens during matrix calculations.
my_config.pad_token_id = tokenizer.pad_token_id

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train["hard_label"])):
    try:
        repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model", token=hf_token)
        if any(f.startswith(f"fold_{fold+1}/") for f in repo_files):
            continue
    except Exception:
        pass

    run_name = f"Scratch-Fold-{fold+1}"
    output_dir = f"{BASE_DIR}/scratch_fold_{fold+1}"

    train_fold_raw = df_train.iloc[train_idx].copy()
    val_fold_raw = df_train.iloc[val_idx].copy()

    train_df_exploded = explode_dataset(train_fold_raw)
    val_df_exploded = explode_dataset(val_fold_raw)

    train_ds = Dataset.from_pandas(train_df_exploded)
    val_ds = Dataset.from_pandas(val_df_exploded)

    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)

    # Calling .from_config() instantiates the architecture with RANDOM WEIGHTS from scratch, rather than downloading pre-trained weights via .from_pretrained().
    model = AutoModelForSequenceClassification.from_config(my_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        # Evaluates the model on the validation dataset at the end of every training epoch.
        eval_strategy="epoch",
        # Saves a model checkpoint to disk at the end of every training epoch.
        save_strategy="epoch",
        # Peak learning rate used by the AdamW optimizer to adjust model weights.
        learning_rate=1e-4,
        # Number of samples processed simultaneously per GPU/CPU device during training.
        per_device_train_batch_size=64,
        # Number of samples processed simultaneously per GPU/CPU device during evaluation.
        per_device_eval_batch_size=64,
        # Number of forward/backward steps to accumulate before updating model weights (1 means update every step).
        gradient_accumulation_steps=1,
        # Total number of complete passes through the training dataset.
        num_train_epochs=5,
        # L2 regularization parameter that penalizes large weights to prevent overfitting.
        weight_decay=0.01,
        # Enables 16-bit floating point precision (mixed-precision) to accelerate training and lower GPU memory consumption.
        fp16=True,
        # Number of CPU processes assigned to load data asynchronously during training.
        dataloader_num_workers=2,
        # Adjusts the learning rate over time using a cosine decay curve.
        lr_scheduler_type="cosine",
        # Number of initial steps where the learning rate scales up linearly from 0 to 1e-4 for training stability.
        warmup_steps=100,
        # Evaluation metric monitored to identify the best-performing model checkpoint.
        metric_for_best_model="eval_map@3",
        # Indicates that a higher value on metric_for_best_model represents a better model.
        greater_is_better=True,
        # Integrates training logs and metric curves directly with Weights & Biases.
        report_to=["wandb"],
        # Name identifier assigned to this training run inside the logging dashboard.
        run_name=run_name,
        # Random seed setting to ensure consistent weight initialization and data shuffling across runs.
        seed=SEED,
        # Disables automatic publishing of intermediate checkpoints to Hugging Face Hub directly from the Trainer.
        push_to_hub=False,
        # Retains only the 1 most recent checkpoint on disk, automatically deleting older ones to save space.
        save_total_limit=1,
        # Automatically reloads the model checkpoint that achieved the best metric score at the end of training.
        load_best_model_at_end=True
    )

    wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], name=run_name, reinit=True)

    # Trainer handles the training loop execution, loss computation, backpropagation, and evaluation routines.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Checks output_dir to locate existing checkpoints in order to resume training if interrupted.
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None

    # Executes training from scratch or resumes progress if last_checkpoint is found.
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=f"fold_{fold+1}",
        repo_id=REPO_ID,
        repo_type="model"
    )

    wandb.finish()
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
