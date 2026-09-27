import gc
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, login
from kaggle_secrets import UserSecretsClient
from sklearn.model_selection import StratifiedKFold
from transformers import (
    # Loads a transformer model with a linear head designed to compute logits for multiple-choice options per input.
    AutoModelForMultipleChoice,
    # Automatically selects and loads the matching tokenizer vocabulary and subword rules for a specified model architecture.
    AutoTokenizer,
    # High-level API that automates model execution, backpropagation, evaluation, and checkpointing.
    Trainer,
    # Defines hyperparameters, hardware optimizations, and execution settings for the Trainer loop.
    TrainingArguments,
    # Sets random seeds across Python, NumPy, and PyTorch to ensure reproducible execution.
    set_seed,
)

# Function that scans a directory for saved checkpoints to allow resuming interrupted training runs.
from transformers.trainer_utils import get_last_checkpoint

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb", "sentencepiece", "protobuf", "bitsandbytes"])
import wandb

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
wandb_key = user_secrets.get_secret("WANDB_API_KEY")

os.environ["HF_TOKEN"] = hf_token
os.environ["WANDB_API_KEY"] = wandb_key
os.environ.setdefault("WANDB_PROJECT", "mcq-ensemble")
os.environ.setdefault("WANDB_ENTITY", "default")
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

login(token=hf_token)
wandb.login(key=wandb_key)

MODEL_NAME = "microsoft/deberta-v3-large"
REPO_ID = os.environ.get("MCQ_ENCODER_REPO", "your-account/mcq-ensemble-encoder")
SEED = 42
MAX_LENGTH = 512

set_seed(SEED)
create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=hf_token)
api = HfApi(token=hf_token)

# AutoTokenizer automatically loads tokenization rules; from_pretrained downloads vocabulary files for MODEL_NAME.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, use_fast=True)
# pad_token fills shorter sequences to uniform length in a batch; eos_token marks the end of a sequence. If pad_token is missing, eos_token is assigned as padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_col(df, candidates):
    return next((c for c in df.columns if c in candidates), None)

df_train = pd.read_csv(os.environ.get("MCQ_TRAIN_PATH", "data/train.csv"))
q_col = get_col(df_train, ['prompt'])
ans_col = get_col(df_train, ['answer'])

label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
df_train["hard_label"] = df_train[ans_col].apply(lambda x: label_map[str(x).strip().upper()])
df_train["unified_question"] = df_train[q_col].astype(str)

cols_to_keep = ["unified_question", "A", "B", "C", "D", "E", "hard_label"]
df_train = df_train[cols_to_keep]

def format_records(df):
    records = []
    for _, row in df.iterrows():
        rec = {
            "question": str(row["unified_question"]),
            "label": int(row["hard_label"]),
            "A": str(row["A"]), "B": str(row["B"]), "C": str(row["C"]), "D": str(row["D"]), "E": str(row["E"])
        }
        records.append(rec)
    return records

def preprocess_function(examples):
    # Replicates each question string 5 times so it can be individually paired with choice options A through E.
    first_sentences = [[q] * 5 for q in examples["question"]]
    second_sentences = [[examples[opt][i] for opt in ['A', 'B', 'C', 'D', 'E']] for i in range(len(examples["question"]))]

    # Flattens nested question and option lists into continuous 1D lists for paired tokenization.
    first_sentences = list(itertools.chain.from_iterable(first_sentences))
    second_sentences = list(itertools.chain.from_iterable(second_sentences))

    # Tokenizes paired question-option text sequences, truncating inputs longer than MAX_LENGTH.
    tokenized = tokenizer(first_sentences, second_sentences, truncation=True, max_length=MAX_LENGTH)
    
    # Reshapes flat token lists into groups of 5 choice option inputs per question sample.
    for k, v in tokenized.items():
        tokenized[k] = [v[i : i + 5] for i in range(0, len(v), 5)]

    tokenized["labels"] = examples["label"]
    return tokenized

# Custom collator class that flattens multiple-choice sequences, dynamically pads them, and reshapes tensors back to 3D dimensions.
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Removes ground-truth label integers from feature dictionaries so they are not padded alongside input token tensors.
        labels = [feature.pop("labels") for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flattens (batch_size, num_choices) feature dicts into a single 1D list so tokenizer.pad can perform dynamic padding.
        flattened = [[{k: v[i] for k, v in f.items()} for i in range(num_choices)] for f in features]
        flattened = list(itertools.chain.from_iterable(flattened))

        # Dynamically pads token sequences to the longest sequence in the batch and converts them into PyTorch tensors.
        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        
        # Reshapes padded tensors back into 3D tensors with shape (batch_size, num_choices, max_sequence_length).
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Converts target option indices into a 1D PyTorch tensor of long integers.
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

# Trainer subclass implementing Adversarial Weight Perturbation (AWP) to increase regularization and improve model generalization.
class AWPTrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        # Computes standard forward pass and backpropagation loss gradient.
        loss = super().training_step(model, inputs, *args, **kwargs)

        # Checks if training has completed at least 1 epoch before applying adversarial perturbations to ensure initial model stability.
        if self.state.epoch >= 1.0:
            # Dictionary storing clean unperturbed weight tensors to restore after computing adversarial gradients.
            backup = {}
            # Step size scalar factor (0.001) determining the magnitude of weight movement along adversarial gradient directions.
            adv_lr = 1e-3
            # Maximum relative bound ratio (0.01) limiting adversarial weight perturbation to within 1% of original parameter values.
            adv_eps = 1e-2

            # Iterates through model parameters to locate word embedding layers for targeted perturbation.
            for name, param in model.named_parameters():
                # Filters for active trainable parameters with valid gradients belonging to embedding layers.
                if param.requires_grad and param.grad is not None and "word_embeddings" in name:
                    # Saves a copy of original unperturbed embedding weights.
                    backup[name] = param.data.clone()

                    # Computes L2 norm of parameter gradients to normalize adversarial step size.
                    norm1 = torch.norm(param.grad)
                    # Computes L2 norm of parameter weights to scale perturbation relative to weight magnitude.
                    norm2 = torch.norm(param.data.detach())

                    # Ensures norm values are non-zero and valid before calculating weight perturbations.
                    if norm1 != 0 and not torch.isnan(norm1):
                        # Calculates scaled adversarial step vector moving parameters in direction of maximum loss increase.
                        adv_step = (adv_lr * param.grad / (norm1 + 1e-6) * (norm2 + 1e-6))
                        # Adds calculated adversarial perturbation directly to embedding weights in-place.
                        param.data.add_(adv_step)

                        # Calculates maximum allowed perturbation magnitude threshold for each weight element.
                        grad_eps = adv_eps * torch.abs(backup[name])
                        # Restricts perturbed weights to stay strictly within [backup - grad_eps, backup + grad_eps] boundaries to prevent excessive deviation.
                        param.data = torch.clamp(
                            param.data,
                            min=backup[name] - grad_eps,
                            max=backup[name] + grad_eps
                        )

            # Executes adversarial loss computation if eligible embedding weights were perturbed.
            if backup:
                # Computes forward pass loss on perturbed model weights to derive adversarial gradients.
                adv_loss = self.compute_loss(model, inputs, return_outputs=False, **kwargs)

                # Scales adversarial loss proportionally when gradient accumulation is active across multiple steps.
                if self.args.gradient_accumulation_steps > 1:
                    adv_loss = adv_loss / self.args.gradient_accumulation_steps

                # Backpropagates adversarial loss gradients using Accelerator.
                self.accelerator.backward(adv_loss)

                # Restores clean original embedding weights prior to executing optimizer update step.
                for name, param in model.named_parameters():
                    if name in backup:
                        param.data = backup[name]

                # Clears memory reference to backup dictionary to free VRAM space.
                del backup
                # Flushes unused PyTorch CUDA memory cache.
                torch.cuda.empty_cache()

        return loss

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    
    # Sorts logits in descending order along option choices and extracts top 3 predicted choice indices per question.
    preds = np.argsort(-logits, axis=1)[:, :3]
    map3 = 0.0
    for i, pred in enumerate(preds):
        if labels[i] in pred:
            rank = np.where(pred == labels[i])[0][0] + 1
            map3 += 1.0 / rank
    return {"eval_map@3": map3 / len(preds)}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train["hard_label"])):
    try:
        repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model", token=hf_token)
        if any(f.startswith(f"fold_{fold+1}/") for f in repo_files):
            continue
    except Exception:
        pass

    run_name = f"DeBERTa-V3-Fold-{fold+1}"
    output_dir = f"./deberta_v3_fold_{fold+1}"

    train_fold = df_train.iloc[train_idx].copy()
    val_fold = df_train.iloc[val_idx].copy()

    train_ds = Dataset.from_list(format_records(train_fold))
    val_ds = Dataset.from_list(format_records(val_fold))

    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)

    # Loads pre-trained model weights for multiple-choice classification using 32-bit floating point precision.
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, token=hf_token, torch_dtype=torch.float32)
    # Disables key-value caching mechanism to reduce VRAM consumption and enable gradient checkpointing compatibility.
    model.config.use_cache = False

    # Freezes the bottom 12 encoder layers to preserve pre-trained features, reduce memory footprint, and accelerate fine-tuning.
    for name, param in model.named_parameters():
        if "deberta.encoder.layer" in name:
            try:
                layer_num = int(name.split("deberta.encoder.layer.")[1].split(".")[0])
                if layer_num < 12:
                    param.requires_grad = False
            except Exception:
                pass

    training_args = TrainingArguments(
        # Directory path where model checkpoints, logs, and configurations will be saved on disk.
        output_dir=output_dir,
        # Computes validation metrics at the end of every training epoch.
        eval_strategy="epoch",
        # Saves a model checkpoint to disk at the end of every training epoch.
        save_strategy="epoch",
        # Peak learning rate parameter used by the 8-bit AdamW optimizer.
        learning_rate=8e-6,
        # Number of training samples processed per batch on each GPU device.
        per_device_train_batch_size=1,
        # Number of validation samples processed per batch on each GPU device during evaluation.
        per_device_eval_batch_size=1,
        # Number of forward/backward steps executed before applying an optimizer weight update step (effective batch size = 16).
        gradient_accumulation_steps=16,
        # Total number of complete training passes over the dataset.
        num_train_epochs=4,
        # L2 weight decay regularization factor applied to weights to prevent overfitting.
        weight_decay=0.01,
        # Disables 16-bit float precision, running training in full 32-bit floating point accuracy.
        fp16=False,
        # Recomputes activation maps during backward passes rather than storing them in memory, drastically cutting VRAM usage.
        gradient_checkpointing=True,
        # Uses standard PyTorch checkpointing implementation instead of re-entrant checkpointing for compatibility.
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # Disables CPU worker subprocesses for data loading in the main PyTorch process.
        dataloader_num_workers=0,
        # Adjusts learning rate according to a cosine decay curve following the warmup phase.
        lr_scheduler_type="cosine",
        # Number of initial steps linearly scaling learning rate from 0 to 8e-6 to stabilize early updates.
        warmup_steps=30,
        # Uses 8-bit quantized AdamW optimizer from bitsandbytes to dramatically reduce optimizer state memory footprint.
        optim="adamw_8bit",
        # Evaluation metric key monitored to select and preserve the best-performing model checkpoint.
        metric_for_best_model="eval_map@3",
        # Specifies that higher evaluation scores indicate a superior model.
        greater_is_better=True,
        # Streams training logs, metrics, and loss values to Weights & Biases dashboard.
        report_to=["wandb"],
        # Identifier string for the current fold run displayed in the logging interface.
        run_name=run_name,
        # Sets global random seed across Python, NumPy, and PyTorch for reproducible runs.
        seed=SEED,
        # Disables automatic uploading of intermediate checkpoints to Hugging Face Hub from Trainer directly.
        push_to_hub=False,
        # Retains dataset columns not directly matching model input signature required by CustomDataCollator.
        remove_unused_columns=False,
        # Retains only the single most recent checkpoint on local storage, deleting older ones.
        save_total_limit=1,
        # Reloads model weights corresponding to the highest eval_map@3 score upon completion of training.
        load_best_model_at_end=True
    )

    wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], name=run_name, reinit=True)

    # AWPTrainer handles training execution, adversarial weight perturbation, gradient accumulation, and evaluation loops.
    trainer = AWPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CustomDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Scans output_dir for existing checkpoints to allow resuming interrupted training runs.
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None

    # Executes training loop starting from initial state or resuming from last_checkpoint.
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
