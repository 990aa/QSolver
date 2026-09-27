import gc
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo, login

try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    HF_TOKEN = us.get_secret("HF_TOKEN")
    WANDB_API_KEY = us.get_secret("WANDB_API_KEY")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment or secrets.")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ.setdefault("WANDB_PROJECT", "mcq-ensemble")
os.environ.setdefault("WANDB_ENTITY", "default")
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ENTITY = os.environ["WANDB_ENTITY"]
os.environ["PYTHONUNBUFFERED"] = "1"

login(token=HF_TOKEN)

if WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = os.environ.get("MCQ_TRAIN_DATASET", "your-account/mcq-ensemble-train")
MODEL_REPO_ID = os.environ.get("MCQ_DECODER_REPO", "your-account/mcq-ensemble-decoder")
BASE_DIR = "./qwen_unsloth_checkpoints"

USE_CONTEXT = True
MAX_LENGTH = 1024 if USE_CONTEXT else 384
SEED = 42

def train_fold_process(fold_idx, gpu_id):
    # Isolates current process execution to assigned GPU device ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Enables memory segment expansion to prevent PyTorch CUDA memory fragmentation errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Sequence-to-sequence collator handling dynamic padding across variable-length sequences.
    import wandb
    from sklearn.model_selection import StratifiedKFold
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    # FastLanguageModel provides memory-efficient 4-bit loading and optimized LoRA adapter integration.
    from unsloth import FastLanguageModel

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_dir = f"{BASE_DIR}/fold_{fold_idx+1}"
    api = HfApi(token=HF_TOKEN)

    already_done = os.path.exists(output_dir) and (
        os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(output_dir, "model.safetensors"))
    )

    if not already_done:
        try:
            repo_files = api.list_repo_files(repo_id=MODEL_REPO_ID, repo_type="model", token=HF_TOKEN)
            if any(f.startswith(f"fold_{fold_idx+1}/") for f in repo_files):
                already_done = True
        except Exception:
            pass

    if already_done:
        return

    hf_dataset = load_dataset(DATASET_NAME, split="train")
    df_all = hf_dataset.to_pandas()

    option_letters = ["A", "B", "C", "D", "E"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    df_all["answer"] = df_all["answer"].astype(str).str.strip().str.upper()
    df_all["label"] = df_all["answer"].map(label_map)

    def format_prompt(row):
        ctx_line = f"Context: {row['context']}\n" if USE_CONTEXT and pd.notna(row["context"]) and row["context"] else ""
        return (
            f"<|im_start|>system\n"
            f"You are a scientific expert. {'Base your answer STRICTLY on the provided Context. ' if USE_CONTEXT else ''}"
            f"Output ONLY the single letter corresponding to the correct option (A, B, C, D, or E).<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{ctx_line}Question: {row['prompt']}\n"
            f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\nE) {row['E']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # Loads base causal language model architecture and tokenizer with 4-bit quantization enabled.
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Name or Hugging Face Hub repository path of base causal model to load.
        model_name=MODEL_NAME,
        # Maximum sequence context window length supported during fine-tuning.
        max_seq_length=MAX_LENGTH,
        # Automatically detects and selects optimal floating-point precision based on GPU hardware.
        dtype=None,
        # Enables 4-bit NormalFloat (NF4) quantization via BitsAndBytes to reduce VRAM consumption.
        load_in_4bit=True,
    )

    def get_answer_token_id(letter):
        probe_prefix = "<|im_start|>assistant\n"
        # Tokenizes prefix template without adding default start or end special tokens.
        ids_prefix = tokenizer(probe_prefix, add_special_tokens=False)["input_ids"]
        # Tokenizes prefix template concatenated with option letter to isolate answer token ID.
        ids_full = tokenizer(probe_prefix + letter, add_special_tokens=False)["input_ids"]
        # Extracts single integer token ID representing option choice letter in assistant response.
        return ids_full[len(ids_prefix):][0]

    option_token_ids = [get_answer_token_id(l) for l in option_letters]

    def preprocess(df, tokenizer):
        rows = df.to_dict("records")
        prompt_strs = [format_prompt(r) for r in rows]
        # Formats target completion strings by appending correct answer choice letter with end-of-turn template token.
        target_strs = [f"{r['answer']}<|im_end|>" for r in rows]
        # Concatenates prompt template text and target answer text into complete sequence strings.
        full_strs = [p + t for p, t in zip(prompt_strs, target_strs)]

        # Tokenizes complete prompt-and-target text into token IDs without inserting extra tokenizer special tokens.
        full_encodings = tokenizer(full_strs, add_special_tokens=False)["input_ids"]
        # Tokenizes prompt template text alone to calculate exact prompt token length for label masking.
        prompt_encodings = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]

        all_input_ids, all_attention_mask, all_labels = [], [], []
        for full_ids, prompt_ids in zip(full_encodings, prompt_encodings):
            prompt_len = len(prompt_ids)
            if full_ids[:prompt_len] != prompt_ids:
                common = 0
                for a, b in zip(full_ids, prompt_ids):
                    if a != b:
                        break
                    common += 1
                prompt_len = common

            # Truncates input sequence from left side if total length exceeds MAX_LENGTH.
            input_ids = full_ids[-MAX_LENGTH:] if len(full_ids) > MAX_LENGTH else full_ids
            # Assigns -100 label ID to prompt tokens so cross-entropy loss is computed exclusively on target response tokens.
            labels = ([-100] * prompt_len + full_ids[prompt_len:])[-MAX_LENGTH:]
            # Attention mask is a binary tensor (1s for valid content tokens, 0s for pad tokens) telling self-attention layers which tokens to process; assigning 1s marks all current tokens as valid content prior to dynamic batch padding.
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        })

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(df_all, df_all["label"]))
    train_idx, val_idx = splits[fold_idx]

    train_fold = df_all.iloc[train_idx].copy()
    val_fold = df_all.iloc[val_idx].copy()

    train_ds = preprocess(train_fold, tokenizer)
    val_ds = preprocess(val_fold, tokenizer)

    def compute_map3(logits, labels):
        preds, targets = [], []
        for i in range(len(labels)):
            label_row = np.array(labels[i])
            # Locates first token index where target label is not masked out by -100.
            pos = np.where(label_row != -100)[0]
            if len(pos) == 0:
                continue
            # Extracts option logit scores generated immediately prior to target answer token.
            preds.append(logits[i, pos[0] - 1])
            slot = np.where(np.array(option_token_ids) == label_row[pos[0]])[0]
            targets.append(int(slot[0]) if len(slot) else 0)
        if not preds:
            return 0.0
        preds = np.array(preds)
        # Sorts predicted option logits in descending order and extracts top 3 choice indices per sample.
        top3 = np.argsort(-preds, axis=1)[:, :3]
        scores = [1.0 if t == p[0] else (0.5 if t == p[1] else (1/3 if t == p[2] else 0.0)) for p, t in zip(top3, targets)]
        return float(np.mean(scores))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        # Slices vocab dimension to keep only logits corresponding to option choice tokens ('A','B','C','D','E'), preventing GPU memory exhaustion.
        return logits[:, :, option_token_ids]

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        return {"map@3": compute_map3(logits, labels)}

    # Wraps base quantized language model with Low-Rank Adaptation (LoRA) trainable adapter parameters.
    model = FastLanguageModel.get_peft_model(
        model,
        # Sets rank dimensionality for LoRA adapter matrices.
        r=32,
        # Specifies target attention and feed-forward projection layers where LoRA adapters are attached.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Scaling factor that scales adapter parameter update weights.
        lora_alpha=64,
        # Dropout probability applied to LoRA adapter layers for regularization.
        lora_dropout=0.05,
        # Disables trainable bias parameters inside LoRA adapter modules.
        bias="none",
        # Enables Unsloth optimized gradient checkpointing to recompute activations during backpropagation, saving VRAM.
        use_gradient_checkpointing="unsloth",
        # Sets random seed for deterministic initialization of adapter layer weights.
        random_state=SEED,
    )
    # Prepares PEFT model for training mode by enabling gradient tracking and forward pass hooks.
    model = FastLanguageModel.for_training(model)

    # DataCollatorForSeq2Seq handles dynamic batch sequence padding and tensor conversion.
    collator = DataCollatorForSeq2Seq(
        # Tokenizer instance used to pad sequences to uniform length.
        tokenizer=tokenizer,
        # Pads sequence lengths to multiples of 8 for optimal Tensor Core hardware utilization.
        pad_to_multiple_of=8,
        # Formats output collated feature tensors as PyTorch tensor objects.
        return_tensors="pt",
        # Dynamically pads sequences to maximum length within current batch.
        padding=True
    )
    run_name = f"qwen_unsloth_fold_{fold_idx+1}"

    training_args = TrainingArguments(
        # Directory path where model adapters and training checkpoints are written.
        output_dir=output_dir,
        # Computes evaluation metrics on validation dataset at the end of each training epoch.
        eval_strategy="epoch",
        # Saves model adapter checkpoint to disk at the end of each training epoch.
        save_strategy="epoch",
        # Logs training loss and progress metrics every 10 step iterations.
        logging_steps=10,
        # Peak learning rate parameter used by AdamW optimizer.
        learning_rate=1e-4,
        # Training batch size processed per GPU device.
        per_device_train_batch_size=4,
        # Evaluation batch size processed per GPU device.
        per_device_eval_batch_size=4,
        # Accumulates evaluation predictions on host memory every 10 steps to prevent GPU memory saturation.
        eval_accumulation_steps=10,
        # Accumulates gradients across 4 steps before executing optimizer update step (effective batch size = 16).
        gradient_accumulation_steps=4,
        # Total complete passes through training dataset.
        num_train_epochs=4,
        # L2 weight regularization factor applied to model parameters to prevent overfitting.
        weight_decay=0.01,
        # Fraction of total training steps allocated to linear learning rate warmup.
        warmup_ratio=0.05,
        # Enables 16-bit brain floating point precision if supported by GPU hardware.
        bf16=torch.cuda.is_bf16_supported(),
        # Enables standard 16-bit floating point precision if bfloat16 is unsupported.
        fp16=not torch.cuda.is_bf16_supported(),
        # Reloads model adapter weights corresponding to best validation score at training completion.
        load_best_model_at_end=True,
        # Metric key monitored to select top-performing checkpoint.
        metric_for_best_model="map@3",
        # Indicates higher score on monitored metric represents superior model quality.
        greater_is_better=True,
        # Disables automatic uploading of intermediate checkpoints to Hugging Face Hub from Trainer directly.
        push_to_hub=False,
        # Directs training metrics to Weights & Biases dashboard or disables logging.
        report_to="wandb" if WANDB_API_KEY else "none",
        # Identifier string for current fold run displayed in experiment tracker dashboard.
        run_name=run_name,
        # Keeps only 1 most recent checkpoint on local storage, automatically deleting older ones.
        save_total_limit=1,
        # Disables average token count normalization across GPU devices in multi-GPU execution.
        average_tokens_across_devices=False,
        # Number of CPU subprocesses allocated for dataset loading.
        dataloader_num_workers=2,
        # Pins host memory tensors to accelerate CPU-to-GPU memory transfer speeds.
        dataloader_pin_memory=True,
    )

    # Trainer handles fine-tuning loops, backpropagation, metric evaluation, and model updates.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if WANDB_API_KEY:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            reinit=True
        )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    api.upload_folder(folder_path=output_dir, path_in_repo=f"fold_{fold_idx+1}", repo_id=MODEL_REPO_ID, repo_type="model")

    if WANDB_API_KEY:
        wandb.finish()

# Sets multiprocessing process start method to spawn for clean CUDA context creation in child processes.
mp.set_start_method("spawn", force=True)
num_gpus = torch.cuda.device_count()
folds_to_run = list(range(5))

if num_gpus >= 2:
    fold_queue = list(folds_to_run)
    active_processes = {}

    for gpu in range(min(num_gpus, 2)):
        if fold_queue:
            f = fold_queue.pop(0)
            # Spawns parallel child process executing fold training on assigned GPU device.
            p = mp.Process(target=train_fold_process, args=(f, gpu))
            p.start()
            active_processes[gpu] = (p, f)

    while active_processes:
        time.sleep(5)
        finished_gpus = []
        for gpu, (p, f) in active_processes.items():
            if not p.is_alive():
                p.join()
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del active_processes[gpu]
            if fold_queue:
                next_f = fold_queue.pop(0)
                p = mp.Process(target=train_fold_process, args=(next_f, gpu))
                p.start()
                active_processes[gpu] = (p, next_f)
else:
    for fold in folds_to_run:
        train_fold_process(fold, 0)
