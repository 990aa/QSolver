import gc
import itertools
import os

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from transformers import (
    # Wraps a transformer backbone with a linear classification/regression head on top for scoring sequences.
    AutoModelForSequenceClassification,
    # Automatically selects and loads the matching tokenizer vocabulary and rules for a specified model architecture.
    AutoTokenizer,
    # High-level API that automates model execution, batching, and data collating during inference.
    Trainer,
    # Defines parameters controlling execution hardware, batch sizes, and logging options during execution.
    TrainingArguments,
    # Sets random seeds across Python, NumPy, and PyTorch to ensure reproducible execution.
    set_seed,
)

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
wandb_key = user_secrets.get_secret("WANDB_API_KEY")

os.environ["HF_TOKEN"] = hf_token
os.environ["WANDB_API_KEY"] = wandb_key
os.environ.setdefault("WANDB_PROJECT", "mcq-ensemble")
os.environ.setdefault("WANDB_ENTITY", "default")

login(token=hf_token)
wandb.login(key=wandb_key)
wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], name="Scratch-V3-Inference")

REPO_ID = os.environ.get("MCQ_SCRATCH_REPO", "your-account/mcq-ensemble-scratch")
TEST_PATH = os.environ.get("MCQ_TEST_PATH", "data/test.csv")
MAX_LENGTH = 256
SEED = 42

set_seed(SEED)

df_test = pd.read_csv(TEST_PATH)
cols = df_test.columns.tolist()

id_col = cols[0]
q_col = 'prompt'

def format_records(df):
    records = []
    for _, row in df.iterrows():
        q = str(row[q_col]) if pd.notna(row[q_col]) else ""
        for opt in ['A', 'B', 'C', 'D', 'E']:
            opt_val = str(row[opt])
            text = f"Question: {q}\nOption: {opt_val}"
            records.append({"text": text})
    return records

test_records = format_records(df_test)
test_ds = Dataset.from_list(test_records)

# AutoTokenizer automatically loads tokenization rules; from_pretrained downloads saved tokenizer files from the specified Hub repo subfolder.
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder="fold_1", token=hf_token, use_fast=True)
# pad_token fills shorter sequences to uniform length in a batch; eos_token marks the end of a text sequence. If pad_token is missing, eos_token is assigned as padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Converts input text into token IDs, truncating sequences longer than MAX_LENGTH while leaving dynamic padding for the data collator.
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=["text"])

# Custom collator class that dynamically pads sequence batches to the maximum sequence length within each batch during evaluation.
class InferenceDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        # Pads input features dynamically to the longest sequence in the batch and converts outputs into PyTorch tensors.
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")

ensemble_logits = np.zeros((len(df_test), 5))

for fold in range(1, 6):
    # AutoModelForSequenceClassification loads the model architecture; from_pretrained fetches trained weights saved for this specific cross-validation fold into memory.
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_ID, 
        subfolder=f"fold_{fold}", 
        token=hf_token, 
        # Loads model weights using half-precision floating point (fp16) to reduce GPU VRAM usage during inference.
        torch_dtype=torch.float16,
        # Configures the classification head to output 1 scalar relevance score per question-option sequence pair.
        num_labels=1
    )
    # Sets the model to evaluation mode, disabling training-specific behavior like dropout layer randomness.
    model.eval()
    
    training_args = TrainingArguments(
        # Directory path where temporary evaluation outputs and execution artifacts are written.
        output_dir="./temp_inference",
        # Number of test samples processed per batch on each GPU/CPU device during inference.
        per_device_eval_batch_size=64,
        # Number of CPU subprocesses used for asynchronous data loading during inference.
        dataloader_num_workers=2,
        # Enables 16-bit floating point mixed precision for faster forward pass computation on the GPU.
        fp16=True,
        # Disables sending metrics or logs to external dashboards during the evaluation loop.
        report_to="none"
    )
    
    # Trainer handles batch iteration, model forward passes, and data collating during inference execution.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=InferenceDataCollator(tokenizer),
    )
    
    # Predicts relevance scores for test samples and collapses 2D prediction outputs into a 1D array.
    predictions = trainer.predict(test_ds).predictions.flatten()
    # Reshapes 1D logits into a 2D matrix of shape (num_questions, 5_options) matching the 5 choices per question.
    predictions = predictions.reshape(-1, 5)
    # Accumulates logit predictions from the current fold model into the running ensemble sum.
    ensemble_logits += predictions
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

# Averages logit scores across all 5 cross-validation fold models to complete the ensemble prediction.
ensemble_logits /= 5.0

# Sorts logits in descending order along axis 1 and extracts the column indices of the top 3 highest scoring options per question.
top3_indices = np.argsort(-ensemble_logits, axis=1)[:, :3]

idx_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
predicted_labels = []

# Converts option column indices into space-separated option string labels (e.g., "A C B").
for row in top3_indices:
    pred_str = " ".join([idx_to_label[i] for i in row])
    predicted_labels.append(pred_str)

df_sub = pd.DataFrame({
    id_col: df_test[id_col],
    'Prediction': predicted_labels
})

df_sub.to_csv("submission.csv", index=False)

wandb_table = wandb.Table(columns=["ID", "Question", "A", "B", "C", "D", "E", "Prediction"])
for idx, row in df_test.iterrows():
    wandb_table.add_data(
        row[id_col], 
        row[q_col], 
        row["A"], 
        row["B"], 
        row["C"], 
        row["D"], 
        row["E"], 
        predicted_labels[idx]
    )

wandb.log({"Inference_Predictions": wandb_table})
wandb.finish()
