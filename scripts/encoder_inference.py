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
    # Loads a transformer model with a multiple-choice scoring head on top to compute logits for candidate options.
    AutoModelForMultipleChoice,
    # Automatically loads the tokenizer vocabulary and subword processing rules for a specified model architecture.
    AutoTokenizer,
    # High-level API that automates model execution, data collating, batching, and inference evaluation loops.
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
wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], name="DeBERTa-V3-Inference")

REPO_ID = os.environ.get("MCQ_ENCODER_REPO", "your-account/mcq-ensemble-encoder")
TEST_PATH = os.environ.get("MCQ_TEST_PATH", "data/test.csv")
MAX_LENGTH = 512
SEED = 42

set_seed(SEED)

df_test = pd.read_csv(TEST_PATH)

id_col = "id"
q_col = "prompt"

def format_records(df):
    records = []
    for _, row in df.iterrows():
        rec = {
            "id": str(row[id_col]),
            "question": str(row[q_col]) if pd.notna(row[q_col]) else "",
            "A": str(row["A"]), "B": str(row["B"]), "C": str(row["C"]), "D": str(row["D"]), "E": str(row["E"])
        }
        records.append(rec)
    return records

test_records = format_records(df_test)
test_ds = Dataset.from_list(test_records)

# AutoTokenizer automatically loads tokenization rules; from_pretrained downloads tokenizer files saved for fold_1 in the Hub repository.
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder="fold_1", token=hf_token, use_fast=True)
# pad_token fills shorter sequences to uniform length in a batch; eos_token marks the end of a sequence. If pad_token is missing, eos_token is assigned as padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Replicates each question text 5 times so it can be paired with each choice option (A through E).
    first_sentences = [[q] * 5 for q in examples["question"]]
    second_sentences = [[examples[opt][i] for opt in ['A', 'B', 'C', 'D', 'E']] for i in range(len(examples["question"]))]
    
    # Flattens nested question and option lists into continuous 1D lists for paired tokenization.
    first_sentences = list(itertools.chain.from_iterable(first_sentences))
    second_sentences = list(itertools.chain.from_iterable(second_sentences))
    
    # Tokenizes paired question-option text sequences, truncating inputs longer than MAX_LENGTH.
    tokenized = tokenizer(first_sentences, second_sentences, truncation=True, max_length=MAX_LENGTH)
    
    # Reshapes flat tokenized lists into groups of 5 choice option arrays per question sample.
    for k, v in tokenized.items():
        tokenized[k] = [v[i : i + 5] for i in range(0, len(v), 5)]
        
    return tokenized

test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=["question", "A", "B", "C", "D", "E"])

# Custom collator class that flattens multiple-choice sequences, performs dynamic batch padding, and reshapes tensors back to 3D dimensions.
class InferenceDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        
        # Flattens (batch_size, num_choices) feature dicts (excluding 'id') into a 1D list so tokenizer.pad can pad sequences dynamically.
        flattened = [[{k: v[i] for k, v in f.items() if k != "id"} for i in range(num_choices)] for f in features]
        flattened = list(itertools.chain.from_iterable(flattened))
        
        # Dynamically pads token sequences to the longest sequence in the batch and converts them into PyTorch tensors.
        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        
        # Reshapes padded tensors back into 3D tensors with shape (batch_size, num_choices, max_sequence_length).
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        return batch

# Initializes a zero array of shape (num_test_samples, 5_options) to accumulate predicted option logits across fold models.
ensemble_logits = np.zeros((len(test_ds), 5))

for fold in range(1, 6):
    # AutoModelForMultipleChoice attaches a scoring head; from_pretrained loads trained weights saved for the current fold in float16 precision.
    model = AutoModelForMultipleChoice.from_pretrained(
        REPO_ID, 
        subfolder=f"fold_{fold}", 
        token=hf_token, 
        # Loads model weights using 16-bit floating point precision to optimize VRAM usage and speed up inference.
        torch_dtype=torch.float16
    )
    # Sets model to evaluation mode, disabling training-specific behaviors like dropout.
    model.eval()
    
    training_args = TrainingArguments(
        # Directory path where temporary evaluation outputs and artifacts are stored.
        output_dir="./temp_inference",
        # Number of test samples processed per batch on each GPU/CPU device during evaluation.
        per_device_eval_batch_size=4,
        # Number of CPU subprocesses used for asynchronous data loading during inference.
        dataloader_num_workers=2,
        # Enables 16-bit floating point mixed precision for faster forward pass computation on the GPU.
        fp16=True,
        # Disables reporting metrics to external logging dashboards during inference.
        report_to="none"
    )
    
    # Trainer handles batch iteration, model forward passes, and data collating during inference.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=InferenceDataCollator(tokenizer),
    )
    
    # Runs forward pass inference on test_ds to extract option logit matrix of shape (num_samples, 5).
    predictions = trainer.predict(test_ds).predictions
    # Adds logit predictions from the current fold model into the running ensemble total.
    ensemble_logits += predictions
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

# Averages logit scores across all 5 cross-validation fold models to complete the ensemble prediction.
ensemble_logits /= 5.0

# Sorts logits in descending order along axis 1 and extracts column indices of the top 3 highest scoring options per question.
top3_indices = np.argsort(-ensemble_logits, axis=1)[:, :3]

idx_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
predicted_labels = []

# Converts option column indices into space-delimited string labels
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
