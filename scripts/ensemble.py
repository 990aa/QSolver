import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", 
    "unsloth", "wandb", "transformers", "sentence-transformers", "huggingface_hub", "datasets", "sentencepiece", "protobuf"
])

import gc
import glob
import itertools
import os
import time

import numpy as np
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, login
from peft import PeftModel
from scipy.special import softmax
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from unsloth import FastLanguageModel

# Define global random seed value to enforce deterministic reproducibility across execution runs
SEED = 42

# Apply the global seed to PyTorch, NumPy, and standard library random number generators
set_seed(SEED)

try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    HF_TOKEN = us.get_secret("HF_TOKEN")
    WANDB_API_KEY = us.get_secret("WANDB_API_KEY")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    login(token=HF_TOKEN)

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ.setdefault("WANDB_PROJECT", "mcq-ensemble")
    os.environ.setdefault("WANDB_ENTITY", "default")
    import wandb
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name="TriModel-OptimizedEnsemble-Sequential",
        reinit=True
    )

MODEL_NAMESPACE = os.environ.get("MCQ_MODEL_NAMESPACE", "your-account")
MODEL1_RAG_QWEN3_REPO = os.environ.get("MCQ_DECODER_REPO", f"{MODEL_NAMESPACE}/mcq-ensemble-decoder")
MODEL2_DEBERTA_REPO = os.environ.get("MCQ_ENCODER_REPO", f"{MODEL_NAMESPACE}/mcq-ensemble-encoder")
MODEL3_SCRATCH_REPO = os.environ.get("MCQ_SCRATCH_REPO", f"{MODEL_NAMESPACE}/mcq-ensemble-scratch")
CACHE_REPOS = [
    os.environ.get("MCQ_CONTEXT_REPO", "your-account/mcq-ensemble-context")
]
TEST_PATH = os.environ.get("MCQ_TEST_PATH", "data/test.csv")
KB_GLOB = os.environ.get("MCQ_KNOWLEDGE_BASE_GLOB", "data/knowledge_base/*.arrow")
BASE_DIR  = "./qwen_unsloth_checkpoints"
os.makedirs(BASE_DIR, exist_ok=True)

df_test = pd.read_csv(TEST_PATH)
cols = df_test.columns.tolist()

id_col = "id"
q_col = "prompt"

# Record total number of test samples to size predictions arrays
NUM_SAMPLES = len(df_test)

# Define function to search and load pre-computed retrieval contexts from Hugging Face Hub
def load_cached_test_context(df, repos):
    # Iterate through target cache repository identifiers
    for repo in repos:
        try:
            # Load training split of cached dataset from Hugging Face Hub
            ds = load_dataset(repo, split="train")
            
            # Convert loaded dataset structure to pandas DataFrame
            df_cached = ds.to_pandas()
            
            # Identify column containing text context
            ctx_col = next((c for c in ["context", "retrieved_context", "text_context"] if c in df_cached.columns), None)
            
            # Identify column containing matching prompt text
            p_col = next((c for c in ["prompt", "question", "text", "stem"] if c in df_cached.columns), None)
            
            # Verify both context and prompt columns exist in cached DataFrame
            if ctx_col and p_col:
                # Construct dictionary mapping cleaned prompt text to retrieved text context
                prompt_map = {str(p).strip(): str(c) for p, c in zip(df_cached[p_col], df_cached[ctx_col]) if pd.notna(p)}
                
                # Normalize prompt strings from incoming test DataFrame
                test_prompts_norm = df["prompt"].astype(str).str.strip()
                
                # Map prompt strings to retrieve corresponding context strings
                matched_contexts = test_prompts_norm.map(prompt_map)
                
                # Count total number of successfully matched non-null context strings
                valid_count = matched_contexts.notna().sum()
                
                # Ensure match rate exceeds 90 percent of dataset length before accepting cache
                if valid_count >= 0.90 * len(df):
                    return matched_contexts.fillna("").tolist(), repo
        except Exception:
            pass
            
    # Return None if no suitable cached repository passed validation
    return None, None

# Attempt loading cached context strings for the test set
cached_contexts, matched_repo = load_cached_test_context(df_test, CACHE_REPOS)

# Assign cached contexts if successfully retrieved, otherwise compute contexts dynamically
if cached_contexts is not None:
    df_test["context"] = cached_contexts
else:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Locate knowledge base Arrow files matching the wildcard pattern
    arrow_files = glob.glob(KB_GLOB)
    
    # Handle scenario where knowledge base files are missing
    if not arrow_files:
        df_test["context"] = ""
    else:
        # Load local Arrow knowledge base dataset files
        kb_dataset = load_dataset("arrow", data_files=arrow_files, split="train")
        
        # Extract passage text list from knowledge base dataset
        kb_texts = kb_dataset["text"]

        # Instantiate TF-IDF vectorizer with sublinear term frequency scaling and 1.5M max features
        vectorizer = TfidfVectorizer(
            stop_words="english", 
            sublinear_tf=True, 
            max_features=1_500_000, 
            dtype=np.float32
        )
        
        # Fit vectorizer and transform knowledge base passages into sparse term-document matrix
        kb_matrix = vectorizer.fit_transform(kb_texts)

        # Select CUDA GPU device if available, otherwise default to CPU execution
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load dense sentence transformer embedding model for passage reranking
        reranker = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

        # Define hybrid RAG pipeline combining sparse TF-IDF filtering and dense vector reranking
        def hybrid_retrieve_contexts(df, top_k_sparse=15, top_k_final=3, batch_size=256, max_words=220):
            final_contexts = []
            
            # Construct weighted search queries prioritizing prompt text over candidate option choices
            weighted_queries = (
                df["prompt"] + " " + df["prompt"] + " " + df["prompt"] + " " +
                df["A"].astype(str) + " " + df["B"].astype(str) + " " +
                df["C"].astype(str) + " " + df["D"].astype(str) + " " + df["E"].astype(str)
            ).fillna("").tolist()
            
            # Construct unweighted queries for dense embedding calculation
            clean_queries = (df["prompt"] + " " + df["A"].astype(str) + " " + df["B"].astype(str)).fillna("").tolist()

            # Iterate through test queries in memory-managed mini-batches
            for i in range(0, len(weighted_queries), batch_size):
                bwq = weighted_queries[i:i + batch_size]
                bcq = clean_queries[i:i + batch_size]
                
                # Compute sparse TF-IDF feature matrix for query batch
                q_matrix = vectorizer.transform(bwq)
                
                # Calculate sparse dot-product similarity scores against knowledge base passages
                scores = q_matrix.dot(kb_matrix.T)
                
                # Compute dense vector embeddings for query batch using sentence transformer
                batch_query_embeds = reranker.encode(bcq, normalize_embeddings=True, show_progress_bar=False)

                # Process each individual query within the current batch
                for j in range(scores.shape[0]):
                    row = scores.getrow(j)
                    if len(row.data) == 0:
                        final_contexts.append("")
                        continue
                    
                    # Extract top K candidate passage indices based on sparse similarity scores
                    top_15_idx = np.argsort(-row.data)[:top_k_sparse]
                    candidate_kb_indices = row.indices[top_15_idx]
                    candidate_passages = [kb_texts[int(idx)] for idx in candidate_kb_indices]

                    # Compute dense embeddings for the retrieved candidate passages
                    passage_embeds = reranker.encode(candidate_passages, normalize_embeddings=True, show_progress_bar=False)
                    q_embed = batch_query_embeds[j]
                    
                    # Calculate dense cosine similarity scores between query and passage vectors
                    dense_scores = np.dot(passage_embeds, q_embed)
                    
                    # Select top final passage indices sorted by dense similarity scores
                    top_final_idx = np.argsort(-dense_scores)[:top_k_final]

                    # Truncate and concatenate passages into a single context string per query
                    best_passages = [" ".join(str(candidate_passages[idx]).split()[:max_words]) for idx in top_final_idx]
                    final_contexts.append(" ".join(best_passages))

                # Periodically trigger garbage collection to prevent host RAM inflation
                if (i + batch_size) % 2000 == 0 or (i + batch_size) >= len(weighted_queries):
                    gc.collect()

            return final_contexts

        # Execute hybrid retrieval to populate context column in test DataFrame
        df_test["context"] = hybrid_retrieve_contexts(df_test)
        
        # Release knowledge base memory allocations and clear PyTorch VRAM cache
        del kb_matrix, kb_texts, kb_dataset, reranker, vectorizer
        gc.collect()
        torch.cuda.empty_cache()

        try:
            # Save generated contexts to local disk as fallback dataset artifact
            cols_to_save = [c for c in [id_col, "prompt", "A", "B", "C", "D", "E", "context"] if c in df_test.columns]
            ds_to_push = Dataset.from_pandas(df_test[cols_to_save].reset_index(drop=True))
            DatasetDict({"train": ds_to_push}).push_to_hub(os.environ.get("MCQ_CONTEXT_REPO", "your-account/mcq-ensemble-context"))
        except Exception:
            pass

# Save test DataFrame containing context strings to local disk CSV file
df_test.to_csv("./test_with_context.csv", index=False)

# Define base model architecture identifier for Model 1 (Qwen3-4B-Instruct)
BASE_MODEL_NAME_QWEN = "Qwen/Qwen3-4B-Instruct-2507"

# Define maximum input token sequence length for Model 1 decoder tokenization
MAX_LENGTH_QWEN = 1280

# Define mini-batch size per forward pass step during Model 1 inference
INFERENCE_BATCH_SIZE_QWEN = 8

# List standard multiple choice answer option letter labels
option_letters = ["A", "B", "C", "D", "E"]

# Define template function formatting questions and retrieved contexts into ChatML prompt strings
def format_qwen3_prompt(row):
    ctx_line = f"Context: {row['context']}\n" if pd.notna(row["context"]) and str(row["context"]).strip() else ""
    return (
        f"<|im_start|>system\n"
        f"You are a scientific expert. Base your answer STRICTLY on the provided Context. "
        f"Output ONLY the single letter corresponding to the correct option (A, B, C, D, or E).<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{ctx_line}Question: {row['prompt']}\n"
        f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\nE) {row['E']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# Format all test DataFrame records into complete Qwen3 ChatML string prompts
qwen_prompts = [format_qwen3_prompt(row) for _, row in df_test.iterrows()]

# Initialize array to accumulate ensembled option probability distributions for Model 1 across 5 folds
m1_ensemble_probs = np.zeros((NUM_SAMPLES, 5))

# Sequentially process each of the 5 cross-validation folds for Model 1
for fold_idx in range(1, 6):
    # Load 4-bit quantized Qwen3 base language model using Unsloth FastLanguageModel
    model_m1, tokenizer_m1 = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME_QWEN,
        max_seq_length=MAX_LENGTH_QWEN,
        dtype=None,
        load_in_4bit=True
    )
    
    # Configure left-side truncation and padding required for causal decoder batch generation
    tokenizer_m1.truncation_side = "left"
    tokenizer_m1.padding_side = "left"
    if tokenizer_m1.pad_token is None:
        tokenizer_m1.pad_token = tokenizer_m1.eos_token

    # Helper function extracting vocabulary token ID corresponding to an answer option letter
    def get_answer_token_id(letter, tokenizer=tokenizer_m1):
        probe = "<|im_start|>assistant\n"
        return tokenizer_m1(probe + letter, add_special_tokens=False)["input_ids"][len(tokenizer_m1(probe, add_special_tokens=False)["input_ids"]):][0]

    # Map option letters A-E to their corresponding vocabulary token IDs
    option_token_ids = [get_answer_token_id(l) for l in option_letters]

    # Load fine-tuned LoRA adapter weights corresponding to current fold index
    model_m1 = PeftModel.from_pretrained(model_m1, MODEL1_RAG_QWEN3_REPO, subfolder=f"fold_{fold_idx}")
    
    # Enable optimized Unsloth fast inference mode
    FastLanguageModel.for_inference(model_m1)

    # Initialize array storing option probabilities for the current fold
    fold_probs = np.zeros((NUM_SAMPLES, 5))

    # Disable gradient tracking to accelerate inference and conserve VRAM
    with torch.inference_mode():
        # Iterate over prompts in mini-batches
        for i in range(0, len(qwen_prompts), INFERENCE_BATCH_SIZE_QWEN):
            batch_prompts = qwen_prompts[i:i + INFERENCE_BATCH_SIZE_QWEN]
            
            # Tokenize batch prompts into input tensor arrays on target GPU device
            inputs = tokenizer_m1(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=MAX_LENGTH_QWEN
            ).to(model_m1.device)
            
            # Forward pass keeping only the final output token logits
            out = model_m1(**inputs, logits_to_keep=1, use_cache=False)
            
            # Extract output logits corresponding specifically to option letter token IDs
            batch_logits = out.logits[:, -1, option_token_ids].float().cpu().numpy()
            
            # Convert raw option logits to calibrated probability distributions via row-wise softmax
            fold_probs[i:i + len(batch_prompts)] = softmax(batch_logits, axis=1)

    # Accumulate 1/5th weight contribution of current fold probabilities into Model 1 ensemble array
    m1_ensemble_probs += fold_probs / 5.0

    # Delete fold model and tokenizer objects from host memory
    del model_m1, tokenizer_m1
    gc.collect()
    torch.cuda.empty_cache()

# Format test records into dictionary list for DeBERTa Multiple Choice tokenization
test_records_deberta = []
for _, row in df_test.iterrows():
    test_records_deberta.append({
        "id": str(row[id_col]),
        "question": str(row["prompt"]) if pd.notna(row["prompt"]) else "",
        "A": str(row["A"]), "B": str(row["B"]), "C": str(row["C"]), "D": str(row["D"]), "E": str(row["E"])
    })

# Construct Hugging Face Dataset from formatted DeBERTa record dictionaries
deberta_ds = Dataset.from_list(test_records_deberta)

# Load fast tokenizer for DeBERTa-v3 Multiple Choice Encoder from repository fold 1
deberta_tokenizer = AutoTokenizer.from_pretrained(
    MODEL2_DEBERTA_REPO, 
    subfolder="fold_1", 
    token=HF_TOKEN, 
    use_fast=True
)

# Ensure pad token is assigned if uninitialized in pre-trained tokenizer
if deberta_tokenizer.pad_token is None:
    deberta_tokenizer.pad_token = deberta_tokenizer.eos_token

# Define preprocessing function tokenizing question stems paired with each candidate answer option
def preprocess_deberta(examples):
    first_sentences = [[q] * 5 for q in examples["question"]]
    second_sentences = [[examples[opt][i] for opt in ['A', 'B', 'C', 'D', 'E']] for i in range(len(examples["question"]))]
    
    # Flatten sentence pair lists for batch tokenization
    first_sentences = list(itertools.chain.from_iterable(first_sentences))
    second_sentences = list(itertools.chain.from_iterable(second_sentences))
    
    # Tokenize sentence pairs truncating to 512 max length
    tokenized = deberta_tokenizer(first_sentences, second_sentences, truncation=True, max_length=512)
    
    # Reshape tokenized outputs into nested lists grouped by 5 options per question sample
    for k, v in tokenized.items():
        tokenized[k] = [v[i : i + 5] for i in range(0, len(v), 5)]
        
    return tokenized

# Apply preprocessing function across entire DeBERTa dataset in batched mode
deberta_ds = deberta_ds.map(preprocess_deberta, batched=True, remove_columns=["question", "A", "B", "C", "D", "E"])

# Custom PyTorch Data Collator padding 5-choice input feature tensors dynamically
class DebertaDataCollator:
    def __init__(self, tokenizer): 
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        
        # Unroll dynamic nested features into flat dictionary list
        flattened = [[{k: v[i] for k, v in f.items() if k != "id"} for i in range(num_choices)] for f in features]
        flattened = list(itertools.chain.from_iterable(flattened))
        
        # Pad batch feature tensors to maximum sequence length within batch
        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        
        # Reshape padded tensor dimensions back to [batch_size, num_choices, max_seq_len]
        return {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

# Instantiate PyTorch DataLoader for DeBERTa inference with batch size of 8
deberta_collator = DebertaDataCollator(deberta_tokenizer)
deberta_loader = DataLoader(deberta_ds, batch_size=8, collate_fn=deberta_collator)

# Initialize array to accumulate ensembled option probability distributions for Model 2 across 5 folds
m2_ensemble_probs = np.zeros((NUM_SAMPLES, 5))

# Sequentially process each of the 5 cross-validation folds for Model 2 (DeBERTa-v3)
for fold in range(1, 6):
    # Load fine-tuned DeBERTa AutoModelForMultipleChoice model in float16 precision onto GPU
    model_deberta = AutoModelForMultipleChoice.from_pretrained(
        MODEL2_DEBERTA_REPO, 
        subfolder=f"fold_{fold}", 
        token=HF_TOKEN, 
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model to evaluation mode disabling dropout layers
    model_deberta.eval()

    fold_logits_deberta = []

    # Execute inference loop using PyTorch DataLoader avoiding unsloth trainer override issues
    with torch.inference_mode():
        for batch in deberta_loader:
            # Transfer batch input tensors to target GPU execution device
            batch = {k: v.to(model_deberta.device) for k, v in batch.items()}
            
            # Forward pass extracting raw multiple choice logits [batch_size, 5]
            out = model_deberta(**batch)
            
            # Append batch output logits array to fold predictions list
            fold_logits_deberta.append(out.logits.cpu().numpy())

    # Concatenate mini-batch logit arrays into single matrix of shape [NUM_SAMPLES, 5]
    raw_logits_deberta = np.concatenate(fold_logits_deberta, axis=0)
    
    # Convert raw logits to probabilities via softmax and accumulate 1/5th weight into ensemble array
    m2_ensemble_probs += softmax(raw_logits_deberta, axis=1) / 5.0

    # Clean up model references and release VRAM memory
    del model_deberta
    gc.collect()
    torch.cuda.empty_cache()

# Format test records into individual question-option text pair strings for Model 3 pointwise classification
def format_scratch_records(df):
    records = []
    for _, row in df.iterrows():
        q = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        for opt in ['A', 'B', 'C', 'D', 'E']:
            opt_val = str(row[opt])
            text = f"Question: {q}\nOption: {opt_val}"
            records.append({"text": text})
    return records

# Construct records list containing 5 separate text items per question sample
scratch_records = format_scratch_records(df_test)

# Build Hugging Face Dataset from pointwise formatted sequence classification records
scratch_ds = Dataset.from_list(scratch_records)

# Load fast tokenizer for Model 3 (QSolver_Scratch) from repository fold 1
scratch_tokenizer = AutoTokenizer.from_pretrained(
    MODEL3_SCRATCH_REPO, 
    subfolder="fold_1", 
    token=HF_TOKEN, 
    use_fast=True
)

# Ensure pad token is assigned if uninitialized in tokenizer
if scratch_tokenizer.pad_token is None:
    scratch_tokenizer.pad_token = scratch_tokenizer.eos_token

# Preprocessing function tokenizing individual text sequence pairs with 256 max length
def preprocess_scratch(examples):
    return scratch_tokenizer(examples["text"], truncation=True, max_length=256)

# Map tokenization function across entire Scratch dataset
scratch_ds = scratch_ds.map(preprocess_scratch, batched=True, remove_columns=["text"])

# Custom PyTorch Data Collator dynamically padding Scratch sequence classification features
class ScratchInferenceDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")

# Instantiate PyTorch DataLoader for Model 3 inference with batch size of 64
scratch_collator = ScratchInferenceDataCollator(scratch_tokenizer)
scratch_loader = DataLoader(scratch_ds, batch_size=64, collate_fn=scratch_collator)

# Initialize matrix accumulating ensembled option logit scores for Model 3 across 5 folds
m3_ensemble_logits = np.zeros((NUM_SAMPLES, 5))

# Sequentially process each of the 5 cross-validation folds for Model 3 (QSolver_Scratch)
for fold in range(1, 6):
    # Load fine-tuned sequence classification model configured for single score regression/classification
    model_scratch = AutoModelForSequenceClassification.from_pretrained(
        MODEL3_SCRATCH_REPO, 
        subfolder=f"fold_{fold}", 
        token=HF_TOKEN, 
        torch_dtype=torch.float16,
        num_labels=1
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model to evaluation mode
    model_scratch.eval()
    
    fold_preds_scratch = []

    # Execute inference loop over Scratch DataLoader
    with torch.inference_mode():
        for batch in scratch_loader:
            # Transfer input batch to GPU device
            batch = {k: v.to(model_scratch.device) for k, v in batch.items()}
            
            # Forward pass obtaining scalar logit output per input sequence
            out = model_scratch(**batch)
            
            # Flatten logit tensor outputs and store in fold predictions list
            fold_preds_scratch.append(out.logits.squeeze(-1).cpu().numpy())

    # Concatenate all mini-batch outputs and reshape flat array of length [NUM_SAMPLES * 5] into matrix [NUM_SAMPLES, 5]
    predictions_scratch = np.concatenate(fold_preds_scratch, axis=0).reshape(-1, 5)
    
    # Accumulate 1/5th weight contribution of raw fold predictions into Model 3 logits matrix
    m3_ensemble_logits += predictions_scratch / 5.0
    
    # Clean up model references and clear PyTorch VRAM cache
    del model_scratch
    gc.collect()
    torch.cuda.empty_cache()

# Convert accumulated ensembled option logits for Model 3 into probability distributions via row-wise softmax
m3_ensemble_probs = softmax(m3_ensemble_logits, axis=1)

# Helper function applying power scaling sharpening transformation to probability distributions
def sharpen_probs(probs, power=1.5):
    p_pow = np.power(probs, power)
    return p_pow / np.sum(p_pow, axis=1, keepdims=True)

# Helper function converting probability arrays into normalized row-wise percentile ranks ranging from 0.0 to 1.0
def to_percentile_ranks(probs):
    ranks = np.zeros_like(probs)
    for i in range(len(probs)):
        ranks[i] = rankdata(probs[i]) / len(probs[i])
    return ranks

# Compute sharpened probability distribution for Model 1 (Qwen3 RAG) using power exponent 1.5
m1_sharp = sharpen_probs(m1_ensemble_probs, power=1.5)

# Compute sharpened probability distribution for Model 2 (DeBERTa-v3) using power exponent 1.8 to accentuate top choice
m2_sharp = sharpen_probs(m2_ensemble_probs, power=1.8)

# Compute sharpened probability distribution for Model 3 (Scratch) using power exponent 1.5
m3_sharp = sharpen_probs(m3_ensemble_probs, power=1.5)

# Compute scale-invariant percentile rank matrices for Model 1
m1_ranks = to_percentile_ranks(m1_ensemble_probs)

# Compute scale-invariant percentile rank matrices for Model 2
m2_ranks = to_percentile_ranks(m2_ensemble_probs)

# Compute scale-invariant percentile rank matrices for Model 3
m3_ranks = to_percentile_ranks(m3_ensemble_probs)

# Initialize matrix storing final combined ensemble score distributions
final_blend_scores = np.zeros_like(m2_ensemble_probs)

# Perform row-by-row dynamic confidence-gated blending
for i in range(NUM_SAMPLES):
    # Sort DeBERTa probability values for current sample in descending order
    deb_sorted = np.sort(m2_ensemble_probs[i])[::-1]
    
    # Calculate margin difference between highest probability choice and second highest choice
    margin = deb_sorted[0] - deb_sorted[1]

    # Assign heavy weight to DeBERTa when top choice prediction confidence margin is high (>= 0.35)
    if margin >= 0.35:
        w_deb, w_scratch, w_qwen = 0.82, 0.11, 0.07
    # Assign moderate weight when DeBERTa prediction margin is medium (>= 0.15)
    elif margin >= 0.15:
        w_deb, w_scratch, w_qwen = 0.62, 0.23, 0.15
    # Allocate higher weight to Scratch and Qwen3 models when DeBERTa prediction is uncertain (< 0.15)
    else:
        w_deb, w_scratch, w_qwen = 0.42, 0.34, 0.24

    # Calculate weighted linear blend of sharpened probability distributions
    p_blend = (w_deb * m2_sharp[i]) + (w_scratch * m3_sharp[i]) + (w_qwen * m1_sharp[i])
    
    # Calculate weighted linear blend of percentile rank distributions
    r_blend = (w_deb * m2_ranks[i]) + (w_scratch * m3_ranks[i]) + (w_qwen * m1_ranks[i])

    # Compute hybrid final score combining 85% sharpened probability blend and 15% percentile rank blend
    final_blend_scores[i] = (0.85 * p_blend) + (0.15 * r_blend)

# Extract indices of top 3 ranked option choices per sample sorted by descending final ensemble score
top3_indices = np.argsort(-final_blend_scores, axis=1)[:, :3]

# Map numerical option indices 0-4 to corresponding answer letter labels A-E
idx_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Format top 3 prediction indices into space-delimited string labels per question sample
predicted_labels = [" ".join([idx_to_label[i] for i in row]) for row in top3_indices]

df_sub = pd.DataFrame({
    id_col: df_test[id_col],
    "Prediction": predicted_labels
})

submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)

if WANDB_API_KEY:
    try:
        m1_conf = np.max(m1_ensemble_probs, axis=1).mean()
        m2_conf = np.max(m2_ensemble_probs, axis=1).mean()
        m3_conf = np.max(m3_ensemble_probs, axis=1).mean()
        
        wandb.log({
            "M1_Qwen3_Mean_Max_Prob": m1_conf,
            "M2_DeBERTa_Mean_Max_Prob": m2_conf,
            "M3_Scratch_Mean_Max_Prob": m3_conf,
        })
        
        wandb_table = wandb.Table(columns=["ID", "Question", "A", "B", "C", "D", "E", "Prediction"])
        for idx, row in df_test.iterrows():
            wandb_table.add_data(
                row[id_col], row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"], predicted_labels[idx]
            )
        wandb.log({"TriModel_OptimizedEnsemble_Predictions": wandb_table})
        wandb.finish()
    except Exception:
        pass
