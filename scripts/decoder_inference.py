import gc
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("WANDB_PROJECT", "mcq-ensemble")
os.environ.setdefault("WANDB_ENTITY", "default")

# Converts raw text collections into sparse term frequency-inverse document frequency (TF-IDF) feature matrices.
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, login

# Loads pre-trained neural transformer models to map text sequences into dense vector embeddings for semantic search.
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    wandb.login(key=WANDB_API_KEY)

BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_REPO_ID = os.environ.get("MCQ_DECODER_REPO", "your-account/mcq-ensemble-decoder")
TEST_CONTEXT_REPO = os.environ.get("MCQ_CONTEXT_REPO", "your-account/mcq-ensemble-context")
NUM_FOLDS = 5
TEST_PATH = os.environ.get("MCQ_TEST_PATH", "data/test.csv")
KB_GLOB = os.environ.get("MCQ_KNOWLEDGE_BASE_GLOB", "data/knowledge_base/*.arrow")
BASE_DIR = "./qwen_unsloth_checkpoints"
USE_CONTEXT = True
MAX_LENGTH = 1024 if USE_CONTEXT else 384
INFERENCE_BATCH_SIZE = 8

wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    config={
        "base_model": BASE_MODEL_NAME,
        "adapter_repo": ADAPTER_REPO_ID,
        "test_context_repo": TEST_CONTEXT_REPO,
        "num_folds": NUM_FOLDS,
        "use_context": USE_CONTEXT,
        "max_length": MAX_LENGTH,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
    }
)

df_test = pd.read_csv(TEST_PATH)
id_col = "id"
q_col = "prompt"

use_cached_retrievals = False

if USE_CONTEXT:
    try:
        cached_ds = load_dataset(TEST_CONTEXT_REPO, split="train")
        df_cached = cached_ds.to_pandas()
        
        if "prompt" in df_cached.columns and "context" in df_cached.columns:
            prompt_to_ctx = dict(zip(df_cached["prompt"], df_cached["context"]))
            if all(p in prompt_to_ctx for p in df_test["prompt"]):
                df_test["context"] = df_test["prompt"].map(prompt_to_ctx)
                use_cached_retrievals = True
    except Exception:
        pass

    if not use_cached_retrievals:
        arrow_files = glob.glob(KB_GLOB)
        kb_dataset = load_dataset("arrow", data_files=arrow_files, split="train")
        kb_texts = kb_dataset["text"]

        # TfidfVectorizer converts text to sparse matrices; stop_words removes common words, sublinear_tf logarithmically scales term frequency (1 + log(TF)), max_features caps vocabulary dimension, and float32 saves memory.
        vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            max_features=1_500_000,
            dtype=np.float32,
        )
        # fit_transform learns vocabulary dictionary from kb_texts and constructs the sparse TF-IDF feature matrix.
        kb_matrix = vectorizer.fit_transform(kb_texts)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # SentenceTransformer loads a neural encoder model onto GPU/CPU to project queries and passages into dense embedding spaces for semantic reranking.
        reranker = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

        def retrieve_contexts(df, batch_size=256, top_k_sparse=15, top_k_final=3):
            # Concatenates prompt and all 5 option text strings into a single search query string per row.
            queries = (
                df["prompt"].astype(str) + " " +
                df["A"].astype(str) + " " +
                df["B"].astype(str) + " " +
                df["C"].astype(str) + " " +
                df["D"].astype(str) + " " +
                df["E"].astype(str)
            ).fillna("").tolist()

            contexts = []
            for i in range(0, len(queries), batch_size):
                batch_q = queries[i : i + batch_size]
                # Converts query batch into sparse vectors and computes dot products against transposed knowledge base matrix to calculate lexical TF-IDF similarity scores.
                sparse_scores = vectorizer.transform(batch_q).dot(kb_matrix.T)
                # Encodes query strings into L2-normalized dense vector embeddings using the neural reranker model.
                q_embeds = reranker.encode(batch_q, normalize_embeddings=True, show_progress_bar=False)

                for j in range(sparse_scores.shape[0]):
                    # Extracts sparse similarity score row vector for the j-th query in the batch.
                    row = sparse_scores.getrow(j)
                    if len(row.data) == 0:
                        contexts.append("")
                        continue

                    # Sorts non-zero score values in descending order to find array positions for top_k_sparse candidate passages.
                    top_sparse_idx = np.argsort(-row.data)[:top_k_sparse]
                    candidate_indices = row.indices[top_sparse_idx]
                    candidate_passages = [kb_texts[int(idx)] for idx in candidate_indices]

                    # Encodes candidate passage texts into normalized dense embeddings and calculates dot product cosine similarities against query embedding.
                    passage_embeds = reranker.encode(candidate_passages, normalize_embeddings=True, show_progress_bar=False)
                    dense_scores = np.dot(passage_embeds, q_embeds[j])

                    # Ranks dense similarity scores in descending order and joins top_k_final candidate passage texts into a single context string.
                    top_final_idx = np.argsort(-dense_scores)[:top_k_final]
                    contexts.append(" ".join([candidate_passages[idx] for idx in top_final_idx]))

                gc.collect()

            return contexts

        df_test["context"] = retrieve_contexts(df_test)

        del kb_matrix, kb_texts, kb_dataset, reranker, vectorizer
        gc.collect()
        torch.cuda.empty_cache()

        try:
            cols_to_save = [c for c in [id_col, "prompt", "A", "B", "C", "D", "E", "context"] if c in df_test.columns]
            ds_to_push = Dataset.from_pandas(df_test[cols_to_save].reset_index(drop=True))
            DatasetDict({"train": ds_to_push}).push_to_hub(TEST_CONTEXT_REPO)
        except Exception:
            pass
else:
    df_test["context"] = ""

def format_prompt(row):
    ctx_line = f"Context: {row['context']}\n" if USE_CONTEXT and pd.notna(row["context"]) and str(row["context"]).strip() else ""
    return (
        f"<|im_start|>system\n"
        f"You are a scientific expert. {'Base your answer STRICTLY on the provided Context. ' if USE_CONTEXT else ''}"
        f"Output ONLY the single letter corresponding to the correct option (A, B, C, D, or E).<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{ctx_line}Question: {row['prompt']}\n"
        f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\nE) {row['E']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def infer_fold_process(fold_idx, gpu_id, test_df):
    # Isolates process execution to assigned GPU device ID.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Prevents PyTorch CUDA memory allocation fragmentation.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # FastLanguageModel provides memory-efficient 4-bit model loading and fast inference acceleration.
    from huggingface_hub import login

    # PeftModel handles loading and attaching saved fine-tuned LoRA adapter layers to base models.
    from peft import PeftModel
    from unsloth import FastLanguageModel

    if HF_TOKEN:
        login(token=HF_TOKEN)

    option_letters = ["A", "B", "C", "D", "E"]

    # Loads base causal language model architecture and tokenizer with 4-bit quantization enabled.
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Name or Hugging Face Hub repository path of base causal model to load.
        model_name=BASE_MODEL_NAME,
        # Maximum sequence context window length supported during inference.
        max_seq_length=MAX_LENGTH,
        # Automatically detects and selects optimal floating-point precision based on GPU hardware.
        dtype=None,
        # Enables 4-bit NormalFloat (NF4) quantization via BitsAndBytes to reduce VRAM consumption.
        load_in_4bit=True,
    )

    # Left truncation preserves prompt text, options, and assistant header when sequence length exceeds MAX_LENGTH.
    tokenizer.truncation_side = "left"
    # Left padding ensures that the final token position (index -1) in every padded batch is the assistant prompt header.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def get_answer_token_id(letter):
        probe_prefix = "<|im_start|>assistant\n"
        ids_prefix = tokenizer(probe_prefix, add_special_tokens=False)["input_ids"]
        ids_full = tokenizer(probe_prefix + letter, add_special_tokens=False)["input_ids"]
        return ids_full[len(ids_prefix):][0]

    option_token_ids = [get_answer_token_id(l) for l in option_letters]

    # Loads fine-tuned LoRA adapter weights saved for the current cross-validation fold onto the base model.
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_REPO_ID,
        subfolder=f"fold_{fold_idx}",
    )

    # Prepares PEFT model for fast inference execution by optimizing internal PyTorch forward pass operators.
    FastLanguageModel.for_inference(model)

    prompts = [format_prompt(row) for _, row in test_df.iterrows()]
    fold_logits = np.zeros((len(test_df), 5))

    # Disables autograd tracking during forward pass execution to minimize VRAM consumption and speed up inference.
    with torch.inference_mode():
        for i in range(0, len(prompts), INFERENCE_BATCH_SIZE):
            batch_prompts = prompts[i:i + INFERENCE_BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(model.device)

            # Computes forward pass output while logits_to_keep=1 calculates logits only for the final generated token position.
            out = model(**inputs, logits_to_keep=1, use_cache=False)
            batch_logits = out.logits[:, -1, option_token_ids].float().cpu().numpy()
            fold_logits[i:i + len(batch_prompts)] = batch_logits

    os.makedirs(BASE_DIR, exist_ok=True)
    out_file = f"{BASE_DIR}/fold_{fold_idx}_logits.npy"
    np.save(out_file, fold_logits)

# Sets multiprocessing process start method to spawn for clean CUDA context creation in child processes.
mp.set_start_method("spawn", force=True)
num_gpus = torch.cuda.device_count()
folds_to_run = list(range(1, NUM_FOLDS + 1))

if num_gpus >= 2:
    fold_queue = list(folds_to_run)
    active_processes = {}

    for gpu in range(min(num_gpus, 2)):
        if fold_queue:
            f = fold_queue.pop(0)
            # Spawns parallel child process executing fold inference on assigned GPU device.
            p = mp.Process(target=infer_fold_process, args=(f, gpu, df_test))
            p.start()
            active_processes[gpu] = (p, f)

    while active_processes:
        time.sleep(3)
        finished_gpus = []
        for gpu, (p, f) in active_processes.items():
            if not p.is_alive():
                p.join()
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del active_processes[gpu]
            if fold_queue:
                next_f = fold_queue.pop(0)
                p = mp.Process(target=infer_fold_process, args=(next_f, gpu, df_test))
                p.start()
                active_processes[gpu] = (p, next_f)
else:
    for fold in folds_to_run:
        infer_fold_process(fold, 0, df_test)

ensemble_logits = np.zeros((len(df_test), 5))

for fold in range(1, NUM_FOLDS + 1):
    logits_path = f"{BASE_DIR}/fold_{fold}_logits.npy"
    if os.path.exists(logits_path):
        fold_logits = np.load(logits_path)
        ensemble_logits += fold_logits

# Averages logit scores across all cross-validation fold models to finalize ensemble predictions.
ensemble_logits /= NUM_FOLDS

# Sorts logits in descending order along option choices and extracts column indices of top 3 predicted choices.
top3_indices = np.argsort(-ensemble_logits, axis=1)[:, :3]
idx_to_label = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
predicted_labels = [" ".join([idx_to_label[i] for i in row]) for row in top3_indices]

df_sub = pd.DataFrame({
    id_col: df_test[id_col],
    "Prediction": predicted_labels
})

df_sub.to_csv("submission.csv", index=False)

wandb.log({"submission_samples": len(df_sub)})
wandb.finish()
