import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login

# Loads pre-trained neural transformer models to map text sequences into dense vector embeddings for semantic search.
from sentence_transformers import SentenceTransformer

# Converts raw text collections into sparse term frequency-inverse document frequency (TF-IDF) feature matrices.
from sklearn.feature_extraction.text import TfidfVectorizer

repo_id = os.environ.get("MCQ_TRAIN_DATASET", "your-account/mcq-ensemble-train")
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    login(token=hf_token)
df_train = load_dataset(repo_id, split="train").to_pandas()
arrow_files = glob.glob(os.environ.get("MCQ_KNOWLEDGE_BASE_GLOB", "data/knowledge_base/*.arrow"))
kb_texts = load_dataset("arrow", data_files=arrow_files, split="train")['text']

# TfidfVectorizer converts text to sparse matrices; stop_words removes noise words, sublinear_tf scales term frequencies logarithmically (1 + log(TF)) to reduce dominant word influence, max_features caps vocabulary size, and float32 saves memory.
vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=1500000, dtype=np.float32)
# fit_transform learns the vocabulary dictionary from kb_texts and constructs the sparse TF-IDF feature matrix.
kb_matrix = vectorizer.fit_transform(kb_texts)

device = "cuda" if torch.cuda.is_available() else "cpu"
# SentenceTransformer loads a neural encoder model onto the GPU/CPU to project queries and passages into dense embedding spaces for semantic reranking.
reranker = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)

def retrieve_contexts(df, batch_size=256, top_k_sparse=15, top_k_final=3):
    # Concatenates the prompt and all 5 multiple-choice options into a single unified search query string per row.
    queries = (
        df['prompt'].astype(str) + " " + 
        df['A'].astype(str) + " " + 
        df['B'].astype(str) + " " + 
        df['C'].astype(str) + " " + 
        df['D'].astype(str) + " " + 
        df['E'].astype(str)
    ).fillna("").tolist()
    
    contexts = []
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i : i + batch_size]
        
        # Converts query batch into sparse vectors and multiplies them by the transposed knowledge base matrix to calculate lexical TF-IDF similarity scores.
        sparse_scores = vectorizer.transform(batch_q).dot(kb_matrix.T)
        
        # Encodes query strings into L2-normalized dense vector embeddings using the neural reranker model.
        q_embeds = reranker.encode(batch_q, normalize_embeddings=True, show_progress_bar=False)
        
        for j in range(sparse_scores.shape[0]):
            # Extracts the row vector containing sparse similarity scores for the j-th query in the current batch.
            row = sparse_scores.getrow(j)
            if len(row.data) == 0:
                contexts.append("")
                continue
            
            # Sorts non-zero score values in descending order to find array positions for the top_k_sparse candidate passages.
            top_sparse_idx = np.argsort(-row.data)[:top_k_sparse]
            candidate_indices = row.indices[top_sparse_idx]
            candidate_passages = [kb_texts[int(idx)] for idx in candidate_indices]
            
            # Encodes candidate passage texts into normalized dense embeddings and calculates dot product cosine similarities against the query embedding.
            passage_embeds = reranker.encode(candidate_passages, normalize_embeddings=True, show_progress_bar=False)
            dense_scores = np.dot(passage_embeds, q_embeds[j])
            
            # Ranks dense similarity scores in descending order and joins the top_k_final candidate passage texts into a single context string.
            top_final_idx = np.argsort(-dense_scores)[:top_k_final]
            contexts.append(" ".join([candidate_passages[idx] for idx in top_final_idx]))
            
        gc.collect()
        
    return contexts

df_train['context'] = retrieve_contexts(df_train)
Dataset.from_pandas(df_train).push_to_hub(repo_id)
