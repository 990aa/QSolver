# QSolver — Smart MCQ Solver

This repository contains the full pipeline for a 5-choice multiple-choice
question answering system, built for the "Smart MCQ Solver" competition.
Every question has a `prompt` and five candidate answers (`A`–`E`), and the
goal is to predict the top 3 most likely correct options, evaluated using
**mAP@3** (Mean Average Precision at 3) — the true answer is worth full
credit if ranked 1st, half credit if ranked 2nd, and a third credit if ranked
3rd.

The final solution is not a single model. It combines three independently
trained models with very different architectures and inductive biases, and
merges their predictions using a confidence-aware ensembling strategy. The
intuition is that models that make different kinds of mistakes will cancel
each other's errors out when combined, producing a more reliable top-3
ranking than any single model could on its own.

## The three models

**Model 1 — RAG-augmented Qwen3 decoder (`scripts/decoder_train.py` / `scripts/decoder_inference.py`)**
A `Qwen/Qwen3-4B-Instruct-2507` causal language model, loaded in 4-bit via
Unsloth and fine-tuned with LoRA adapters (5-fold cross-validation, one
adapter per fold). Each question is formatted as a ChatML-style prompt
containing a retrieved context passage, the question, and the five options,
and the model is trained to output the single correct answer letter. At
inference time, the logits for the five option-letter tokens only are
extracted from the model's final output position and turned into a
probability distribution, which avoids scoring across the entire vocabulary.

**Model 2 — DeBERTa-v3-large encoder (`scripts/encoder_train.py` / `scripts/encoder_inference.py`)**
A `microsoft/deberta-v3-large` model with a native multiple-choice head
(`AutoModelForMultipleChoice`), trained with 5-fold cross-validation. Each
question is paired independently with each of its five options (so one
question becomes 5 tokenized sequences), and the model directly outputs one
logit per option. Training uses a custom `AWPTrainer` that applies
Adversarial Weight Perturbation to the embedding layer after the first
epoch, which nudges the model toward more robust decision boundaries and
tends to reduce overfitting on this kind of small labeled dataset.

**Model 3 — Small transformer trained from scratch (`scripts/scratch_train.py` / `scripts/scratch_inference.py`)**
A lightweight transformer built from a custom `AutoConfig` (4 layers, hidden
size 256, 4 attention heads) rather than a downloaded pretrained checkpoint —
its weights start entirely random. Each question-option pair is formatted as
a single text string (`Question: ...\nOption: ...`) and scored independently
with a regression-style single-logit sequence classification head. This
model is intentionally much smaller and weaker than the other two; it acts
as a diverse, low-correlation signal in the final ensemble precisely because
it learns very different patterns than the large pretrained encoders/decoders.

All three models are trained with `StratifiedKFold` (5 folds, seeded) so that
class balance across the five answer letters is preserved in every split,
and each fold's weights are pushed to a dedicated Hugging Face Hub model
repository (`fold_1` ... `fold_5` subfolders) rather than kept only as local
checkpoints, so inference can be run independently of training.

## Retrieval (`scripts/retrieval.py`)

Both the RAG-based decoder's training data and its test-time inputs rely on
retrieved context passages, since a general-purpose LLM's own memorized
knowledge is not always sufficient or precise enough to answer domain-specific
questions confidently. `retrieval.py` builds a two-stage retrieval pipeline
over the knowledge base:

1. **Sparse filtering** — a `TfidfVectorizer` over the full knowledge base
   quickly narrows millions of candidate passages down to the top 15 most
   lexically similar passages per query (prompt + all 5 options combined).
2. **Dense reranking** — a `BAAI/bge-small-en-v1.5` sentence embedding model
   re-scores those 15 candidates by semantic similarity and keeps the top 3,
   which are concatenated into a single context string.

This hybrid approach is much faster than running dense embedding search over
the entire knowledge base directly, while still getting the semantic
precision of a neural reranker on the final shortlist. The resulting
`context` column is pushed back to the Hub so training and inference scripts
can reuse it without recomputing retrieval every time.

## Ensembling (`scripts/ensemble.py`)

`scripts/ensemble.py` is the final inference script that ties everything together
end to end, and is what actually produces `submission.csv`. At a high level
it does the following:

1. **Loads or computes context** for the test set — it first checks Hugging
   Face Hub for a previously cached retrieval result matching ≥90% of the
   test prompts, and only re-runs the full hybrid retrieval pipeline from
   `retrieval.py`-style logic if no valid cache is found. This avoids
   recomputing an expensive retrieval pass every time inference is re-run.

2. **Runs inference with all three models**, each across their own 5 folds:
   - Qwen3 decoder logits are restricted to the 5 option-letter token IDs and
     softmax-normalized into probabilities per fold, then averaged across
     folds.
   - DeBERTa multiple-choice logits are softmax-normalized per fold and
     averaged across folds.
   - The scratch model's per-option logits are averaged across folds first,
     then softmax-normalized once at the end.

3. **Sharpens and rank-transforms each model's probabilities.** `sharpen_probs`
   raises probabilities to a power greater than 1 (1.5 for the decoder and
   scratch model, 1.8 for DeBERTa) and renormalizes, which pushes a model's
   own top choice further above its other choices — this is useful because
   softmax probabilities are often "too flat" to reflect true model
   confidence. `to_percentile_ranks` converts each model's raw probabilities
   into within-row percentile ranks, giving a scale-invariant signal that
   isn't distorted if one model's probabilities happen to be more spread out
   or more compressed than another's.

4. **Blends the three models with confidence-gated weights.** Rather than
   using one fixed set of ensemble weights for every question, the blend
   looks at DeBERTa's own margin between its top and second choice for each
   question:
   - Large margin (≥0.35) → DeBERTa is trusted heavily (82% weight), since a
     wide margin usually means it is genuinely confident and correct.
   - Medium margin (≥0.15) → weights shift moderately toward the other two
     models.
   - Small margin (<0.15) → DeBERTa's weight drops significantly in favor of
     the scratch model and Qwen3 decoder, since a narrow margin usually means
     DeBERTa itself is unsure and the other models' independent judgment
     becomes more valuable.

   The final per-question score is 85% the confidence-weighted blend of
   sharpened probabilities, plus 15% the confidence-weighted blend of
   percentile ranks, combining both an absolute-confidence view and a
   scale-invariant relative-ranking view of the three models' outputs.

5. **Extracts the top 3 options** by sorting the final blended scores in
   descending order per question, converts the winning indices back to
   letters (e.g. `A C B`), and writes `submission.csv`.

If Weights & Biases credentials are available, the script also logs each
model's mean top-choice confidence and a full prediction table for manual
inspection.

## Environment and credentials

All training and inference scripts expect to run in a Kaggle notebook
environment (or equivalent), reading:
- `HF_TOKEN` — a Hugging Face access token, used both to download base
  models/tokenizers and to push/pull fine-tuned fold checkpoints and cached
  datasets to the Hub.
- `WANDB_API_KEY` — optional; if present, training and inference metrics,
  logs, and prediction tables are streamed to Weights & Biases. If absent,
  scripts fall back to `report_to="none"` and skip W&B logging entirely
  rather than failing.

Both are read via `kaggle_secrets.UserSecretsClient` when available, falling
back to plain environment variables otherwise, so the same scripts can run
outside of Kaggle as long as those two variables are exported beforehand.

## Reproducibility

Every script sets a global seed (`SEED = 42`) across Python, NumPy, and
PyTorch, and cross-validation splits are generated with a seeded
`StratifiedKFold`, so re-running training or inference produces the same
fold assignments and, modulo hardware-level nondeterminism in GPU kernels,
closely matching results each time.

## Repository layout

```text
src/mcq_ensemble/       Import-safe configuration, validation, and ensemble math
scripts/                Explicit training, retrieval, and inference workloads
docs/assets/            Images extracted from the implementation report
REPORT.md               Detailed implementation report
```

Install and validate with `uv`:

```powershell
uv sync --extra gpu --group dev
uv run --group dev ruff check .
uv run --group dev ty check src
```

The workload scripts are intentionally not run by validation because they download
large models, require credentials, and target GPU hardware. Launch one explicitly
only after setting `MCQ_*`, `HF_TOKEN`, and optional `WANDB_API_KEY` environment variables:

```powershell
uv run --extra gpu python scripts/ensemble.py
```
