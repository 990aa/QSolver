"""Environment-backed configuration for local and hosted runs."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings; secrets are read only when a workload starts."""

    train_path: Path = Path(os.getenv("MCQ_TRAIN_PATH", "data/train.csv"))
    test_path: Path = Path(os.getenv("MCQ_TEST_PATH", "data/test.csv"))
    knowledge_base_glob: str = os.getenv("MCQ_KNOWLEDGE_BASE_GLOB", "data/knowledge_base/*.arrow")
    output_path: Path = Path(os.getenv("MCQ_OUTPUT_PATH", "outputs/submission.csv"))
    model_namespace: str = os.getenv("MCQ_MODEL_NAMESPACE", "your-account")
    wandb_project: str = os.getenv("WANDB_PROJECT", "mcq-ensemble")

    @property
    def hf_token(self) -> str:
        """Return the optional Hub token without persisting or printing it."""
        return os.getenv("HF_TOKEN", "")

    @property
    def wandb_api_key(self) -> str:
        """Return the optional experiment-tracking token."""
        return os.getenv("WANDB_API_KEY", "")