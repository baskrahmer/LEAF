from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass
class Config:
    seed: int = 42
    debug: bool = False
    version: str = field(default_factory=lambda: datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    use_wandb: bool = False
    use_gpu: bool = True
    num_workers: int = 0
    experiment_name: Optional[str] = None

    # Data settings
    sample_size: int = 20
    test_size: float = 0.2
    cache_data: bool = True
    data_analysis: bool = False
    ciqual_filename: str = "ciqual.csv"
    products_filename: str = "products.jsonl"
    drop_singular_classes: bool = False

    # Model and generic training
    model_name: str = "hf-internal-testing/tiny-random-bert"
    objective: Literal["classification", "regression", "hybrid"] = "hybrid"
    pooling: Literal["mean", "cls"] = "mean"
    alpha: float = 0.5  # Only used for hybrid head
    fp16: bool = False
    max_length: int = 32
    train_batch_size: int = 64
    test_batch_size: int = 64
    accumulate_grad_batches: int = 1
    gradient_clipping_value: Optional[float] = None
    es_patience: int = 10
    es_delta: float = 0.0
    finetune_last_layer: bool = False

    # MLM settings
    mlm_model_path: str = ""  # Path to .pt file
    mlm_train_steps: int = 0
    mlm_val_steps: int = 5
    mlm_learning_rate: float = 1e-5
    mlm_probability: float = 0.15

    # Objective training settings
    train_steps: int = 10
    val_steps: int = 5
    learning_rate: float = 1e-5
    mlm_score_metric: bool = False
    score_metric: str = "test_all_mae"

    # Artefact settings
    save_path: str = "output"
    push_to_hub: bool = False
    push_ds_to_hub: bool = False
    hub_repo_id: str = ""
