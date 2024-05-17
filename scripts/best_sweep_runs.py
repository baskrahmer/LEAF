from src.config import Config
from src.main import main

base_config_kwargs = {
    "use_wandb": True,
    "sample_size": 0,
    "train_steps": 100000,
    "val_steps": 5000,
    "model_name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "score_metric": "test_all_mae",
}

script_configs = [
    Config(
        experiment_name="LEAF_C",
        objective="classification",
        train_batch_size=256,
        learning_rate=0.05,
        **base_config_kwargs
    ),
    Config(
        experiment_name="LEAF_R",
        objective="regression",
        train_batch_size=64,
        learning_rate=0.005,
        **base_config_kwargs
    ),
    Config(
        experiment_name="LEAF_H",
        objective="hybrid",
        train_batch_size=256,
        learning_rate=0.005,
        **base_config_kwargs
    ),
]

for c in script_configs:
    main(c)
