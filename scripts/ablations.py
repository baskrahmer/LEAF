from src.config import Config
from src.main import main

base_config_kwargs = {
    "use_wandb": True,
    "sample_size": 0,
    "train_steps": 100000,
    "val_steps": 5000,
    "train_batch_size": 256,
    "test_batch_size": 256,
    "objective": "classification",
    "score_metric": "test_all_mae",
}

script_configs = [
    Config(
        experiment_name="LLFT",
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        finetune_last_layer=True,
        **base_config_kwargs
    ),
    Config(
        experiment_name="M3_LOWERLR",
        model_name="BAAI/bge-m3",
        learning_rate=0.005,
        **base_config_kwargs
    ),
    Config(
        experiment_name="M3_LOWLR",
        model_name="BAAI/bge-m3",
        learning_rate=0.02,
        **base_config_kwargs
    ),
    Config(
        experiment_name="M3",
        model_name="BAAI/bge-m3",
        learning_rate=0.05,
        **base_config_kwargs
    ),
]

for c in script_configs:
    main(c)
