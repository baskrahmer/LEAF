from src.config import Config
from src.main import main

base_config_kwargs = {
    "use_wandb": True,
    "sample_size": 0,
    "train_steps": 100000,
    "val_steps": 5000,
    "learning_rate": 0.05,
    "train_batch_size": 256,
    "score_metric": "test_all_mae",
    "drop_singular_classes": True,
}

script_configs = [
    Config(
        experiment_name="MP",
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        pooling="mean",
        gradient_clipping_value=0.1,
        **base_config_kwargs
    ),
    Config(
        experiment_name="NC",
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        pooling="cls",
        gradient_clipping_value=None,
        **base_config_kwargs
    ),
    Config(
        experiment_name="MP_NC",
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        pooling="mean",
        gradient_clipping_value=None,
        **base_config_kwargs
    ),
    Config(
        experiment_name="M3",
        model_name="BAAI/bge-m3",
        pooling="cls",
        gradient_clipping_value=0.1,
        **base_config_kwargs
    ),
    Config(
        experiment_name="MP_M3",
        model_name="BAAI/bge-m3",
        pooling="mean",
        gradient_clipping_value=0.1,
        **base_config_kwargs
    ),
    Config(
        experiment_name="LLFT",
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        pooling="mean",
        gradient_clipping_value=0.1,
        finetune_last_layer=True,
        **base_config_kwargs
    ),
]

for c in script_configs:
    main(c)
