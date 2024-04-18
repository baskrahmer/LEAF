import numpy as np
import wandb

from src.config import Config
from src.main import main

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "learning_rate": {"values": [0.01, 0.001, 0.0001, 0.00001]},
        "batch_size": {"values": [64, 128, 256]},
        "objective": {"values": ["classification", "regression", "hybrid"]},
        "mlm_model_path": {"values": ["", "mlm_model.pt"]},  # TODO: add valid path once MLM sweep is done
    },
}
n_runs = np.prod([len(v["values"]) for v in sweep_configuration["parameters"].values()])
sweep_id = wandb.sweep(sweep=sweep_configuration, project="leaf")


def wrapped_main():
    wandb.init(project="leaf")
    c = Config(
        use_wandb=True,
        sample_size=0,
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        train_steps=100000,
        val_steps=5000,
        data_analysis=True,
        learning_rate=wandb.config.learning_rate,
        train_batch_size=wandb.config.batch_size,
        objective=wandb.config.objective,
        mlm_model_path=wandb.config.mlm_model_path,
        score_metric="test_mae",
    )
    score = main(c)
    wandb.log({"score": score})


wandb.agent(sweep_id, function=wrapped_main, count=n_runs)
