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
    },
}
n_runs = np.prod([len(v["values"]) for v in sweep_configuration["parameters"].values()])
sweep_id = wandb.sweep(sweep=sweep_configuration, project="leaf")


def wrapped_main():
    wandb.init(project="leaf")
    c = Config(
        use_wandb=True,
        sample_size=10000,
        model_name="distilbert/distilbert-base-multilingual-cased",
        mlm_train_steps=1000,
        mlm_val_steps=100,
        learning_rate=wandb.config.learning_rate,
        train_batch_size=wandb.config.batch_size,
        score_metric="macro-perplexity",
    )
    score = main(c)
    wandb.log({"score": score})


wandb.agent(sweep_id, function=wrapped_main, count=n_runs)
