import numpy as np
import wandb

from src.config import Config
from src.main import main

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 5e-4, 1e-5, 5e-5]},
        "accumulate_grad_batches": {"values": [1, 2, 4]},
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
        mlm_train_steps=100000,
        mlm_val_steps=5000,
        data_analysis=True,
        learning_rate=wandb.config.learning_rate,
        accumulate_grad_batches=wandb.config.accumulate_grad_batches,
        train_batch_size=64,
        test_batch_size=128,
        mlm_score_metric=True,
        score_metric="test_all_perplexity",
    )
    score = main(c)
    wandb.log({"score": score})


wandb.agent(sweep_id, function=wrapped_main, count=n_runs)
