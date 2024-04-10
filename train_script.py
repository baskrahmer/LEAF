from src.config import Config
from src.main import main

c = Config(
    use_wandb=True,
    sample_size=10000,
    model_name="distilbert/distilbert-base-multilingual-cased",
    train_steps=1000,
    val_steps=100,
)

main(c)
