from src.config import Config
from src.main import main

c = Config(
    use_wandb=True,
    sample_size=10000,
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
    train_steps=1000,
    val_steps=100,
)

main(c)
