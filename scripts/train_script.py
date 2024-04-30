from src.config import Config
from src.main import main

c = Config(
    data_analysis=True,
    use_wandb=True,
    sample_size=10000,
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
    train_steps=10,
    val_steps=5,
    learning_rate=0.01,
    train_batch_size=256,
    objective="hybrid",
    score_metric="test_all_mae",
)

main(c)
