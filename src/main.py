import tempfile

from src.config import Config
from src.data import get_dataset
from src.preprocess import filter_data
from src.train import train


def main(c: Config):
    dataset = get_dataset(c, filter_data(c), c.test_size)
    if c.mlm_train_steps:
        data_path = filter_data(c, mlm=True)
        mlm_dataset = get_dataset(c, data_path, c.test_size, cls_dataset=dataset)
        model = train(c, dataset=mlm_dataset, base_model=None, mlm=True) if c.mlm_train_steps else None
    else:
        model = None
    model = train(c, dataset=dataset, base_model=model) if c.train_steps else None


if __name__ == "__main__":
    c = Config()

    if not c.debug:
        main(c)

    else:
        c.train_steps = 10
        c.mlm_train_steps = 10
        c.val_steps = 5
        c.mlm_val_steps = 5
        c.accumulate_grad_batches = 1
        c.num_workers = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            c.save_path = tmpdirname
            main(c)
