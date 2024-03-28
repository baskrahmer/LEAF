import tempfile

from config import Config
from preprocess import filter_data
from train import train


def main(c: Config):
    if c.mlm_train_steps:
        data_path = filter_data(c, mlm=True)
        model = train(c, data_path=data_path, base_model=None, mlm=True) if c.mlm_train_steps else None
    else:
        model = None
    model = train(c, data_path=filter_data(c), base_model=model) if c.train_steps else None


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
