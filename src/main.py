import tempfile

from src.config import Config
from src.data import get_dataset
from src.preprocess import filter_data
from src.train import train


def main(c: Config) -> float:
    return init_run(c)


def run(c: Config) -> float:
    dataset = get_dataset(c, filter_data(c), c.test_size)
    if c.mlm_train_steps:
        data_path = filter_data(c, mlm=True)
        mlm_dataset = get_dataset(c, data_path, c.test_size, cls_dataset=dataset)
        model, report = train(c, dataset=mlm_dataset, base_model=None, mlm=True) if c.mlm_train_steps else None
        if c.mlm_score_metric:
            return report[c.score_metric]
    else:
        model = None
    model, report = train(c, dataset=dataset, base_model=model) if c.train_steps else None
    return report[c.score_metric]


def init_run(c: Config) -> float:
    if c.use_gpu:
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if c.debug:
        c.train_steps = 10
        c.mlm_train_steps = 10
        c.val_steps = 5
        c.mlm_val_steps = 5
        c.accumulate_grad_batches = 1
        c.num_workers = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            c.save_path = tmpdir
            return run(c)
    else:
        return run(c)


if __name__ == "__main__":
    main(Config())
