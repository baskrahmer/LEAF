import copy

from src.config import Config
from src.main import main

dummy_config = Config()


def test_train_classification():
    c = copy.deepcopy(dummy_config)
    c.objective = "classification"
    main(c)


def test_train_regression():
    c = copy.deepcopy(dummy_config)
    c.objective = "regression"
    main(c)


def test_train_hybrid():
    c = copy.deepcopy(dummy_config)
    c.objective = "hybrid"
    main(c)


def test_train_mlm():
    c = copy.deepcopy(dummy_config)
    c.mlm_train_steps = 10
    c.mlm_val_steps = 5
    main(c)
