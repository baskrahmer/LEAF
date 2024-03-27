from config import Config
from preprocess import filter_data
from train import train

if __name__ == "__main__":
    c = Config()

    data_path = filter_data(c)
    model = train(c, data_path=data_path, model=None, mlm=True) if c.mlm_train_steps else None
    model = train(c, data_path=data_path, model=model) if c.train_steps else None
