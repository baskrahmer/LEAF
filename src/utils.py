import os
import torch
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
from typing import List

from src.config import Config


def get_loggers(c: Config):
    loggers = [CSVLogger(save_dir=c.save_path, name=c.version)]
    if c.use_wandb:
        loggers.append(WandbLogger(name=c.version, save_dir=c.save_path, version=c.version, project="perturbers"))
    return loggers


def get_callbacks(c: Config) -> List[pl.callbacks.Callback]:
    return [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=c.es_delta,
            patience=c.es_patience,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_on_train_epoch_end=False,
            dirpath=os.path.join(c.save_path, c.version),
            every_n_epochs=1
        ),
    ]


def get_collate_fn(tokenizer: PreTrainedTokenizerBase):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt', padding=True)

    def collate_fn(batch: List) -> dict:
        encodings = collator([_["encodings"] for _ in batch])
        return_dict = {
            "input_ids": encodings.data["input_ids"],
            "attention_mask": encodings.data["attention_mask"],
            "lang": [_["lang"] for _ in batch]
        }
        if "classes" in batch[0]:
            return_dict["classes"] = torch.tensor([_["classes"] for _ in batch])  # TODO set dtype
        if "regressands" in batch[0]:
            return_dict["regressands"] = torch.tensor([_["regressands"] for _ in batch])  # TODO set dtype
        return return_dict

    return collate_fn
