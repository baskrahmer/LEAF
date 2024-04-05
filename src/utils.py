import os
from typing import Any
from typing import List

import torch
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import PreTrainedTokenizerBase

from src.config import Config
from src.preprocess import get_ciqual_data


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


def get_mlm_collate_fn(tokenizer, mlm_probability):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

    def collate_fn(batch: List[dict[str, Any]]) -> dict[str, Any]:
        languages = [_.pop("lang") for _ in batch]
        return_dict = collator(batch).data
        return_dict["lang"] = languages
        return return_dict

    return collate_fn


def get_collate_fn(tokenizer: PreTrainedTokenizerBase):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt', padding=True)

    def collate_fn(batch: List[dict[str, Any]]) -> dict[str, Any]:
        encodings = collator([_["encodings"] for _ in batch])
        return_dict = {
            "input_ids": encodings.data["input_ids"],
            "attention_mask": encodings.data["attention_mask"],
            "lang": [_["lang"] for _ in batch]
        }
        if "classes" in batch[0]:
            return_dict["classes"] = torch.tensor([_["classes"] for _ in batch])
        if "regressands" in batch[0]:
            return_dict["regressands"] = torch.tensor([_["regressands"] for _ in batch]).unsqueeze(-1)
        return return_dict

    return collate_fn


def get_class_mapping(train_ds, val_ds):
    all_labels = sorted(set(train_ds['label'] + val_ds['label']))
    class_to_idx = {c: i for i, c in enumerate(all_labels)}
    return class_to_idx


def get_ciqual_mapping(c: Config):
    ciqual_data = get_ciqual_data(c)
    class_to_co2e = {str(c): co2 for c, co2 in zip(ciqual_data["Code AGB"], ciqual_data["Score unique EF"])}  # TODO
    return class_to_co2e
