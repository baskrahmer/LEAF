from typing import Optional

import torch
from datasets import DatasetDict
from lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoModelForMaskedLM

from src.config import Config
from src.model import LightningWrapper, LEAFModel, get_tokenizer
from src.preprocess import prepare_inputs, prepare_inputs_mlm
from src.utils import get_loggers, get_callbacks, get_collate_fn, get_mlm_collate_fn, get_class_mapping, \
    get_ciqual_mapping


def train(c: Config, dataset: DatasetDict, base_model: Optional[PreTrainedModel], mlm: bool = False) -> PreTrainedModel:
    seed_everything(c.seed, workers=True)

    tokenizer, tokenizer_kwargs = get_tokenizer(c)

    train_ds = dataset["train"]
    val_ds = dataset["test"]

    if mlm:
        map_fn = lambda x: prepare_inputs_mlm(x, tokenizer, tokenizer_kwargs)
        collate_fn = get_mlm_collate_fn(tokenizer=tokenizer, mlm_probability=c.mlm_probability)
        class_to_idx = {}
        idx_to_co2e = {}
    else:
        class_to_idx = get_class_mapping(train_ds, val_ds)
        class_to_co2e = get_ciqual_mapping(c)
        idx_to_co2e = {idx: class_to_co2e[c] for c, idx in class_to_idx.items()}
        map_fn = lambda x: prepare_inputs(x, tokenizer, tokenizer_kwargs, class_to_idx, class_to_co2e)
        collate_fn = get_collate_fn(tokenizer)
    train_ds = train_ds.map(map_fn)
    val_ds = val_ds.map(map_fn)

    dl_kwargs = {"collate_fn": collate_fn, "num_workers": c.num_workers}
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=c.train_batch_size, **dl_kwargs)
    val_dataloader = DataLoader(val_ds, shuffle=False, batch_size=c.test_batch_size, **dl_kwargs)

    if mlm:
        base_model = AutoModelForMaskedLM.from_pretrained(c.model_name)
    else:
        base_model = LEAFModel(c, num_classes=len(class_to_idx.keys()), base_model=base_model,
                               idx_to_co2e=idx_to_co2e)
    lightning_model = LightningWrapper(
        c,
        tokenizer,
        model=base_model,
        num_classes=len(class_to_idx.keys()),
        mlm=mlm,
        languages=set(train_ds.unique("lang") + val_ds.unique("lang")),
        classes=set(train_ds.unique("label") + val_ds.unique("label")) if not mlm else set())

    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and c.use_gpu) else "cpu",
        enable_checkpointing=True,
        max_steps=c.mlm_train_steps if mlm else c.train_steps,
        val_check_interval=c.mlm_val_steps if mlm else c.val_steps,
        callbacks=get_callbacks(c),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=16 if c.fp16 else 32,
        logger=get_loggers(c),
        check_val_every_n_epoch=None,
        gradient_clip_val=c.gradient_clipping_value,
        accumulate_grad_batches=c.accumulate_grad_batches,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(
        dataloaders=val_dataloader,
        ckpt_path='best',
    )
    del lightning_model

    if c.save_path:
        with open(c.save_path + f"model{'_mlm' if mlm else ''}.pt", "wb") as f:
            torch.save(base_model.state_dict(), f)
        tokenizer.save_pretrained(c.save_path)

    if c.push_to_hub:
        base_model.push_to_hub(c.hub_repo_id)
        tokenizer.push_to_hub(c.hub_repo_id)

    return base_model
