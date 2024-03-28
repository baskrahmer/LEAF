import torch
from datasets import DatasetDict
from datasets import load_dataset
from lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoModelForMaskedLM

from config import Config
from src.model import LightningWrapper, LEAFModel, get_tokenizer
from src.preprocess import prepare_inputs, get_ciqual_data
from src.utils import get_loggers, get_callbacks, get_collate_fn


def get_dataset(data_path: str, test_size: float) -> DatasetDict:
    return load_dataset("json", data_files=data_path).train_test_split(test_size=test_size)


def train(c: Config, data_path: str, model, mlm: bool = False) -> PreTrainedModel:
    # TODO add correct model typing hint and integrate this argument
    seed_everything(c.seed, workers=True)

    if c.debug:
        c.train_steps = 10
        c.val_steps = 5
        c.accumulate_grad_batches = 1
        c.num_workers = 0

    tokenizer, tokenizer_kwargs = get_tokenizer(c)
    dataset = get_dataset(data_path, c.test_size)

    train_ds = dataset["train"]  # TODO train_test_split
    val_ds = dataset["test"]

    if c.debug:  # TODO add debug statement earlier on
        train_ds = train_ds.select(range(128))
        val_ds = val_ds.select(range(128))

    # TODO sort this by ascending label alphabetically
    # TODO refactor to get_mappings type of function or something, or get_map_fn function
    class_to_idx = {c: i for i, c in enumerate(set(train_ds['label'] + val_ds['label']))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    ciqual_data = get_ciqual_data()
    class_to_co2e = {str(c): co2 for c, co2 in zip(ciqual_data["Code AGB"], ciqual_data["Score unique EF"])}  # TODO
    del ciqual_data

    map_fn = lambda x: prepare_inputs(x, tokenizer, tokenizer_kwargs, class_to_idx, class_to_co2e)
    train_ds = train_ds.map(map_fn, num_proc=max(c.num_workers, 1))
    val_ds = val_ds.map(map_fn, num_proc=max(c.num_workers, 1))

    collate_fn = get_collate_fn(tokenizer)

    dl_kwargs = {"collate_fn": collate_fn, "num_workers": c.num_workers}
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=c.train_batch_size, **dl_kwargs)
    val_dataloader = DataLoader(val_ds, shuffle=False, batch_size=c.test_batch_size, **dl_kwargs)

    if mlm:
        base_model = AutoModelForMaskedLM.from_pretrained(c.model_name)
    else:
        base_model = LEAFModel(c, num_classes=len(class_to_idx.keys()), base_model=model)
    model = LightningWrapper(c, tokenizer, model=base_model, num_classes=len(class_to_idx.keys()), mlm=mlm)

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
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(
        dataloaders=val_dataloader,
        ckpt_path='best',
    )

    if c.save_path:
        # TODO use pytorch saving mechanism
        model.model.save_pretrained(c.save_path)
        tokenizer.save_pretrained(c.save_path)

    if c.push_to_hub:
        model.model.push_to_hub(c.hub_repo_id)
        tokenizer.push_to_hub(c.hub_repo_id)

    return model.model
