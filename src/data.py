from collections import Counter
from typing import Optional

from datasets import DatasetDict
from datasets import load_dataset, concatenate_datasets

from src.config import Config


def get_dataset(c: Config, data_path: str, test_size: float, cls_dataset: Optional[DatasetDict] = None) -> DatasetDict:
    """
    Load and preprocess the dataset by turning it into another JSON file. The CLS dataset is created first and
    subsequently used prevent contamination of the MLM dataset.
    """
    dataset = load_dataset("json", data_files=data_path)["train"]
    if cls_dataset is not None:
        cls_train = cls_dataset["train"].remove_columns("label")
        cls_test = cls_dataset["test"].remove_columns("label")
    column_name = "stratification_column"
    if cls_dataset is not None:
        map_fn = lambda x: {column_name: x['lang']}
    else:
        map_fn = lambda x: {column_name: f"{x['lang']}_{x['label']}"}
    dataset = dataset.map(map_fn, num_proc=4)
    dataset = dataset.class_encode_column(column_name)
    if c.drop_singular_classes:
        value_counts = Counter(dataset[column_name])
        dataset = dataset.filter(lambda x: value_counts[x[column_name]] > (1 // test_size if not c.debug else 1))
    dataset = dataset.train_test_split(test_size=test_size, stratify_by_column=column_name).remove_columns(column_name)
    if cls_dataset is not None:
        dataset["train"] = concatenate_datasets([dataset["train"], cls_train])
        dataset["test"] = concatenate_datasets([dataset["test"], cls_test])
    return dataset
