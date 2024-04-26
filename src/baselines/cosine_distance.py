import copy
import csv
from typing import Literal

import torch
import torch.nn.functional as F
from lightning import seed_everything
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import Config
from src.data import get_dataset
from src.preprocess import filter_data
from src.utils import get_lci_name_mapping


class CosineBaseline:

    def __init__(self, c: Config, model_name: str, label_to_lci: dict, pooling_mode: Literal["cls", "mean"] = "mean"):
        self.c = c
        self.pooling_mode = pooling_mode
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.requires_grad = False

        # build embedding table
        self.sorted_keys = sorted(label_to_lci.keys())
        self.idx_to_label = {idx: k for idx, k in enumerate(self.sorted_keys)}
        table = []
        for label in tqdm(self.sorted_keys, desc="Building embedding table"):
            tokens = self.tokenizer(label_to_lci[label], return_tensors="pt", padding=True, truncation=True)
            if self.pooling_mode == "cls":
                embedding = self.model(**tokens).last_hidden_state[0, 0]
            else:
                embedding = self.model(**tokens).last_hidden_state[0].mean(dim=0)
            table.append(F.normalize(embedding, dim=0))
        self.table = torch.stack(table)

        self.label_to_lci = label_to_lci
        self.lci_to_label = {v: k for k, v in label_to_lci.items()}

    def __call__(self, input_text: str):
        tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        if self.pooling_mode == "cls":
            embedding = self.model(**tokens).last_hidden_state[:, 0]
        else:
            embedding = self.model(**tokens).last_hidden_state.mean(dim=1)
        embedding = F.normalize(embedding)

        # Calculate cosine similarity
        similarities = embedding @ self.table.T
        return self.idx_to_label[similarities.argmax().item()]


def main(c: Config, model_name="sentence-transformers/distiluse-base-multilingual-cased-v2", pooling_mode="mean"):
    seed_everything(c.seed)
    test_set = get_dataset(c, filter_data(c), c.test_size)["test"]
    label_to_lci = get_lci_name_mapping(c)

    cosine_classifier = CosineBaseline(c, model_name=model_name, label_to_lci=label_to_lci, pooling_mode=pooling_mode)

    predictions = []
    for product in tqdm(test_set, desc="Getting predictions"):
        sample = copy.deepcopy(product)
        sample['predicted'] = cosine_classifier(product["product_name"])
        predictions.append(sample)

    with open(f'cosine_preds_{pooling_mode}_pooling.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    for pooling_mode in ["cls", "mean"]:
        main(
            c=Config(sample_size=0),
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
            pooling_mode=pooling_mode,
        )
