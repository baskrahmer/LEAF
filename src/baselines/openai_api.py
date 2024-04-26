import copy
import csv
import json
import random
from collections import Counter
from typing import List

import tiktoken
from lightning import seed_everything
from openai import OpenAI
from tqdm import tqdm

from src.config import Config
from src.data import get_dataset
from src.preprocess import filter_data
from src.utils import get_lci_name_mapping


class OpenAIBaseline:

    def __init__(self, c: Config, model_name: str):
        self.c = c
        self.model_name = model_name
        self.client = OpenAI()

    def __call__(self, input_text: str, categories: List[str]):
        api_response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=64,
            temperature=0.1,
            seed=self.c.seed,
            function_call={"name": "category_results"},
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant.
Your task is to classify the text string given by the user.
The string can be presented in any language.
You must pick a class from the permitted categories you are provided, even if the correct class is not in the list.
""",
                },
                {
                    "role": "user",
                    "content": input_text,
                }
            ],
            functions=[
                {
                    "name": "category_results",
                    "description": "Report the category of the product.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": categories,
                                "description": "Permitted categories"
                            },
                        },
                        "required": ["category"],
                    },
                }
            ]
        )
        return json.loads(api_response.choices[0].message.function_call.arguments).get("category", "")


def main(c: Config, model_name="gpt-3.5-turbo", choice_splits=2, min_predictions=1000):
    seed_everything(c.seed)
    test_set = get_dataset(c, filter_data(c), c.test_size)["test"]
    label_to_lci = get_lci_name_mapping(c)
    lci_to_label = {v: k for k, v in label_to_lci.items()}

    ai_classifier = OpenAIBaseline(c, model_name=model_name)

    predictions = []
    n_predicted = 0
    categories = list(lci_to_label.keys())
    for product in tqdm(test_set, desc="Getting predictions"):
        random.shuffle(categories)
        split_outputs = []
        for i in range(choice_splits):
            split = categories[i * len(categories) // choice_splits:(i + 1) * len(categories) // choice_splits]
            out = ai_classifier(product["product_name"], categories=split)
            if out in categories:
                split_outputs.append(out)
        sample = copy.deepcopy(product)
        sample['predicted_raw'] = split_outputs
        if split_outputs:
            out = ai_classifier(product["product_name"], categories=split_outputs)
            sample['predicted'] = lci_to_label.get(out)
        else:
            sample['predicted'] = None

        predictions.append(sample)
        n_predicted += int(sample['predicted'] is not None)

        if n_predicted >= min_predictions:
            break

    with open('openai_preds.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)


def compress_key_strings(lci_to_label, model_name):
    short_lci_mapping, n_commas = {}, 1
    while len(short_lci_mapping) < len(lci_to_label):
        short_keys = {}
        global_counts = Counter(short_lci_mapping)
        for k in lci_to_label.keys():
            if k not in short_lci_mapping:
                short_k = ",".join(k.split(",")[:n_commas])
                short_keys[short_k] = k
                global_counts[short_k] += 1
                # TODO do counts here and merge with the previous counts
        counts = Counter(short_keys.keys())
        for k, v in counts.items():
            if v == 1:
                short_lci_mapping[short_keys[k]] = k
        n_commas += 1

    enc = tiktoken.encoding_for_model(model_name)
    n_tokens_before = sum([len(enc.encode(l)) for l in list(short_lci_mapping.keys())])
    n_tokens_after = sum([len(enc.encode(l)) for l in list(short_lci_mapping.values())])
    return short_lci_mapping


if __name__ == "__main__":
    main(Config(
        sample_size=0,
    ))
