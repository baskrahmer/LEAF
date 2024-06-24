import copy
import csv
import json
import random
from collections import Counter
from typing import List

from lightning import seed_everything
from openai import OpenAI
from tqdm import tqdm

from src.config import Config
from src.data import get_dataset
from src.preprocess import filter_data
from src.utils import get_lci_name_mapping

ASSISTANT_PROMPT = """You are a helpful assistant.
Your task is to classify the text string given by the user.
The string can be presented in any language.
You must pick a class from the permitted categories you are provided, even if the correct class is not in the list.
"""

ASSISTANT_PROMPT_PARAPHRASED = """You are an assistant dedicated to providing support.
Your objective is to categorize the text provided by the user.
This text may be in any language.
You must choose a category from the allowed list of options, even if the most appropriate category isn't included.
"""

LINGUIST_PROMPT = """You are an expert linguist and text classifier.
Your task is to classify the text string given by the user.
The string can be presented in any language.
You must pick the correct class from the list of permitted categories, even if the correct class is not in the list.
"""

ENVIRONMENTALIST_PROMPT = """You are an expert in assessing the environmental impact of food products.
Your task is to classify the text string given by the user.
The string can be presented in any language.
You must pick the correct class from the list of permitted categories, even if the correct class is not in the list.
"""

MINIMAL_PROMPT = """Your task is to classify the text string given by the user.
The string can be presented in any language.
You must pick the correct class from the list of permitted categories, even if the correct class is not in the list.
"""


class OpenAIBaseline:

    def __init__(self, c: Config, model_name: str, system_prompt: str = ASSISTANT_PROMPT):
        self.c = c
        self.model_name = model_name
        self.client = OpenAI()
        self.system_prompt = system_prompt

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
                    "content": self.system_prompt,
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
        counts = Counter(short_keys.keys())
        for k, v in counts.items():
            if v == 1:
                short_lci_mapping[short_keys[k]] = k
        n_commas += 1

    return short_lci_mapping


def main(
        c: Config,
        model_name: str = "gpt-3.5-turbo",
        choice_splits: int = 2,
        min_predictions: int = 1000,
        experiment_name: str = "openai",
        system_prompt: str = ASSISTANT_PROMPT,
):
    seed_everything(c.seed)
    test_set = get_dataset(c, filter_data(c), c.test_size)["test"]
    label_to_lci = get_lci_name_mapping(c)
    lci_to_label = {v: k for k, v in label_to_lci.items()}

    ai_classifier = OpenAIBaseline(c, model_name=model_name, system_prompt=system_prompt)

    predictions = []
    n_predicted = 0
    categories = list(lci_to_label.keys())
    for product in tqdm(test_set, desc="Getting predictions"):
        random.shuffle(categories)
        split_outputs = []
        hallucinations = []
        for i in range(choice_splits):
            split = categories[i * len(categories) // choice_splits:(i + 1) * len(categories) // choice_splits]
            out = ai_classifier(product["product_name"], categories=split)
            if out in categories:
                split_outputs.append(out)
            else:
                hallucinations.append(out)
        sample = copy.deepcopy(product)
        sample['predicted_raw'] = split_outputs
        if split_outputs:
            out = ai_classifier(product["product_name"], categories=split_outputs)
            sample['predicted'] = lci_to_label.get(out)
            if out not in lci_to_label:
                hallucinations.append(out)
        else:
            sample['predicted'] = None
        sample['hallucinations'] = hallucinations

        predictions.append(sample)
        n_predicted += int(sample['predicted'] is not None)

        if n_predicted >= min_predictions:
            break

    with open(f'{experiment_name}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    shared_kwargs = {
        "c": Config(sample_size=0),
        "model_name": "gpt-3.5-turbo",
        "choice_splits": 2,
        "min_predictions": 200,
    }
    main(
        experiment_name="assistant",
        system_prompt=ASSISTANT_PROMPT,
        **shared_kwargs,
    )
    main(
        experiment_name="assistant_paraphrased",
        system_prompt=ASSISTANT_PROMPT_PARAPHRASED,
        **shared_kwargs,
    )
    main(
        experiment_name="linguist",
        system_prompt=LINGUIST_PROMPT,
        **shared_kwargs,
    )
    main(
        experiment_name="environmentalist",
        system_prompt=ENVIRONMENTALIST_PROMPT,
        **shared_kwargs,
    )
    main(
        experiment_name="minimal",
        system_prompt=MINIMAL_PROMPT,
        **shared_kwargs,
    )
