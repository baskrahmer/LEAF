import json
import os
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.config import Config


def filter_data(c: Config, mlm: bool = False) -> str:
    config_hash = str(hash(tuple([c.sample_size, c.test_size])))

    filtered_products_filename = "products_filtered.jsonl" if not mlm else "products_mlm.jsonl"
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    ciqual_path = os.path.join(data_dir, c.ciqual_filename)
    products_path = os.path.join(data_dir, c.products_filename)
    output_path = os.path.join(data_dir, config_hash)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    filtered_products_path = os.path.join(output_path, filtered_products_filename)

    # Return cached data path if it exists
    if c.cache_data and os.path.exists(filtered_products_path):
        return filtered_products_path

    ciqual_data = pd.read_csv(ciqual_path)
    ciqual_to_agb = {str(c): str(a) for c, a in zip(ciqual_data["Code CIQUAL"], ciqual_data["Code AGB"])}
    agb_set = set(ciqual_data["Code AGB"])
    del ciqual_data

    columns = ["product_name", "lang"]
    # Note: other potential columns of interest include ecoscore_grade, ecoscore_tags, ecoscore_data,
    #  ciqual_food_name_tags, categories_old,  category_properties, categories_properties and categories_properties

    i = 0
    lang_frequencies, label_frequencies = {}, {}
    with open(products_path) as f, open(filtered_products_path, 'w') as out_file:
        for line in tqdm(f):
            product = json.loads(line)

            if mlm:
                filtered_entry = {c: product.get(c) for c in columns}
                out_file.write(json.dumps(filtered_entry))
                out_file.write('\n')

            elif "categories_properties" in product:
                categories = product.pop("categories_properties")
                if categories.get("agribalyse_food_code:en") in agb_set:
                    label = categories.get("agribalyse_food_code:en")
                elif categories.get("ciqual_food_code:en") in ciqual_to_agb:
                    # TODO investigate why this happens
                    label = ciqual_to_agb[categories.get("ciqual_food_code:en")]
                elif categories.get("agribalyse_proxy_food_code:en") in agb_set:
                    label = categories.get("agribalyse_proxy_food_code:en")
                else:
                    continue

                filtered_entry = {c: product.get(c) for c in columns}
                filtered_entry["label"] = label
                out_file.write(json.dumps(filtered_entry))
                out_file.write('\n')

            else:
                continue

            lang = product.get("lang")
            lang_frequencies[lang] = lang_frequencies.get(lang, 0) + 1
            if not mlm:
                label_frequencies[label] = label_frequencies.get(label, 0) + 1

            i += 1
            if c.sample_size and i > c.sample_size:
                break

    if c.data_analysis:
        # TODO make histogram
        footprint_scores = pd.read_csv(ciqual_path)["Score unique EF"]

    return filtered_products_path


def prepare_inputs_mlm(sample: dict, tokenizer: PreTrainedTokenizerBase, tokenizer_kwargs: dict) -> dict:
    encodings = tokenizer(sample.pop('product_name'), **tokenizer_kwargs)
    sample['input_ids'] = encodings.input_ids
    sample['attention_mask'] = encodings.attention_mask
    sample.pop('lang')  # TODO make this work with collator so we dont have to pop
    return sample


def prepare_inputs(sample: dict, tokenizer: PreTrainedTokenizerBase, tokenizer_kwargs: dict,
                   class_to_idx: dict, class_to_co2e: dict) -> dict:
    sample['encodings'] = tokenizer(sample.pop('product_name'), **tokenizer_kwargs)
    sample['regressands'] = class_to_co2e[sample['label']]
    sample['classes'] = class_to_idx[sample['label']]
    return sample


def get_ciqual_data(c: Config):
    # TODO deduplicate this with other CIQUAL loading logic
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    ciqual_path = os.path.join(data_dir, c.ciqual_filename)
    return pd.read_csv(ciqual_path)
