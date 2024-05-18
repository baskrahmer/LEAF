import json
import os
from typing import Tuple, Any, TextIO, List

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.config import Config
from src.data import get_ciqual_data
from src.plot import make_data_analysis_report


def filter_data(c: Config, mlm: bool = False) -> str:
    config_hash = str(hash(tuple([c.sample_size, c.test_size, mlm])))

    filtered_products_filename = "products_filtered.jsonl" if not mlm else "products_mlm.jsonl"
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    products_path = os.path.join(data_dir, c.products_filename)
    output_path = os.path.join(data_dir, config_hash)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    filtered_products_path = os.path.join(output_path, filtered_products_filename)

    # Return cached data path if it exists
    if c.cache_data and os.path.exists(filtered_products_path):
        return filtered_products_path

    ciqual_data = get_ciqual_data(c)
    ciqual_to_agb = {str(c): str(a) for c, a in zip(ciqual_data["Code CIQUAL"], ciqual_data["Code AGB"])}
    agb_set = set(ciqual_data["Code AGB"])
    del ciqual_data

    columns = ["product_name", "lang"]
    # Note: other potential columns of interest include ecoscore_grade, ecoscore_tags, ecoscore_data,
    #  ciqual_food_name_tags, categories_old,  category_properties, categories_properties and categories_properties

    i = 0
    lang_frequencies, label_frequencies, lang_label_frequencies = {}, {}, {}
    with open(products_path) as f, open(filtered_products_path, 'w') as out_file:
        for line in tqdm(f):
            product = json.loads(line)
            successful = False

            if not product.get("product_name"):
                continue

            if "categories_properties" in product:
                successful, label = extract_labelled_point(agb_set, ciqual_to_agb, columns, out_file, product, mlm)

            if mlm:
                if not successful:
                    filtered_entry = {c: product.get(c) for c in columns}
                    out_file.write(json.dumps(filtered_entry))
                    out_file.write('\n')

                else:
                    continue

            lang = product.get("lang")
            lang_frequencies[lang] = lang_frequencies.get(lang, 0) + 1
            if not mlm:
                if lang not in lang_label_frequencies:
                    lang_label_frequencies[lang] = {}
                label_frequencies[label] = label_frequencies.get(label, 0) + 1
                if label not in lang_label_frequencies[lang]:
                    lang_label_frequencies[lang][label] = 0
                lang_label_frequencies[lang][label] += 1

            i += 1
            if c.sample_size and i > c.sample_size:
                break

    if c.data_analysis:
        make_data_analysis_report(c, lang_frequencies, label_frequencies, lang_label_frequencies, output_path, mlm)

    return filtered_products_path


def extract_labelled_point(agb_set: set, ciqual_to_agb: dict[str, str], columns: List[str], out_file: TextIO,
                           product: dict, mlm: bool) -> Tuple[bool, Any]:
    categories = product.pop("categories_properties")
    if categories.get("agribalyse_food_code:en") in agb_set:
        label = categories.get("agribalyse_food_code:en")
    elif categories.get("ciqual_food_code:en") in ciqual_to_agb:
        label = ciqual_to_agb[categories.get("ciqual_food_code:en")]
    elif categories.get("agribalyse_proxy_food_code:en") in agb_set:
        label = categories.get("agribalyse_proxy_food_code:en")
    else:
        return False, None
    if not mlm:
        filtered_entry = {c: product.get(c) for c in columns}
        filtered_entry["label"] = label
        out_file.write(json.dumps(filtered_entry))
        out_file.write('\n')
    return True, label


def prepare_inputs_mlm(sample: dict, tokenizer: PreTrainedTokenizerBase, tokenizer_kwargs: dict) -> dict:
    encodings = tokenizer(sample.pop('product_name'), **tokenizer_kwargs)
    sample['input_ids'] = encodings.input_ids
    sample['attention_mask'] = encodings.attention_mask

    # Fixes DataSet.map dropping the "lang" column for unknown reasons. This can probably also use os.sleep as a fix;
    #  it just seems to need some time to process the data.
    if "lang" not in sample:
        raise ValueError
    return sample


def prepare_inputs(sample: dict, tokenizer: PreTrainedTokenizerBase, tokenizer_kwargs: dict,
                   class_to_idx: dict, class_to_ef: dict) -> dict:
    sample['encodings'] = tokenizer(sample.pop('product_name'), **tokenizer_kwargs)
    sample['regressands'] = class_to_ef[sample['label']]
    sample['classes'] = class_to_idx[sample['label']]
    return sample
