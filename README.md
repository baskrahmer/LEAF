# LEAF: **L**inguistic **E**missions **A**nalysis of **F**ood products

![LEAF_concept_whitebg](https://github.com/baskrahmer/LEAF/assets/24520725/63c88582-55f3-4b48-bb95-31be33a43ef3)

Predict tne environmental impact of food products in over 50 languages! Use a pretrained model, or train your own. Code
accompanying the submission for ClimateNLP 2024 ACL workshop.

## Loading models from Hugging Face

The models are available on the Hugging Face model hub:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("baskra/leaf-large")
model = AutoModel.from_pretrained("baskra/leaf-large", trust_remote_code=True)

model(**tokenizer("Nutella", return_tensors="pt"))
# {'logits': tensor([[-12.2842, ...]]), 'class_idx': tensor([1553]), 'ef_score': tensor([0.0129]), 'class': ['Chocolate spread with hazelnuts']}
```

## Navigating the repository

- `data/`: Contains sample data to run the tests with. The full data can be fetched by running `get_data.sh` in the root
  directory.
- `scripts/`: Contains the scripts to run the experiments in the paper.
- `src/`: Contains the source code for the project.
- `src/baselines/`: Contains scripts to run the baseline models.
- `tests/`: Contains simple end-to-end tests.

## Installation

To install the project, run the following commands:

```bash
pip install -e .[dev]
```

The project is developed and tested with Python 3.10.

## Citation

When using this model, please consider citing it as follows:

**BibTeX:**

```bibtex
@inproceedings{krahmer-2024-leaf,
    title = "{LEAF}: Predicting the Environmental Impact of Food Products based on their Name",
    author = "Krahmer, Bas",
    editor = "Stammbach, Dominik  and
      Ni, Jingwei  and
      Schimanski, Tobias  and
      Dutia, Kalyan  and
      Singh, Alok  and
      Bingler, Julia  and
      Christiaen, Christophe  and
      Kushwaha, Neetu  and
      Muccione, Veruska  and
      A. Vaghefi, Saeid  and
      Leippold, Markus",
    booktitle = "Proceedings of the 1st Workshop on Natural Language Processing Meets Climate Change (ClimateNLP 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.climatenlp-1.10",
    pages = "133--142",
    abstract = "Although food consumption represents a sub- stantial global source of greenhouse gas emis- sions, assessing the environmental impact of off-the-shelf products remains challenging. Currently, this information is often unavailable, hindering informed consumer decisions when grocery shopping. The present work introduces a new set of models called LEAF, which stands for Linguistic Environmental Analysis of Food Products. LEAF models predict the life-cycle environmental impact of food products based on their name. It is shown that LEAF models can accurately predict the environmental im- pact based on just the product name in a multi- lingual setting, greatly outperforming zero-shot classification methods. Models of varying sizes and capabilities are released, along with the code and dataset to fully reproduce the study.",
}
```
