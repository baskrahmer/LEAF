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

[More Information Needed]

**APA:**

[More Information Needed]