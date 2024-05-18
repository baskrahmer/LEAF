# LEAF: **L**inguistic **E**missions **A**nalysis of **F**ood products

![LEAF_concept_whitebg](https://github.com/baskrahmer/LEAF/assets/24520725/63c88582-55f3-4b48-bb95-31be33a43ef3)

Predict tne environmental impact of food products in over 50 languages! Use a pretrained model, or train your own. Code
accompanying the submission for ClimateNLP 2024 ACL workshop.

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

## Features

- [x] Data loading
    - [x] Ciqual data
        - [x] Extract EF from data
    - [x] OpenFoodFacts data
        - [x] Filter data with known CIQUAL class
    - [x] Data loading/saving
- [x] Exploratory data analysis
    - [x] Product distribution across languages
    - [x] Product distribution across classes
    - [x] Environmental footprint score across classes
- [x] Model definition
    - [x] MLM on unlabelled data
    - [x] Classification model
    - [x] Regression model
    - [x] Hybrid model
    - [x] Baseline models
        - [x] Cosine similarity with LCI name
        - [x] Zero-shot autoregressive LLM
- [x] Training pipeline
- [x] Class imbalances
    - [x] Train/test splits balanced by language and class
- [x] Evaluation pipeline
    - [x] Split across products
    - [x] Split across languages
    - [x] Classification metrics (accuracy, F1)
    - [x] Regression metrics (MAE)
- [x] Experiments
    - [x] Grid search
    - [ ] Learnable alpha
    - [x] Pooling mechanisms
    - [x] Longer training runs
- [x] Upload to HuggingFace
- [ ] Streamlit demo
- [x] Testing & reproducibility
- [x] Documentation
- [x] Writeup & submission