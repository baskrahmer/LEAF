# LEAF: **L**inguistic **E**missions **A**nalysis of **F**ood products

### TODO

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