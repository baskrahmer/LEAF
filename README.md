# LEAF: **L**inguistic **E**nvironmental **A**nalysis of **F**ood products

### TODO

- [x] Data loading
    - [x] Ciqual data
        - [x] Extract Co2e from data
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
    - [ ] Baseline models
        - [ ] Cosine similarity with LCI name
        - [ ] Zero-shot autoregressive LLM
- [x] Training pipeline
- [x] Class imbalances
    - [x] Train/test splits balanced by language and class
    - [ ] Language and class weights in loss function
- [x] Evaluation pipeline
    - [x] Split across products
    - [x] Split across languages
    - [x] Classification metrics (accuracy, F1)
    - [x] Regression metrics (MAE)
- [ ] Streamlit demo
- [ ] Testing & reproducibility
- [ ] Documentation
- [ ] Writeup & submission