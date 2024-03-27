# LEAF: **L**inguistic **E**nvironmental **A**nalysis of **F**ood products

### TODO

- [ ] Data loading
    - [ ] Ciqual data
        - [ ] Extract Co2e from data
    - [ ] OpenFoodFacts data
        - [ ] Filter data with known CIQUAL class
    - [ ] Join and save as artefact
- [ ] Exploratory data analysis
    - [ ] Product distribution across languages
    - [ ] Product distribution across classes
    - [ ] Environmental footprint score across classes
- [ ] Model definition
    - [ ] MLM on unlabelled data
    - [ ] Classification model
        - [ ] Averaged Co2E by probability classes
    - [ ] Regression model
    - [ ] Baseline models
        - [ ] Cosine similarity with LCI name
        - [ ] Zero-shot autoregressive LLM
- [ ] Training pipeline
- [ ] Evaluation methodology
    - [ ] Split across products
    - [ ] Split across languages
    - [ ] Classification metrics (accuracy, F1)
    - [ ] Regression metrics (MAE)
- [ ] Streamlit demo
- [ ] Testing & reproducibility
- [ ] Writeup & submission