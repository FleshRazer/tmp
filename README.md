# F23-PMLDL-Assignment-1

Danil Andreev, B20-AI <br>
d.andreev@innopolis.university

## Notebooks

- <a target="_blank" href=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Data Exploration <br>
- <a target="_blank" href=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Llama2 7B Chat Fine-Tuning <br>
- <a target="_blank" href=""> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> T5v1.1 Large Fine-Tuning

## Repository Structure

```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources.
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data.
│
├── models       # Trained and serialized models, final checkpoints.
│
├── notebooks    # Jupyter notebooks. Naming convention is a number (for ordering),
│                  and a short delimited description, e.g.
│                  "1.0-initial-data-exporation.ipynb".
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'.
└── src                 # Source code for use in this assignment.
    │                 
    ├── data            # Scripts to download or generate data.
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions.
    │   ├── predict_model.py
    │   └── train_model.py
    │   
    └── visualization   # Scripts to create exploratory and results oriented visualizations.
        └── visualize.py
```

