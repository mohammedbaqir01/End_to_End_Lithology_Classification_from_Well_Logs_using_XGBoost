# Lithology Classification from Well Logs using XGBoost

## Project Overview

This petroleum engineering project applies supervised machine learning to classify **lithology** from **well log** data using the **XGBoost** algorithm. The model predicts three lithology classes — **Sandstone, Shale, and Limestone** — from petrophysical well log measurements, supporting faster and more consistent geological interpretation.

Lithology interpretation can be subjective and time-consuming. This project aims to **assist** engineers by providing fast, consistent lithology predictions, helping reduce interpretation variability rather than replacing expert judgment.

## Project Structure

```text
├── data/                         # Raw and processed well log datasets
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── .gitkeep
│
├── figures/                      # Figures and visualizations used in the README
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── models/                       # Trained machine learning artifacts
│   ├── ClassificationModel.pkl
│   └── label_encoder.pkl
│
├── notebooks/                    # Jupyter notebooks for EDA, development, and experiments
│   └── Lithology_Classification_from_Well_Logs_using_XGBoost.ipynb
│
├── reports/                      # Model evaluation results
│   ├── classification_report.txt
│   └── cross_validation_results.txt
│
├── src/                          # Main source-code package
│   ├── data/                     # Data loading, cleaning, and splitting
│   ├── evaluation/               # Model evaluation and reporting
│   ├── features/                 # Feature engineering
│   ├── modeling/                 # Model training and cross-validation
│   ├── serving/                  # API serving code
│   ├── tests/                    # Project tests
│   ├── utils/                    # Utility functions
│   ├── main.py                   # FastAPI application
│   └── run_pipeline.py           # End-to-end ML pipeline entry point
│
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Well Log Data

The dataset is based on the **FORCE 2020** well-log dataset.

## Results

| Metric | Score |
|--------|------:|
| Accuracy | 0.93 |
| Macro F1-Score | 0.86 |

### Confusion Matrix

| Class ID | Lithology |
|:--------:|-----------|
| 0 | Limestone |
| 1 | Sandstone |
| 2 | Shale |

![confusion_matrix](figures/confusion_matrix.png)

### Feature Importance

![feature_importance](figures/feature_importance.png)

## Machine Learning Workflow

```text
Well Log Data
      │
      ▼
Quick Data Exploration
      │
      ▼
Train/Test Split (by well)
      │
      ├─────────────────────────────┐
      │                             │
      ▼                             ▼
Training Set                  Testing Set
      │                             │
      ▼                             │
Data Cleaning                       │
      │                             │
      ▼                             │
Feature Engineering                 │
      │                             │
      ▼                             │
XGBoost Model Training              │
      │                             │
      ▼                             │
Group Cross-Validation              │
      │                             │
      └──────────────┐              │
                     ▼              ▼
             Final Model Evaluation
                     │
                     ▼
              Feature Importance
```

## How to Run

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mohammedbaqir01/Lithology-Classification-from-Well-Logs-using-XGBoost-Model.git
cd Lithology-Classification-from-Well-Logs-using-XGBoost-Model
pip install -r requirements.txt
```

### Run the ML Pipeline

The pipeline entry point is located inside `src/`.

Because the project uses imports such as `src.features...`, run it from the **project root** as a module:

```bash
python -m src.run_pipeline
```

### Run the API

The FastAPI application is defined in `src/main.py`.

Run it with Uvicorn:

```bash
python -m uvicorn src.main:app --reload
```

You can also run the module directly:

```bash
python -m src.main
```

> Run these commands from the project root directory, not from inside `src/`.

## Deployment

The FastAPI application is containerized with Docker and deployed on Microsoft Azure.

**Cloud Platform:** Microsoft Azure  
**API Framework:** FastAPI  
**Containerization:** Docker

### Live API

The API is publicly deployed and accessible through Azure:

[Swagger UI](http://lithologyapi.uaenorth.azurecontainer.io:8000/docs)


## License

Project is distributed under the MIT License.

## Author

**Mohammed Baqer Ahmed**

Petroleum Engineering graduate with a focus on Machine Learning applications in the oil and gas industry.

- 📧 **Email:** <mohammedbaqir010@gmail.com>
- 💼 **LinkedIn:** https://www.linkedin.com/in/mohammed-baqer-ahmed-079098280
- 🐙 **GitHub:** https://github.com/mohammedbaqir01
