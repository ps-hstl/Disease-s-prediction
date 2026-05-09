# Multiple Disease Prediction System

A machine-learning web application that predicts diseases from patient-reported symptoms using an ensemble of three classifiers — **SVM, Naive Bayes, and Random Forest** — and returns a majority-vote final diagnosis.

---

## Features

- **41 diseases** and **132 symptoms** covered
- **Ensemble voting** across three independent ML models
- **Confidence score** and per-model breakdown displayed for every prediction
- **Interactive Streamlit UI** with multi-select symptom picker and probability bar chart
- **Data Explorer tab** — training distribution charts and raw data preview
- **CLI mode** — run predictions directly from the terminal
- Clean, modular codebase: training, inference, and UI are fully separated

---

## Project Structure

```
disease-predictor/
├── app.py                  # Streamlit web application
├── cli.py                  # Command-line interface
├── requirements.txt
├── data/
│   ├── Training.csv        # 4 920-sample training dataset (132 symptoms, 41 diseases)
│   └── Testing.csv         # Hold-out test set
├── models/                 # Auto-generated after training
│   ├── svm_model.pkl
│   ├── naivebayes_model.pkl
│   ├── randomforest_model.pkl
│   ├── label_encoder.pkl
│   └── meta.pkl
└── utils/
    ├── train_models.py     # Training pipeline
    └── predictor.py        # Inference logic
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python utils/train_models.py
```

This trains all three classifiers, prints cross-validation accuracy, and saves artefacts to `models/`.

### 3a. Launch the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

### 3b. Use the CLI

```bash
# Interactive prompt
python cli.py

# Direct prediction
python cli.py --symptoms "Itching,Skin Rash,Nodal Skin Eruptions"
```

---

## How It Works

1. **Pre-processing** — symptoms are one-hot encoded into a 132-dimensional binary vector.
2. **Inference** — all three models predict independently.
3. **Ensemble** — the final disease label is chosen by majority vote; ties are broken by the Random Forest prediction.
4. **Confidence** — the Random Forest's maximum class probability is shown as a confidence percentage.

### Models

| Model | Description |
|---|---|
| SVM (RBF kernel) | Good generalisation on high-dimensional sparse data |
| Gaussian Naive Bayes | Fast baseline; works well with binary features |
| Random Forest (200 trees) | Provides feature-importance insight and probability estimates |

---

## Dataset

The dataset contains 4 920 labelled samples across 41 diseases and 132 binary symptom features.  
Source: Publicly available symptom-disease dataset (Kaggle — disease prediction using machine learning).

### Diseases Covered

| | | | | |
|---|---|---|---|---|
| Fungal infection | AIDS | Acne | Alcoholic hepatitis | Allergy |
| Arthritis | Bronchial Asthma | Cervical spondylosis | Chicken pox | Chronic cholestasis |
| Common Cold | Dengue | Diabetes | Dimorphic haemorrhoids | Drug Reaction |
| GERD | Gastroenteritis | Heart attack | Hepatitis A/B/C/D/E | Hypertension |
| Hyperthyroidism | Hypoglycaemia | Hypothyroidism | Impetigo | Jaundice |
| Malaria | Migraine | Osteoarthritis | Paralysis | Peptic ulcer |
| Pneumonia | Psoriasis | Tuberculosis | Typhoid | UTI |
| Varicose veins | Vertigo | | | |

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — SVM, Naive Bayes, Random Forest, cross-validation
- **Streamlit** — interactive web UI
- **pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — visualisations

---

## Disclaimer

This project is built for **educational and portfolio purposes only**.  
It does **not** constitute medical advice. Always consult a qualified healthcare professional for any medical concerns.
