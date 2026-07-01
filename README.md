# AstroLens AI
### P7: AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

AstroLens AI is an end-to-end machine learning pipeline for detecting and classifying exoplanet transit signals from noisy astronomical light curves. The project combines astrophysical signal processing with machine learning to identify potential exoplanets, estimate their orbital parameters, and provide an explainable confidence score.
Developed by Team Pratyagra for the Hack2Skill Bharatiya Antariksh Hackathon 2026.

---

# Features

- Automatic TESS light curve download
- Light curve preprocessing and normalization
- Transit detection using Box Least Squares (BLS)
- Feature extraction
- Random Forest based classification
- Explainable confidence scoring
- Automatic visualization generation
- Report generation
- CSV export of predictions
- Modular and extensible architecture

---

# Project Structure

```
AstroLensAI/
│
├── main.py
├── train.py
├── predict.py
├── config.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── downloader.py
│   ├── preprocessing.py
│   ├── transit_detection.py
│   ├── feature_extraction.py
│   ├── classifier.py
│   ├── confidence.py
│   ├── visualization.py
│   ├── report.py
│   └── utils.py
│
├── data/
│   └── kepler.csv
│
├── models/
│
├── outputs/
│   ├── figures/
│   └── reports/
│
└── tests/
```

---

# Pipeline

```
TESS Light Curve

        │

        ▼

Download Data

        │

        ▼

Preprocessing

        │

        ▼

Transit Detection (BLS)

        │

        ▼

Feature Extraction

        │

        ▼

Machine Learning Classification

        │

        ▼

Confidence Estimation

        │

        ▼

Visualization + Report
```

---

# Dataset

Training is performed using the Kepler Exoplanet Archive dataset.

Expected location:

```
data/
    kepler.csv
```

Required label column:

```
koi_disposition
```

Classes:

- CONFIRMED
- FALSE POSITIVE

---

# Machine Learning Model

Current classifier:

- Random Forest Classifier

Input features:

- Orbital Period
- Transit Duration
- Transit Depth
- Signal-to-Noise Ratio (SNR)

Output:

- Exoplanet
- False Positive

The trained model and feature scaler are automatically saved in the `models/` directory.

---

# Confidence Engine

AstroLens AI introduces an explainable confidence scoring system.

The final confidence score combines:

| Component | Weight |
|------------|--------|
| Machine Learning Probability | 40% |
| BLS Detection Strength | 25% |
| Signal-to-Noise Ratio | 20% |
| Data Quality | 10% |
| Transit Consistency | 5% |

Final confidence is reported as a percentage.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd AstroLensAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Place the Kepler dataset inside:

```
data/kepler.csv
```

Run:

```bash
python train.py
```

Expected output:

```
Loading Kepler Dataset...

Training Random Forest...

Accuracy ...

Model Saved.
```

Generated files:

```
models/
    classifier.pkl
    scaler.pkl
```

---

# Prediction

Run:

```bash
python predict.py
```

Example input:

```
25155310
```

Pipeline:

- Download TESS light curve
- Preprocess
- Detect transit
- Extract features
- Predict class
- Estimate confidence
- Generate report
- Save plots

---

# Generated Outputs

Figures:

```
outputs/figures/

lightcurve.png

folded_transit.png

bls_periodogram.png
```

Reports:

```
outputs/reports/

TIC_xxxxx_report.txt
```

CSV Summary:

```
outputs/results.csv
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Astropy
- Lightkurve
- Scikit-learn
- Matplotlib
- Joblib

---

# Novelty of AstroLens AI

Unlike traditional transit detection pipelines that rely solely on statistical detection, AstroLens AI integrates astrophysical analysis with explainable artificial intelligence.

Key innovations include:

- Hybrid Physics + AI pipeline
- Automated transit detection using Box Least Squares
- Machine learning based false-positive rejection
- Explainable confidence estimation
- Automated parameter estimation
- Publication-quality visualizations
- End-to-end automation from raw data to final report

The modular architecture also allows future integration of deep learning models such as CNNs, LSTMs, and Transformers without redesigning the pipeline.

---

# Future Improvements

- XGBoost classifier
- LightGBM classifier
- CNN-based light curve classification
- Transformer architecture
- Bayesian uncertainty estimation
- Multi-sector TESS analysis
- GPU acceleration
- Real-time transit monitoring

---

# Authors

AstroLens AI by Team Pratyagra
Pranav Rajeev Nair (VIT Vellore)
Advait Krishna Chodankar (VIT Vellore)
Vinay K. (VIT Vellore)

Developed as an AI-enabled pipeline for automated exoplanet detection using machine learning and astrophysical signal processing.

---

# License

This project is intended for academic and research purposes.
