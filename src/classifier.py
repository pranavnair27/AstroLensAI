"""
===========================================================
AstroLens AI
Classifier Module
===========================================================
Random Forest Classifier
===========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from config import *

from src.feature_extraction import feature_vector


# ==========================================================
# Training
# ==========================================================

def train_classifier():

    print("\nLoading Kepler Dataset...")

    df = pd.read_csv(KEPLER_DATASET)

    df = df[df["koi_disposition"].isin(
        ["CONFIRMED", "FALSE POSITIVE"]
    )]

    features = [

        "koi_period",

        "koi_duration",

        "koi_depth",

        "koi_model_snr"

    ]

    df = df.dropna(subset=features)

    X = df[features]

    y = df["koi_disposition"].map({

        "CONFIRMED": 1,

        "FALSE POSITIVE": 0

    })

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        min_samples_split=MIN_SAMPLES_SPLIT,

        random_state=RANDOM_STATE

    )

    print("Training Random Forest...")

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(X_test)

    print("\nAccuracy")

    print(

        accuracy_score(

            y_test,

            predictions

        )

    )

    print("\nClassification Report")

    print(

        classification_report(

            y_test,

            predictions

        )

    )

    print("\nConfusion Matrix")

    print(

        confusion_matrix(

            y_test,

            predictions

        )

    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(

        model,

        MODEL_PATH

    )

    joblib.dump(

        scaler,

        SCALER_PATH

    )

    print("\nModel Saved.")

    return model, scaler


# ==========================================================
# Load Model
# ==========================================================

def load_classifier():

    if not os.path.exists(MODEL_PATH):

        print("\nNo trained model found.")

        print("Training model...")

        return train_classifier()

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ==========================================================
# Prediction
# ==========================================================

def classify_candidate(

        model,

        scaler,

        features

):

    vector = np.array([

        [

            features["period"],

            features["duration"],

            features["depth"],

            features["snr"]

        ]

    ])

    vector = scaler.transform(vector)

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = np.max(probability)

    if prediction == 1:

        label = "EXOPLANET"

    else:

        label = "FALSE POSITIVE"

    return label, confidence


# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance(model):

    names = [

        "Period",

        "Duration",

        "Depth",

        "SNR"

    ]

    importance = model.feature_importances_

    print("\nFeature Importance\n")

    for n, i in zip(names, importance):

        print(

            f"{n:15s}"

            f"{i:.3f}"

        )


# ==========================================================
# Standalone
# ==========================================================

if __name__ == "__main__":

    train_classifier()