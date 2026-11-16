"""TabPFN binary classification; model is loaded lazily and reused per process."""
from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'customers.csv'
FEATURES = ['tenure_months', 'monthly_charges', 'support_tickets',
            'contract_type', 'payment_method', 'auto_pay']
CONTRACTS = ['month_to_month', 'one_year', 'two_year']
PAYMENTS = ['card', 'bank_transfer', 'electronic_check']


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df.empty or df[FEATURES + ['churn']].isna().any().any():
        raise ValueError('Dataset must be nonempty with no missing values')
    return df


def encode(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    encoded = pd.get_dummies(df[FEATURES], columns=['contract_type', 'payment_method'], dtype=float)
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded.astype('float32')


@lru_cache(maxsize=1)
def get_model():
    # Import lazily so unit tests can run without downloading TabPFN weights.
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    df = load_data()
    X = encode(df)
    y = df['churn'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # FAST variant for an approachable CPU-oriented demo. Requires TabPFN license acceptance.
    model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predicted = (probability >= 0.5).astype(int)
    metrics = {
        'test_rows': int(len(X_test)),
        'accuracy': round(float(accuracy_score(y_test, predicted)), 3),
        'precision': round(float(precision_score(y_test, predicted, zero_division=0)), 3),
        'recall': round(float(recall_score(y_test, predicted, zero_division=0)), 3),
        'f1': round(float(f1_score(y_test, predicted, zero_division=0)), 3),
        'roc_auc': round(float(roc_auc_score(y_test, probability)), 3),
    }
    return model, list(X.columns), metrics


def predict(record: dict) -> float:
    model, columns, _ = get_model()
    X_new = encode(pd.DataFrame([record]), columns)
    return float(model.predict_proba(X_new)[:, list(model.classes_).index(1)][0])
