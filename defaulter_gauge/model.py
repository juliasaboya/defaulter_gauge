"""Inferência sobre os indicadores comuns ao treinamento e ao formulário."""
import pandas as pd

from .data import BILLS, INPUTS, PAYMENTS, ROOT, STATUS, uci_history
from .features import indicators

ARTIFACT = ROOT / 'artifacts' / 'currency_model.joblib'
METRICS = ARTIFACT.with_name('currency_metrics.json')
LEGACY = ARTIFACT.parent / 'legacy'
THRESHOLD = 0.5


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return indicators(uci_history(frame))


def predict_history(model, history: pd.DataFrame) -> float:
    return float(model.predict_proba(indicators(history))[0, 1])


def predict(model, values: dict) -> float:
    return predict_history(model, uci_history(pd.DataFrame([values])))


def classify(probability):
    return probability >= THRESHOLD
