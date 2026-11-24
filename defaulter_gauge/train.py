"""Treina e avalia os modelos sobre uma divisão fixa da base UCI."""
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DATA, TARGET, validate
from .model import ARTIFACT, METRICS, features

SEED = 42
THRESHOLD = 0.5


def evaluate(y, probability) -> dict:
    predicted = probability >= THRESHOLD
    return {
        'roc_auc': roc_auc_score(y, probability),
        'average_precision': average_precision_score(y, probability),
        'brier': brier_score_loss(y, probability),
        'accuracy': accuracy_score(y, predicted),
        'precision': precision_score(y, predicted, zero_division=0),
        'recall': recall_score(y, predicted, zero_division=0),
        'f1': f1_score(y, predicted, zero_division=0),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def train() -> dict:
    frame = validate(pd.read_csv(DATA))
    train_rows, test_rows = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=SEED
    )
    x_train, x_test = features(train_rows), features(test_rows)
    y_train, y_test = train_rows[TARGET], test_rows[TARGET]
    models = {
        'naive_bayes': CalibratedClassifierCV(
            make_pipeline(StandardScaler(), GaussianNB()),
            method='sigmoid',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        ),
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        'prior_baseline': DummyClassifier(strategy='prior'),
    }
    results = {
        'train_rows': len(train_rows),
        'test_rows': len(test_rows),
        'test_defaults': int(y_test.sum()),
        'threshold': THRESHOLD,
        'models': {},
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        results['models'][name] = evaluate(y_test, model.predict_proba(x_test)[:, 1])

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models['naive_bayes'], ARTIFACT)
    METRICS.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return results


if __name__ == '__main__':
    if not DATA.is_file():
        raise SystemExit('CSV ausente. Execute python -m defaulter_gauge.data primeiro.')
    print(json.dumps(train(), indent=2))
