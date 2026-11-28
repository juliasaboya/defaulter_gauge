"""Compara modelos Naive Bayes com seleção restrita ao desenvolvimento."""
import json
from itertools import combinations
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DATA, INPUTS, TARGET, validate
from .estimators import IndicatorBins, SelectIndicators
from .features import GROUPS
from .model import ARTIFACT, LEGACY, METRICS, THRESHOLD, classify, features

SEED = 42


def evaluate(y, probability) -> dict:
    predicted = classify(probability)
    return {
        'roc_auc': roc_auc_score(y, probability),
        'average_precision': average_precision_score(y, probability),
        'brier': brier_score_loss(y, probability),
        'accuracy': accuracy_score(y, predicted),
        'balanced_accuracy': balanced_accuracy_score(y, predicted),
        'precision': precision_score(y, predicted, zero_division=0),
        'recall': recall_score(y, predicted, zero_division=0),
        'f1': f1_score(y, predicted, zero_division=0),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def folds():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def candidate(groups, family, parameters):
    columns = [column for group in groups for column in GROUPS[group]]
    select = SelectIndicators(columns)
    if family == 'gaussian':
        steps = (StandardScaler(), GaussianNB(**parameters))
    else:
        steps = (
            IndicatorBins(n_bins=parameters['n_bins']),
            CategoricalNB(alpha=parameters['alpha'], min_categories=max(10, parameters['n_bins'])),
        )
    return CalibratedClassifierCV(
        make_pipeline(select, *steps),
        method='sigmoid', cv=folds(), ensemble=False,
    )


def rank(experiment):
    return (
        experiment['development_average_precision_mean'], experiment['development_roc_auc_mean'],
        -experiment['development_brier_mean'],
    )


def explore(x, y, no_delay):
    settings = {
        'gaussian': [{'var_smoothing': value} for value in (1e-9, 0.01)],
        'categorical': [
            {'n_bins': bins, 'alpha': alpha}
            for bins in (5, 10) for alpha in (0.5, 2.0)
        ],
    }
    partitions = list(folds().split(x, y))
    experiments = []
    for size in range(1, len(GROUPS) + 1):
        for groups in combinations(GROUPS, size):
            for family, options in settings.items():
                for parameters in options:
                    scores = []
                    cohort_scores = []
                    for train_index, validation_index in partitions:
                        model = candidate(groups, family, parameters)
                        model.fit(x.iloc[train_index], y.iloc[train_index])
                        probability = model.predict_proba(x.iloc[validation_index])[:, 1]
                        scores.append(evaluate(y.iloc[validation_index], probability))
                        cohort = no_delay.iloc[validation_index].to_numpy()
                        cohort_scores.append(evaluate(y.iloc[validation_index][cohort], probability[cohort]))
                    experiments.append({
                        'groups': groups, 'family': family, 'parameters': parameters,
                        **{
                            f'development_{metric}_mean': float(np.mean([score[metric] for score in scores]))
                            for metric in ('average_precision', 'roc_auc', 'brier', 'accuracy')
                        },
                        'development_ap_std': float(np.std([score['average_precision'] for score in scores])),
                        'development_no_delay_ap_mean': float(np.mean([
                            score['average_precision'] for score in cohort_scores
                        ])),
                    })
            print(f'Validação concluída: {" + ".join(groups)}', flush=True)
    return sorted(experiments, key=rank, reverse=True)


def legacy_features(frame):
    """Reproduz a transformação histórica usada pelo artefato arquivado."""
    raw = frame.loc[:, INPUTS]
    limit = raw['LIMIT_BAL']
    result = pd.DataFrame({'log_limite': np.log(limit)})
    for month, (status, bill, paid) in enumerate(zip(INPUTS[1:4], INPUTS[4:7], INPUTS[7:]), 1):
        result[f'atraso_{month}'] = raw[status].clip(lower=0)
        result[f'fatura_{month}'] = np.arcsinh(raw[bill].div(limit).clip(-10, 10))
        result[f'pagamento_{month}'] = np.arcsinh(raw[paid].div(limit).clip(-10, 10))
    return result


def compare_legacy(train_rows, test_rows, y_train):
    x_train, x_test = legacy_features(train_rows), legacy_features(test_rows)
    path = LEGACY / 'model.joblib'
    if not path.is_file():
        path = ARTIFACT.with_name('model.joblib')
    if path.is_file():
        legacy = joblib.load(path)
    else:
        legacy = CalibratedClassifierCV(
            make_pipeline(StandardScaler(), GaussianNB()), method='sigmoid', cv=folds(),
        ).fit(x_train, y_train)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x_train, y_train)
    return {
        'legacy_naive_bayes': legacy.predict_proba(x_test)[:, 1],
        'logistic_regression': logistic.predict_proba(x_test)[:, 1],
    }


def save(path, write):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = type(path)(handle.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train() -> dict:
    frame = validate(pd.read_csv(DATA))
    development, test = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=SEED,
    )
    x_train, y_train = features(development), development[TARGET]
    no_delay_train = x_train['meses_com_atraso'].eq(0)
    experiments = explore(x_train, y_train, no_delay_train)

    winner = experiments[0]
    finalists = {
        'naive_bayes': winner,
        'alternative_family': next(run for run in experiments if run['family'] != winner['family']),
        'financial_only': next(run for run in experiments if 'atrasos' not in run['groups']),
        'financial_no_delay': max(
            (run for run in experiments if 'atrasos' not in run['groups']),
            key=lambda run: run['development_no_delay_ap_mean'],
        ),
    }
    fitted = {}
    selections = {}
    for name, run in finalists.items():
        model = candidate(run['groups'], run['family'], run['parameters']).fit(x_train, y_train)
        selector = model.calibrated_classifiers_[0].estimator.named_steps['selectindicators']
        selections[name] = {
            **run,
            'features': selector.selected_,
            'conditional_redundancies': selector.redundant_,
            'univariate_f': selector.relevance_,
        }
        fitted[name] = model

    x_test, y_test = features(test), test[TARGET]
    no_delay_test = x_test['meses_com_atraso'].eq(0)
    probabilities = {name: model.predict_proba(x_test)[:, 1] for name, model in fitted.items()}
    probabilities.update(compare_legacy(development, test, y_train))
    prior = DummyClassifier(strategy='prior').fit(x_train, y_train)
    probabilities['prior_baseline'] = prior.predict_proba(x_test)[:, 1]
    cohort_prior = np.full(no_delay_test.sum(), y_train[no_delay_train].mean())
    results = {
        'train_rows': len(development),
        'test_rows': len(test),
        'test_defaults': int(y_test.sum()),
        'threshold': THRESHOLD,
        'selected': selections,
        'development_experiments': experiments,
        'models': {name: evaluate(y_test, probability) for name, probability in probabilities.items()},
        'no_delay': {
            'train_rows': int(no_delay_train.sum()),
            'train_defaults': int(y_train[no_delay_train].sum()),
            'test_rows': int(no_delay_test.sum()),
            'test_defaults': int(y_test[no_delay_test].sum()),
            'models': {
                **{
                    name: evaluate(y_test[no_delay_test], probability[no_delay_test])
                    for name, probability in probabilities.items()
                },
                'cohort_prior': evaluate(y_test[no_delay_test], cohort_prior),
            },
        },
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    save(ARTIFACT, lambda path: joblib.dump(fitted['naive_bayes'], path))
    save(METRICS, lambda path: path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    ))
    return results


if __name__ == '__main__':
    if not DATA.is_file():
        raise SystemExit('CSV ausente. Execute python -m defaulter_gauge.data primeiro.')
    results = train()
    print(json.dumps({key: results[key] for key in ('models', 'no_delay')}, indent=2))
