import asyncio
import json

import joblib
import pandas as pd
import pytest
from nicegui.testing import user_simulation

from defaulter_gauge.app import index
from defaulter_gauge.data import DATA, TARGET, validate
from defaulter_gauge.model import ARTIFACT, BILLS, INPUTS, METRICS, PAYMENTS, STATUS, features, predict


@pytest.fixture(scope='module')
def dataset():
    return validate(pd.read_csv(DATA))


@pytest.fixture
def record(dataset):
    return dataset.loc[dataset.index[0], INPUTS].to_dict()


def test_dataset(dataset):
    assert dataset.shape == (30_000, 25)
    assert dataset[TARGET].value_counts().to_dict() == {0: 23_364, 1: 6_636}


def test_features_preserve_credits_and_month_order(record):
    record.update({'LIMIT_BAL': 1000, BILLS[0]: -200, STATUS[0]: -2, STATUS[-1]: 3})
    row = features(pd.DataFrame([record])).iloc[0]
    assert row['fatura_1'] < 0
    assert row['atraso_1'] == 0
    assert row['atraso_3'] == 3


@pytest.mark.parametrize(('field', 'invalid'), [
    ('LIMIT_BAL', 0), ('LIMIT_BAL', float('inf')), (PAYMENTS[0], -1), (STATUS[0], 10),
])
def test_invalid_inputs(record, field, invalid):
    record[field] = invalid
    with pytest.raises(ValueError):
        features(pd.DataFrame([record]))


def test_trained_artifact(record):
    probability = predict(joblib.load(ARTIFACT), record)
    assert 0 <= probability <= 1
    metrics = json.loads(METRICS.read_text())
    matrix = metrics['models']['naive_bayes']['confusion_matrix']
    assert sum(map(sum, matrix)) == metrics['test_rows'] == 6_000
    assert sum(matrix[1]) == metrics['test_defaults'] == 1_327


def test_form(record):
    async def simulate():
        async with user_simulation(index) as user:
            await user.open('/')
            user.find('Calcular risco').click()
            await user.should_see('Preencha todos os campos antes de calcular.')
            for name, value in record.items():
                widget = next(iter(user.find(marker=name).elements))
                widget.set_value(max(0, int(value)) if name in STATUS else float(value))
            user.find('Calcular risco').click()
            await user.should_see('Probabilidade estimada de inadimplência')
            expected = predict(joblib.load(ARTIFACT), record)
            await user.should_see(f'{expected:.1%}'.replace('.', ','))
            user.find('Limpar').click()
            await user.should_not_see('Probabilidade estimada de inadimplência')

    asyncio.run(simulate())
