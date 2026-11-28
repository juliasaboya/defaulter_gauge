import asyncio
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from nicegui.testing import user_simulation

from defaulter_gauge.app import index, parse_money
from defaulter_gauge.data import DATA, TARGET, validate
from defaulter_gauge.estimators import IndicatorBins
from defaulter_gauge.model import (
    ARTIFACT, BILLS, INPUTS, METRICS, PAYMENTS, STATUS, classify, features, predict,
)


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
    assert row['utilizacao_1'] == -0.2
    assert row['atraso_1'] == 0
    assert row['atraso_3'] == 3


def scenario(limit, bills, payments, delays):
    return dict(zip(INPUTS, (limit, *delays, *bills, *payments)))


def test_payment_is_matched_to_previous_statement():
    values = scenario(1000, [400, 600, 800], [900, 100, 900], [1, 0, 1])
    row = features(pd.DataFrame([values])).iloc[0]
    assert row['pagamento_limite_1'] == 0.9
    assert row['cobertura_fatura_1'] == 0.25
    assert row['cobertura_fatura_2'] == 1.5
    assert row['cobertura_media'] == 0.875
    assert row['variacao_fatura_12'] == 0.2
    assert row['meses_com_atraso'] == 2
    assert row['maior_sequencia_atraso'] == 1


def test_categorical_binning_preserves_rare_delays_and_nonzero_ratios():
    frame = pd.DataFrame({
        'atraso_1': [0] * 100 + [1, 3],
        'utilizacao_1': [0] * 96 + [1, 2, 3, 4, 5, 6],
    })
    bins = IndicatorBins(n_bins=5).fit(frame)
    actual = bins.transform(pd.DataFrame({'atraso_1': [0, 1, 9], 'utilizacao_1': [0, 1, 9]}))
    np.testing.assert_array_equal(actual, [[0, 0], [1, 1], [9, 1]])


@pytest.fixture(scope='module')
def trained_model():
    return joblib.load(ARTIFACT)


@pytest.mark.parametrize('factor', [0.01, 0.1, 0.5, 2, 6, 10, 100])
@pytest.mark.parametrize('values', [
    scenario(5000, [4000] * 3, [1000] * 3, [0, 0, 0]),
    scenario(5000, [-2000, 0, -1], [0, 100, 0], [-2, 1, 2]),
    scenario(5000, [3000, 10000, 2000], [0, 6000, 15000], [1, 2, 9]),
    scenario(5000, [0, 0, 0], [0, 0, 0], [0, 0, 0]),
    scenario(1e-200, [-1e200, 1e200, 1e-200], [1e-200, 0, 1e200], [0, 1, 0]),
    scenario(1e-250, [1e-250, -5e-252, 2e-251], [2e-251, 0, 1e-251], [0, 0, 1]),
])
def test_currency_invariance(trained_model, values, factor):
    scaled = {key: value if key in STATUS else value * factor for key, value in values.items()}
    original = features(pd.DataFrame([values]))
    rescaled = features(pd.DataFrame([scaled]))
    assert np.isfinite(original.to_numpy()).all()
    np.testing.assert_allclose(original, rescaled, rtol=1e-12, atol=1e-12)
    probability = predict(trained_model, values)
    rescaled_probability = predict(trained_model, scaled)
    assert probability == pytest.approx(rescaled_probability, abs=1e-12, rel=1e-12)
    assert classify(probability) == classify(rescaled_probability)


def test_currency_invariance_across_uci_dataset(trained_model, dataset):
    values = dataset.loc[:, INPUTS]
    monetary = ['LIMIT_BAL', *BILLS, *PAYMENTS]
    original = features(values)
    probability = trained_model.predict_proba(original)[:, 1]
    for factor in (0.01, 0.1, 0.5, 2, 6, 10, 100):
        scaled = values.copy()
        scaled[monetary] *= factor
        rescaled = features(scaled)
        rescaled_probability = trained_model.predict_proba(rescaled)[:, 1]
        np.testing.assert_allclose(original, rescaled, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(probability, rescaled_probability, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(classify(probability), classify(rescaled_probability))


@pytest.mark.parametrize(('value', 'expected'), [
    ('1.234,56', 1234.56), ('-1.234,56', -1234.56), ('0', 0), ('1234,5', 1234.5),
])
def test_brazilian_money(value, expected):
    assert parse_money(value) == expected


@pytest.mark.parametrize('value', ['1,234.56', '1.00', 'Infinity', 'NaN', '1e6', '9' * 400])
def test_invalid_money(value):
    with pytest.raises(ValueError):
        parse_money(value)


@pytest.mark.parametrize(('field', 'invalid'), [
    ('LIMIT_BAL', 0), ('LIMIT_BAL', float('inf')), (BILLS[0], float('-inf')),
    (PAYMENTS[0], -1), (PAYMENTS[2], float('nan')), (STATUS[0], 10),
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
            await user.should_see('Informe os valores financeiros dos últimos três meses em reais (R$).')
            for label in ('Há 3 meses', 'Há 2 meses', 'Último mês', 'Limite do cartão'):
                await user.should_see(label)
            user.find('Calcular risco').click()
            await user.should_see('Preencha todos os campos antes de calcular.')
            for name, value in record.items():
                widget = next(iter(user.find(marker=name).elements))
                widget.set_value(max(0, int(value)) if name in STATUS else f'{value:.2f}'.replace('.', ','))
            limit_field = user.find(marker='LIMIT_BAL')
            limit = next(iter(limit_field.elements))
            assert limit.props['prefix'] == 'R$'
            limit.set_value('1234,5')
            limit_field.trigger('blur')
            assert limit.value == '1.234,50'
            limit.set_value('1,234.56')
            user.find('Calcular risco').click()
            await user.should_see('Use o formato brasileiro nos valores financeiros, como 1.234,56.')
            limit.set_value(f"{record['LIMIT_BAL']:,.2f}".translate(str.maketrans(',.', '.,')))
            user.find('Calcular risco').click()
            await user.should_see('Probabilidade estimada de inadimplência')
            expected = predict(joblib.load(ARTIFACT), record)
            await user.should_see(f'{expected:.1%}'.replace('.', ','))
            user.find('Limpar').click()
            await user.should_not_see('Probabilidade estimada de inadimplência')
            for name in INPUTS:
                field = next(iter(user.find(marker=name).elements))
                assert field.value == (None if name in STATUS else '')

    asyncio.run(simulate())
