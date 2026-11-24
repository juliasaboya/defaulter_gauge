"""Transformação compartilhada pelo treinamento e pela interface."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts' / 'model.joblib'
METRICS = ARTIFACT.with_name('metrics.json')
STATUS = ('PAY_3', 'PAY_2', 'PAY_0')
BILLS = ('BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1')
PAYMENTS = ('PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1')
INPUTS = ['LIMIT_BAL', *STATUS, *BILLS, *PAYMENTS]


def features(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame.loc[:, INPUTS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Preencha todos os campos com números válidos.')
    limit = raw['LIMIT_BAL']
    if (limit <= 0).any():
        raise ValueError('O limite precisa ser maior que zero.')
    if (raw[list(PAYMENTS)] < 0).any().any():
        raise ValueError('Os pagamentos precisam ser iguais ou maiores que zero.')
    if not raw[list(STATUS)].isin(range(-2, 10)).all().all():
        raise ValueError('Escolha um status de pagamento para cada mês.')

    result = pd.DataFrame(index=frame.index)
    # Códigos sem atraso são agrupados. Valores negativos da fatura são créditos.
    result['log_limite'] = np.log(limit)
    for month, (status, bill, paid) in enumerate(zip(STATUS, BILLS, PAYMENTS), 1):
        result[f'atraso_{month}'] = raw[status].clip(lower=0)
        for label, column in [('fatura', bill), ('pagamento', paid)]:
            with np.errstate(over='ignore'):
                ratio = raw[column].div(limit).clip(-10, 10)
            result[f'{label}_{month}'] = np.arcsinh(ratio)
    return result


def predict(model, values: dict) -> float:
    return float(model.predict_proba(features(pd.DataFrame([values])))[0, 1])
