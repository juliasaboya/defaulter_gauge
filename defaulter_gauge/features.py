"""Indicadores para um histórico de três meses em ordem cronológica."""
import numpy as np
import pandas as pd

MONTHS = (1, 2, 3)
LIMIT = 'limite'
BILL = tuple(f'fatura_mes_{month}' for month in MONTHS)
PAID = tuple(f'pagamento_mes_{month}' for month in MONTHS)
DELAY = tuple(f'atraso_mes_{month}' for month in MONTHS)
HISTORY = (LIMIT, *BILL, *PAID, *DELAY)
RATIO_CAP = 10.0

GROUPS = {
    'utilizacao': (
        *(f'utilizacao_{month}' for month in MONTHS),
        'utilizacao_media', 'utilizacao_maxima', 'utilizacao_desvio', 'utilizacao_tendencia',
    ),
    'pagamento': (
        *(f'pagamento_limite_{month}' for month in MONTHS),
        'cobertura_fatura_1', 'cobertura_fatura_2', 'cobertura_media', 'cobertura_desvio',
        'pagamento_limite_media', 'pagamento_limite_desvio', 'pagamento_limite_tendencia',
    ),
    'evolucao': (
        'variacao_fatura_12', 'variacao_fatura_23',
        'variacao_pagamento_12', 'variacao_pagamento_23',
        'pressao_financeira_tendencia',
    ),
    'atrasos': (
        *(f'atraso_{month}' for month in MONTHS),
        'meses_com_atraso', 'frequencia_atraso', 'maior_sequencia_atraso', 'atraso_recente',
    ),
}

def _ratio(numerator, denominator, *, cap=RATIO_CAP):
    # Divisão precede o corte. Nenhuma tolerância é aplicada em unidades monetárias.
    with np.errstate(over='ignore', under='ignore'):
        ratio = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0,
        )
    return ratio.clip(-cap, cap)


def _change(values):
    """Variação simétrica entre meses, limitada a [-1, 1] e definida em zero."""
    earlier, later = values[:, :-1], values[:, 1:]
    scale = np.maximum(np.abs(earlier), np.abs(later))
    with np.errstate(under='ignore'):
        a = np.divide(earlier, scale, out=np.zeros_like(earlier), where=scale > 0)
        b = np.divide(later, scale, out=np.zeros_like(later), where=scale > 0)
    return _ratio(b - a, np.abs(a) + np.abs(b), cap=1)


def indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Recebe pagamentos por mês de ocorrência, referentes à fatura anterior."""
    raw = history.loc[:, HISTORY].apply(pd.to_numeric, errors='raise').astype(float)
    limit = raw[LIMIT].to_numpy()[:, None]
    bill = raw.loc[:, BILL].to_numpy()
    paid = raw.loc[:, PAID].to_numpy()
    delay = raw.loc[:, DELAY].to_numpy()
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Preencha todos os campos com números finitos.')
    if (limit <= 0).any():
        raise ValueError('O limite precisa ser maior que zero.')
    if (paid < 0).any():
        raise ValueError('Os pagamentos precisam ser iguais ou maiores que zero.')
    if not np.isin(delay, np.arange(-2, 10)).all():
        raise ValueError('Escolha um status de pagamento para cada mês.')

    utilization = _ratio(bill, limit)
    payment = _ratio(paid, limit)
    coverage = _ratio(paid[:, 1:], bill[:, :-1])
    has_debt = bill[:, :-1] > 0
    count = has_debt.sum(axis=1)
    coverage_mean = _ratio(coverage.sum(axis=1), count)
    squared_error = np.where(has_debt, (coverage - coverage_mean[:, None]) ** 2, 0)
    # A variância pode chegar a 100 antes da raiz.
    coverage_std = np.sqrt(_ratio(squared_error.sum(axis=1), count, cap=RATIO_CAP ** 2))

    bill_change, payment_change = _change(bill), _change(paid)
    # As duas exposições têm pares observados. Julho não tem fatura anterior na janela.
    pressure = utilization[:, :2] - payment[:, 1:]
    delay = delay.clip(min=0)
    overdue = delay > 0
    total = overdue.sum(axis=1)
    consecutive = np.where(
        overdue[:, 1] & (overdue[:, 0] | overdue[:, 2]), total, (total > 0).astype(int),
    )

    columns = {
        **{f'utilizacao_{m}': utilization[:, m - 1] for m in MONTHS},
        'utilizacao_media': utilization.mean(axis=1),
        'utilizacao_maxima': utilization.max(axis=1),
        'utilizacao_desvio': utilization.std(axis=1),
        'utilizacao_tendencia': (utilization[:, 2] - utilization[:, 0]) / 2,
        **{f'pagamento_limite_{m}': payment[:, m - 1] for m in MONTHS},
        # -1 identifica ausência de saldo devedor para este par.
        'cobertura_fatura_1': np.where(has_debt[:, 0], coverage[:, 0], -1),
        'cobertura_fatura_2': np.where(has_debt[:, 1], coverage[:, 1], -1),
        'cobertura_media': np.where(count > 0, coverage_mean, -1),
        'cobertura_desvio': coverage_std,
        'pagamento_limite_media': payment.mean(axis=1),
        'pagamento_limite_desvio': payment.std(axis=1),
        'pagamento_limite_tendencia': (payment[:, 2] - payment[:, 0]) / 2,
        'variacao_fatura_12': bill_change[:, 0],
        'variacao_fatura_23': bill_change[:, 1],
        'variacao_pagamento_12': payment_change[:, 0],
        'variacao_pagamento_23': payment_change[:, 1],
        'pressao_financeira_tendencia': pressure[:, 1] - pressure[:, 0],
        **{f'atraso_{m}': delay[:, m - 1] for m in MONTHS},
        'meses_com_atraso': total,
        'frequencia_atraso': total / 3,
        'maior_sequencia_atraso': consecutive,
        'atraso_recente': overdue[:, 2].astype(int),
    }
    # Estabiliza empates em cortes categóricos perante o arredondamento de ponto flutuante.
    return pd.DataFrame(columns, index=raw.index).round(12)
