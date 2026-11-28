"""Interface local do Defaulter Gauge."""
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
from nicegui import ui

if __name__ == '__main__' and not __package__:
    # Permite executar este arquivo diretamente pelo IDE.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defaulter_gauge.data import BILLS, PAYMENTS, STATUS
from defaulter_gauge.model import ARTIFACT, METRICS, predict

DELAYS = {
    0: 'Sem atraso',
    **{month: f'{month} mês' if month == 1 else f'{month} meses' for month in range(1, 9)},
    9: '9 meses ou mais',
}
MONEY_PATTERN = re.compile(r'-?(?:[0-9]+|[0-9]{1,3}(?:\.[0-9]{3})+)(?:,[0-9]{1,2})?')
BRAZILIAN_SEPARATORS = str.maketrans(',.', '.,')

STYLE = """
<style>
body { background: #f0f5f4; color: #142d2e; }
.hero, .panel { border-radius: 22px; padding: 26px; width: 100%; }
.hero { background: #113b39; color: white; }
.panel { background: white; border: 1px solid #d8e6e2; }
.month { background: #f5f8f7; border-radius: 14px; padding: 14px; min-width: 0; }
.months { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; width: 100%; }
@media (max-width: 760px) { .months { grid-template-columns: 1fr; } }
</style>
"""


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(ARTIFACT)


def parse_money(value: str) -> float:
    value = value.strip()
    if not MONEY_PATTERN.fullmatch(value):
        raise ValueError('Use o formato brasileiro nos valores financeiros, como 1.234,56.')
    result = float(value.replace('.', '').replace(',', '.'))
    if not math.isfinite(result):
        raise ValueError('Preencha todos os campos com números finitos.')
    return result


def money_field(label: str) -> ui.input:
    field = ui.input(label, prefix='R$').props('outlined inputmode=decimal').classes('w-full')

    def format_on_blur():
        if not field.value.strip():
            return
        try:
            value = parse_money(field.value)
        except ValueError:
            return
        field.set_value(f'{value:,.2f}'.translate(BRAZILIAN_SEPARATORS))

    field.on('blur', format_on_blur)
    return field


def index():
    ui.add_head_html(STYLE)
    ui.colors(primary='#146c5c')
    with ui.column().classes('w-full max-w-6xl mx-auto gap-5 px-2 py-5'):
        with ui.column().classes('hero gap-2'):
            ui.label('Defaulter Gauge').classes('text-4xl font-bold')
            ui.label('Estime o risco de inadimplência no próximo mês usando o histórico recente do cartão.')

        with ui.column().classes('panel gap-4'):
            ui.label('Histórico de três meses').classes('text-2xl font-bold')
            ui.label('Informe os valores financeiros dos últimos três meses em reais (R$).')
            fields: dict[str, ui.input | ui.select] = {
                'LIMIT_BAL': money_field('Limite do cartão'),
            }
            with ui.element('div').classes('months'):
                for label, status, bill, paid in zip(
                    ['Há 3 meses', 'Há 2 meses', 'Último mês'], STATUS, BILLS, PAYMENTS
                ):
                    with ui.column().classes('month gap-3'):
                        ui.label(label).classes('font-bold text-base')
                        fields[bill] = money_field('Valor da fatura')
                        fields[paid] = money_field('Valor pago')
                        fields[status] = ui.select(DELAYS, label='Atraso naquele mês').props('outlined').classes('w-full')
            for name, field in fields.items():
                field.mark(name)
            ui.label(
                'Registre o valor pago no mês em que o pagamento foi feito. '
                'Informe zero se não houve pagamento e use fatura negativa para saldo credor.'
            ).classes('text-sm')

            def calculate():
                result.set_visibility(False)
                error.set_text('')
                if any(field.value is None or field.value == '' for field in fields.values()):
                    error.set_text('Preencha todos os campos antes de calcular.')
                    return
                try:
                    values = {
                        name: parse_money(field.value) if isinstance(field, ui.input) else field.value
                        for name, field in fields.items()
                    }
                    probability = predict(load_model(), values)
                except (TypeError, ValueError) as exc:
                    error.set_text(str(exc))
                    return
                percentage.set_text(f'{probability:.1%}'.replace('.', ','))
                gauge.set_value(probability)
                result.set_visibility(True)

            def clear():
                for field in fields.values():
                    if isinstance(field, ui.input):
                        field.set_value('')
                    else:
                        field.set_value(None)
                error.set_text('')
                result.set_visibility(False)

            with ui.row():
                ui.button('Calcular risco', on_click=calculate).props('unelevated').set_enabled(ARTIFACT.is_file())
                ui.button('Limpar', on_click=clear).props('flat')
            error = ui.label().classes('text-red-800').props('role=alert')
            if not ARTIFACT.is_file():
                error.set_text('Modelo ausente. Execute python -m defaulter_gauge.train e reinicie o aplicativo.')

        with ui.column().classes('panel gap-2').props('aria-live=polite') as result:
            ui.label('Probabilidade estimada de inadimplência').classes('text-lg font-semibold')
            percentage = ui.label().classes('text-5xl font-bold')
            gauge = ui.linear_progress(value=0, show_value=False).props(
                'size=14px rounded aria-label="Probabilidade estimada"'
            )
            with ui.row().classes('w-full justify-between text-sm'):
                ui.label('0%')
                ui.label('100%')
        result.set_visibility(False)

        with ui.expansion('Sobre o modelo e a avaliação').classes('panel'):
            ui.label(
                'Experimento acadêmico com Naive Bayes calibrado e indicadores proporcionais. '
                'A base reúne clientes de cartão de Taiwan em 2005. '
                'A aplicação ainda não tem validação externa para a população brasileira.'
            )
            if METRICS.is_file():
                summary = json.loads(METRICS.read_text(encoding='utf-8'))
                model_metrics = summary['models']['naive_bayes']
                ui.label(
                    f"Teste com {summary['test_rows']:,} clientes. "
                    f"ROC-AUC {model_metrics['roc_auc']:.3f} e Brier {model_metrics['brier']:.3f}."
                )
            ui.link('Default of Credit Card Clients · UCI', 'https://doi.org/10.24432/C55S3H', new_tab=True)


if __name__ == '__main__':
    ui.run(index, title='Defaulter Gauge', host='127.0.0.1', port=8080, reload=False, show=False)
