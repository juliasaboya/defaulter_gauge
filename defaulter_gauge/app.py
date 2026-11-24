"""Interface local do Defaulter Gauge."""
import json
from functools import lru_cache

import joblib
from nicegui import ui

from .model import ARTIFACT, BILLS, METRICS, PAYMENTS, STATUS, predict

DELAYS = {
    0: 'Sem atraso',
    **{month: f'{month} mês' if month == 1 else f'{month} meses' for month in range(1, 9)},
    9: '9 meses ou mais',
}

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


def index():
    ui.add_head_html(STYLE)
    ui.colors(primary='#146c5c')
    with ui.column().classes('w-full max-w-6xl mx-auto gap-5 px-2 py-5'):
        with ui.column().classes('hero gap-2'):
            ui.label('Defaulter Gauge').classes('text-4xl font-bold')
            ui.label('Estime o risco de inadimplência no próximo mês usando o histórico recente do cartão.')

        with ui.column().classes('panel gap-4'):
            ui.label('Histórico de três meses').classes('text-2xl font-bold')
            ui.label('Informe todos os valores em dólares taiwaneses (NT$), a moeda da base de treinamento.')
            fields = {'LIMIT_BAL': ui.number('Limite do cartão (NT$)', format='%.2f').props('outlined')}
            with ui.element('div').classes('months'):
                for label, status, bill, paid in zip(
                    ['Mais antigo', 'Intermediário', 'Mais recente'], STATUS, BILLS, PAYMENTS
                ):
                    with ui.column().classes('month gap-3'):
                        ui.label(label).classes('font-bold text-base')
                        fields[bill] = ui.number('Valor da fatura', format='%.2f').props('outlined').classes('w-full')
                        fields[paid] = ui.number('Valor pago', format='%.2f').props('outlined').classes('w-full')
                        fields[status] = ui.select(DELAYS, label='Atraso naquele mês').props('outlined').classes('w-full')
            for name, field in fields.items():
                field.mark(name)
            ui.label('Informe zero se não houve pagamento. Use fatura negativa para saldo credor.').classes('text-sm')

            def calculate():
                result.set_visibility(False)
                error.set_text('')
                values = {key: element.value for key, element in fields.items()}
                if any(value is None for value in values.values()):
                    error.set_text('Preencha todos os campos antes de calcular.')
                    return
                try:
                    probability = predict(load_model(), values)
                except (TypeError, ValueError) as exc:
                    error.set_text(str(exc))
                    return
                percentage.set_text(f'{probability:.1%}'.replace('.', ','))
                gauge.set_value(probability)
                result.set_visibility(True)

            def clear():
                for field in fields.values():
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
            ui.label('Experimento acadêmico com Gaussian Naive Bayes calibrado. A base reúne clientes de cartão de Taiwan em 2005. As estimativas se referem a esse contexto e servem à demonstração acadêmica.')
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
