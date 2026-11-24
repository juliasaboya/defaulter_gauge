# Defaulter Gauge

Sistema de previsão de risco de inadimplência financeira utilizando o algoritmo Naive Bayes. A interface estima a probabilidade de inadimplência no mês seguinte com base no limite do cartão, nas faturas, nos pagamentos e no atraso observado nos três últimos meses.

## Executar localmente

Use Python 3.11. Os comandos a seguir são para macOS ou Linux. No Windows, use `py -3.11 -m venv .venv` e ative com `.venv\Scripts\Activate.ps1`.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m defaulter_gauge.data
python -m defaulter_gauge.train
python -m pytest -q
python -m defaulter_gauge.app
```

Abra [Defaulter Gauge](http://127.0.0.1:8080/). Se o ambiente e o modelo já estiverem prontos, basta executar `.venv/bin/python -m defaulter_gauge.app`. A instalação e o download da base exigem internet. O treino e a previsão usam arquivos locais.

## Dados e método

A base **Default of Credit Card Clients**, de I-Cheng Yeh, reúne 30.000 clientes de Taiwan, com histórico de abril a setembro de 2005 e 23 variáveis preditoras originais. A classe 1 indica inadimplência no mês seguinte (6.636 casos). A classe 0 indica ausência desse evento (23.364 casos). Uma fatura em aberto ou um pagamento parcial pode fazer parte do histórico de entrada. O evento previsto é o rótulo de inadimplência do mês seguinte. [Dataset UCI e licença CC BY 4.0](https://doi.org/10.24432/C55S3H)

O script baixa o ZIP oficial, lê o XLS com xlrd, identifica o cabeçalho pelos nomes das colunas e normaliza o nome do alvo no CSV. Se o ZIP falhar, usa a biblioteca oficial `ucimlrepo`. O CSV validado fica em `data/credit_card_clients.csv`.

O modelo utiliza o limite do cartão e os três meses mais recentes da base, de julho a setembro. O limite sofre transformação logarítmica. Os valores de fatura e pagamento são divididos pelo limite, truncados ao intervalo de −10 a 10 e transformados com arco-seno hiperbólico. Os códigos de status −2, −1 e 0 entram como zero meses de atraso. O formulário pede dólares taiwaneses (NT$), admite faturas negativas para saldo credor e usa a mesma transformação do treino. ID, dados demográficos e meses anteriores a julho ficam fora do modelo.

A divisão estratificada com semente 42 reserva 24.000 registros para treino e 6.000 para teste. O Gaussian Naive Bayes usa calibração sigmoide por validação cruzada estratificada de cinco partes dentro do treino. O teste avalia o Naive Bayes, uma regressão logística e um modelo constante que usa a prevalência da classe positiva no treino. As métricas de classificação usam limiar de 0,5. A interface mostra a probabilidade estimada.

| Modelo | ROC-AUC | Precisão média | Brier | Sensibilidade a 0,5 |
| --- | ---: | ---: | ---: | ---: |
| Gaussian Naive Bayes calibrado | 0,720 | 0,477 | 0,145 | 43,3% |
| Regressão logística | 0,741 | 0,507 | 0,142 | 32,8% |
| Prevalência do treino | 0,500 | 0,221 | 0,172 | 0,0% |

O Naive Bayes classificou 575 dos 1.327 casos positivos como inadimplência no teste. A validação mede desempenho em clientes da mesma base histórica. A aplicação tem finalidade acadêmica e sua validade para outros países, períodos e populações precisaria ser estudada antes de qualquer uso em decisões de crédito.

O treino gera `artifacts/model.joblib` e `artifacts/metrics.json`. Carregue apenas artefatos Joblib gerados pelo próprio projeto. O ambiente, os dados e os arquivos gerados ficam fora do Git.
