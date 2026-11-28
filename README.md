# Defaulter Gauge

O Defaulter Gauge é uma demonstração acadêmica que estima a probabilidade de inadimplência no mês seguinte a partir do histórico recente do cartão. O modelo usa proporções financeiras e histórico de atrasos. A interface recebe os valores em reais, e o cálculo usa a mesma unidade em todos os campos monetários.

## Cartilha e glossário

Abra [docs/index.html](docs/index.html) no navegador para consultar a cartilha, buscar termos e filtrar por categoria. Para editar o conteúdo, abra esse mesmo arquivo no VS Code. As definições ficam na lista `id="termos"`. A cartilha funciona localmente, sem instalar dependências ou iniciar o aplicativo de previsão.

## Executar localmente

Use Python 3.11. Os comandos abaixo são para macOS ou Linux. No Windows, crie o ambiente com `py -3.11 -m venv .venv` e ative com `.venv\Scripts\Activate.ps1`.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m defaulter_gauge.data
python -m defaulter_gauge.train
python -m pytest -q
python -m defaulter_gauge.app
```

Com o ambiente e o modelo treinado já disponíveis, execute `.venv/bin/python -m defaulter_gauge.app` e abra [Defaulter Gauge](http://127.0.0.1:8080/). A instalação e o download exigem internet. Treinamento e inferência usam arquivos locais.

### Executar pelo IDE

Selecione o interpretador do ambiente virtual do projeto (`.venv/bin/python` no macOS e Linux ou `.venv\Scripts\python.exe` no Windows) e execute diretamente o arquivo `defaulter_gauge/app.py`. Você também pode iniciá-lo pelo terminal, na raiz do projeto e com o ambiente virtual ativo.

```sh
python defaulter_gauge/app.py
```

Se a aplicação exibir **Modelo ausente**, execute `python -m defaulter_gauge.train` e reinicie o app. Se o CSV estiver ausente, execute `python -m defaulter_gauge.data` antes do treino.

### Preencher o formulário

Informe o limite do cartão e, para **Há 3 meses**, **Há 2 meses** e **Último mês**, preencha a fatura, o pagamento e o atraso. Os sete campos monetários exibem `R$` e formatam valores válidos ao sair do campo, como `1.234,56`. Registre pagamentos pelo mês em que foram efetuados. Fatura negativa indica saldo credor. Use zero para um mês sem pagamento. A saída apresenta a probabilidade em porcentagem.

## Base e correspondência temporal

A base **Default of Credit Card Clients** contém 30.000 clientes de Taiwan, com histórico de abril a setembro de 2005 e 6.636 casos de inadimplência no mês seguinte. Usamos julho, agosto e setembro. ID, dados demográficos e informações anteriores a julho ficam fora dos atributos. O módulo `data.py` baixa o ZIP oficial, lê a planilha, valida a base e a salva em `data/credit_card_clients.csv`. A biblioteca `ucimlrepo` é a alternativa se o ZIP falhar. [Ficha e dicionário da UCI](https://doi.org/10.24432/C55S3H) · [ZIP oficial](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip)

A UCI descreve as séries como valores de fatura e valores do pagamento anterior, datados pelo mês em que foram pagos. Interpretamos o pagamento de agosto em relação à fatura de julho e o pagamento de setembro em relação à fatura de agosto. No desenvolvimento, houve 3.248 e 3.268 correspondências exatas de valores positivos nesses pares, respectivamente, contra 497 e 535 ao parear pelo mesmo mês. O pareamento é uma aproximação por ciclo. A base não vincula transações individuais a faturas. O pagamento de julho entra em pagamento/limite e nas tendências, pois a fatura de junho fica fora da janela. [Dicionário da UCI](https://doi.org/10.24432/C55S3H)

| Posição | Mês | Fatura | Pagamento | Atraso |
| --- | --- | --- | --- | --- |
| Há 3 meses | Julho | `BILL_AMT3` | `PAY_AMT3` | `PAY_3` |
| Há 2 meses | Agosto | `BILL_AMT2` | `PAY_AMT2` | `PAY_2` |
| Último mês | Setembro | `BILL_AMT1` | `PAY_AMT1` | `PAY_0` |

## Atributos e independência monetária

O Gaussian Naive Bayes anterior recebia dez atributos. `log_limite` dependia da escala monetária. `fatura_1…3` e `pagamento_1…3` já eram razões sobre o limite, truncadas e transformadas por arco-seno hiperbólico. `atraso_1…3` representavam meses de atraso com os códigos −2, −1 e 0 agrupados em zero.

O novo conjunto candidato remove `log_limite`, usa razões truncadas sem arco-seno hiperbólico e mantém a codificação mensal dos atrasos. Os demais indicadores candidatos são:

| Grupo | Indicadores avaliados |
| --- | --- |
| Utilização | Fatura/limite em cada mês, média, máximo, desvio padrão populacional e inclinação entre o primeiro e o terceiro mês. |
| Pagamento | Pagamento/limite por mês, média, desvio padrão populacional e inclinação. Pagamento/fatura anterior por par, média e desvio padrão dos pares com dívida. |
| Evolução | Variação simétrica das faturas e pagamentos em cada transição. Mudança da pressão financeira entre os pares observados, usando fatura/limite menos pagamento/limite. |
| Atrasos | Gravidade mensal, contagem, frequência, maior sequência de meses com atraso e atraso no mês mais recente. |

Para qualquer fator comum positivo \(k\), as razões satisfazem \((ka)/(kb)=a/b\). A variação simétrica \((b-a)/(|a|+|b|)\) preserva a mesma propriedade e vale zero se os dois montantes são zero. O cálculo reescala o par antes da subtração para evitar estouro. Médias, desvios e tendências operam sobre razões. Os códigos de atraso já são adimensionais.

O limite deve ser positivo e todos os valores devem ser finitos. A utilização preserva o sinal da fatura e valores superiores a 100%. Pagamentos maiores que a fatura também podem produzir coberturas acima de 100%. O corte de magnitudes extremas só se aplica às razões, em ±10. Uma fatura zero ou negativa recebe código de cobertura −1 e fica fora da média e do desvio da cobertura. Sem pares válidos, a média recebe −1 e o desvio vale zero. A etapa final arredonda os indicadores a 12 casas decimais para estabilizar a discretização.

## Método experimental

Mantivemos a divisão histórica estratificada de 24.000 clientes para desenvolvimento e 6.000 para teste, com semente 42, e o limiar fixo de 0,5. O teste original já havia sido usado para publicar e comparar as métricas do MVP. A comparação desta versão usa as mesmas pessoas, enquanto uma estimativa confirmatória exigiria outra amostra.

A busca usa cinco dobras estratificadas no desenvolvimento. Dentro de cada treino, uma validação cruzada interna de cinco dobras produz previsões para a calibração sigmoide. O classificador base final de cada candidato é reajustado com todo o conjunto de treino daquela etapa. A seleção descarta colunas constantes, ordena as demais pela estatística F univariada e remove colunas com Spearman absoluto ≥ 0,85 em pelo menos uma das classes diante de um indicador já retido. Isso reduz redundâncias observáveis. A independência condicional completa continua sendo uma hipótese do modelo.

Avaliamos os quatro grupos isolados e todas as suas combinações. Para cada combinação, testamos `GaussianNB` padronizado com `var_smoothing` de 1e−9 ou 0,01 e `CategoricalNB` com discretização ordinal de 5 ou 10 faixas e `alpha` de 0,5 ou 2. Os cortes dos atributos contínuos são ajustados dentro do treino. A gravidade do atraso mantém os níveis inteiros de 0 a 9, inclusive níveis raros ou ausentes em uma dobra. Foram 90 configurações. A seleção maximiza a average precision média das dobras externas de desenvolvimento, com ROC-AUC e Brier como desempates. A seleção também fixa o melhor candidato da outra família, o melhor candidato sem atrasos no desenvolvimento completo e o melhor candidato financeiro segundo a average precision no recorte de desenvolvimento sem atraso. A avaliação no teste vem depois dessas decisões. [Guia do Naive Bayes no scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html) · [Calibração de probabilidades](https://scikit-learn.org/stable/modules/calibration.html)

## Resultados no desenvolvimento

A melhor configuração foi o **CategoricalNB com calibração sigmoide, dez faixas e alpha 0,5**, usando utilização e atrasos. A AP média foi 0,4681, com desvio de 0,0188 entre dobras. Atrasos foi o melhor grupo isolado (0,4238). Entre os candidatos financeiros avaliados na população completa de desenvolvimento, evolução teve a maior AP (0,3751). Com os quatro grupos, a melhor AP foi 0,4458.

| Grupos | Melhor família | AP média | ROC-AUC média |
| --- | --- | ---: | ---: |
| Utilização + Atrasos | CategoricalNB | 0,4681 | 0,7366 |
| Evolução + Atrasos | CategoricalNB | 0,4645 | 0,7288 |
| Utilização + Evolução + Atrasos | CategoricalNB | 0,4607 | 0,7282 |
| Pagamento + Atrasos | GaussianNB | 0,4527 | 0,6983 |
| Pagamento + Evolução + Atrasos | GaussianNB | 0,4488 | 0,6984 |
| Utilização + Pagamento + Atrasos | GaussianNB | 0,4460 | 0,6942 |
| Utilização + Pagamento + Evolução + Atrasos | GaussianNB | 0,4458 | 0,6943 |
| Atrasos | GaussianNB | 0,4238 | 0,7136 |
| Evolução | CategoricalNB | 0,3751 | 0,6685 |
| Utilização + Evolução | CategoricalNB | 0,3693 | 0,6698 |
| Utilização + Pagamento + Evolução | CategoricalNB | 0,3648 | 0,6767 |
| Pagamento + Evolução | CategoricalNB | 0,3637 | 0,6731 |
| Utilização + Pagamento | CategoricalNB | 0,3408 | 0,6648 |
| Pagamento | CategoricalNB | 0,3324 | 0,6571 |
| Utilização | CategoricalNB | 0,2877 | 0,6072 |

O modelo selecionado usa cinco indicadores. `atraso_1` preserva a gravidade do atraso de julho, e `utilizacao_1` usa a fatura de julho sobre o limite. Os novos `maior_sequencia_atraso`, `utilizacao_desvio` e `utilizacao_tendencia` medem persistência do atraso, desvio padrão da utilização e sua inclinação ao longo dos três meses. Os indicadores de pagamento e evolução ficaram fora da combinação vencedora. A seleção também descartou a contagem e a frequência de atrasos, cuja correlação com a maior sequência chegou a 0,9998 dentro das classes.

## Resultados no teste histórico

Dos 6.000 clientes, 1.327 tiveram o desfecho positivo. A matriz segue a ordem `[[verdadeiros negativos, falsos positivos], [falsos negativos, verdadeiros positivos]]`.

| Modelo | Acurácia | Acurácia balanceada | Precision | Recall | F1 | ROC-AUC | AP | Brier | Matriz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GaussianNB legado | 0,7997 | 0,6685 | 0,5610 | 0,4333 | 0,4889 | 0,7200 | 0,4771 | 0,1453 | `[[4223, 450], [752, 575]]` |
| **CategoricalNB selecionado** | 0,7918 | 0,6155 | 0,5545 | 0,2992 | 0,3886 | 0,7320 | 0,4599 | 0,1501 | `[[4354, 319], [930, 397]]` |
| GaussianNB proporcional | 0,7850 | 0,6210 | 0,5223 | 0,3271 | 0,4022 | 0,6974 | 0,4428 | 0,1537 | `[[4276, 397], [893, 434]]` |
| CategoricalNB financeiro, melhor no total | 0,7777 | 0,5384 | 0,4882 | 0,1093 | 0,1786 | 0,6670 | 0,3635 | 0,1610 | `[[4521, 152], [1182, 145]]` |
| CategoricalNB financeiro, melhor no recorte | 0,7788 | 0,5000 | 0,0000 | 0,0000 | 0,0000 | 0,6778 | 0,3585 | 0,1599 | `[[4673, 0], [1327, 0]]` |
| Regressão logística histórica | 0,8153 | 0,6408 | 0,6682 | 0,3278 | 0,4398 | 0,7414 | 0,5074 | 0,1418 | `[[4457, 216], [892, 435]]` |
| Prevalência do desenvolvimento | 0,7788 | 0,5000 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,2212 | 0,1723 | `[[4673, 0], [1327, 0]]` |

A acurácia passou de 79,97% para 79,18%, abaixo da meta de 90%. A ROC-AUC subiu de 0,7200 para 0,7320. A AP caiu de 0,4771 para 0,4599, a precision passou de 56,10% para 55,45% e o recall caiu de 43,33% para 29,92%. A 0,5, o modelo identificou 178 inadimplências a menos e produziu 131 falsos positivos a menos. A acurácia balanceada, o F1 e o Brier também pioraram.

### Clientes sem atraso nos três meses

O recorte incluiu 17.199 clientes no desenvolvimento, com 2.136 inadimplências, e 4.361 no teste, com 564 inadimplências (12,93%). O candidato financeiro com os três grupos foi escolhido pela AP no recorte de desenvolvimento e treinado no desenvolvimento completo. No recorte de teste, obteve AP de 0,1878 e ROC-AUC de 0,5893, o que indica discriminação modesta frente à prevalência. A 0,5, ele e o modelo principal classificaram todos como negativos, com recall zero e 87,07% de acurácia. O histórico de três meses não permite estabelecer se a inadimplência observada foi a primeira da vida financeira.

| Modelo | ROC-AUC | AP | Recall a 0,5 |
| --- | ---: | ---: | ---: |
| GaussianNB legado | 0,5449 | 0,1581 | 0,0000 |
| **CategoricalNB selecionado** | 0,5790 | 0,1629 | 0,0000 |
| GaussianNB proporcional | 0,5006 | 0,1423 | 0,0000 |
| CategoricalNB financeiro, melhor no total | 0,5762 | 0,1720 | 0,0035 |
| CategoricalNB financeiro, melhor no recorte | 0,5893 | 0,1878 | 0,0000 |
| Prevalência do recorte no desenvolvimento | 0,5000 | 0,1293 | 0,0000 |

## Testes de independência monetária

Os testes reescalam simultaneamente limite, faturas e pagamentos por 0,01, 0,1, 0,5, 2, 6, 10 e 100, mantendo os atrasos. Eles verificam atributos finitos, igualdade de atributos e probabilidades com tolerância absoluta e relativa de 1e−12 e igualdade da classificação a 0,5. Os seis perfis incluem o exemplo R$ 5.000 / R$ 4.000 / R$ 1.000, faturas negativas, meses com zero dívida e zero pagamento, pagamento e fatura acima do limite e razões com valores muito pequenos ou extremos. Uma verificação automatizada adicional repete os sete fatores nos 30.000 registros da base. A propriedade se aplica quando o reescalonamento é representável em ponto flutuante. Arredondar o valor financeiro antes da entrada pode mudar a razão.

## Arquitetura e expansão

`data.py` adapta as colunas da UCI ao histórico cronológico. Um futuro adaptador deverá documentar como define limite, fatura, pagamento por mês de ocorrência e atraso. `features.py` transforma o histórico sem conhecer nomes de colunas da UCI. `estimators.py` concentra a seleção e a discretização ajustadas durante o treino. `train.py` controla os experimentos e avalia os modelos. `model.py` aplica os mesmos indicadores na inferência, e `app.py` cuida da entrada em reais e da apresentação da probabilidade.

Para investigar generalização internacional, é preciso comparar definições de rótulo, perfis de crédito, distribuições dos indicadores e calibração por país e período. Testes externos com separação por país permitiriam comparar treino local, transferência entre países e treino multinacional.

A independência da unidade monetária é uma propriedade do cálculo. Sua validade socioeconômica exige evidência própria. Comportamento financeiro, prevalência de inadimplência, políticas de crédito, juros, condições econômicas, distribuições de atributos e definições de inadimplência variam. A base retrata Taiwan em 2005. O modelo ainda não possui validação externa com a população brasileira e se destina à demonstração acadêmica.

## Artefatos e restauração

O treino gera `artifacts/currency_model.joblib` e `artifacts/currency_metrics.json`. O JSON contém os 90 experimentos, atributos selecionados, associações univariadas, correlações removidas e métricas completas. Carregue apenas Joblib gerado pelo próprio projeto. Dados, ambiente e artefatos continuam fora do Git.

Os artefatos originais ficaram em `artifacts/model.joblib` e `artifacts/metrics.json`, com cópias em `artifacts/legacy/` no ambiente da migração. O artefato legado requer a versão anterior do pré-processamento e da interface. Para executar a versão anterior em uma pasta isolada, a partir da raiz do projeto:

```sh
projeto="$PWD"
legado="$(mktemp -d)"
git archive 83e5e0c | tar -x -C "$legado"
mkdir -p "$legado/artifacts"
cp artifacts/legacy/model.joblib artifacts/legacy/metrics.json "$legado/artifacts/"
(cd "$legado" && "$projeto/.venv/bin/python" -m defaulter_gauge.app)
```
