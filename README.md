# Previsão de preços de conjuntos LEGO com redes neurais

Projeto acadêmico da disciplina de **Inteligência Artificial — Ciência da Computação, PUC-SP**. O laboratório utiliza uma rede neural artificial do Scikit-learn para estimar preços ausentes de conjuntos LEGO.

**Integrantes:** Nicolas Mariano da Silva e Pedro Henrique Isamu Yoshissaro

## Objetivo

Explorar a base LEGO, identificar e tratar dados faltantes e criar a coluna `price`, preservando os preços conhecidos e estimando os ausentes com uma ANN.

## Base de dados

Fonte: [Maven Analytics — LEGO Sets](https://mavenanalytics.io/data-playground/lego-sets).

A base contém lançamentos entre **1970 e 2022**, com informações sobre tema, categoria, peças, minifiguras, idade recomendada e preço de varejo nos Estados Unidos.

| Informação | Quantidade |
|---|---:|
| Conjuntos LEGO | 18.457 |
| Colunas originais | 14 |
| Preços conhecidos | 6.982 |
| Preços ausentes estimados | 11.475 |

Os preços estão em **dólares americanos (USD), no lançamento**. Não representam preços atuais ou valores de revenda.

## Etapas realizadas

1. Download e leitura dos CSVs com Pandas.
2. Análise das colunas e visualização de valores ausentes com Missingno, Matplotlib e Seaborn.
3. Remoção de duplicatas exatas, padronização de textos e tratamento de valores numéricos inválidos.
4. Separação dos registros com preço conhecido em treino, validação e teste.
5. Pré-processamento dos atributos e treinamento de redes neurais.
6. Seleção da configuração pelo menor erro absoluto médio na validação.
7. Avaliação em teste reservado e comparação com a previsão pela mediana.
8. Análise da importância das colunas por permutação na validação.
9. Reajuste da configuração escolhida com todos os preços conhecidos e preenchimento dos preços ausentes.
10. Exportação dos resultados, métricas e modelo.

## Atributos utilizados

| Coluna | Informação |
|---|---|
| `year` | Ano de lançamento |
| `pieces` | Número de peças |
| `minifigs` | Número de minifiguras |
| `agerange_min` | Idade mínima recomendada |
| `theme` | Tema do conjunto |
| `themeGroup` | Grupo do tema |
| `category` | Categoria do produto |

A coluna `US_retailPrice` é o alvo e não entra como atributo. Identificadores e URLs ficam fora do modelo. `name` e `subtheme` também não são utilizados nesta versão.

## Modelo e avaliação

Foi utilizado o **`MLPRegressor`**, uma rede neural para regressão, com ativação ReLU, otimizador Adam, regularização L2 e parada antecipada.

O pipeline aplica imputação pela mediana, indicadores de ausência e padronização aos atributos numéricos. As categorias recebem um marcador de ausência e codificação one-hot. O pré-processamento é ajustado no conjunto de treino.

Foram comparadas arquiteturas com **64 e 32 neurônios** e **128 e 64 neurônios**, combinadas com alvo padronizado ou transformado por `log1p` e padronizado. A configuração selecionada utiliza **duas camadas ocultas, com 64 e 32 neurônios, e transformação logarítmica do alvo**.

| Partição | Registros |
|---|---:|
| Treino | 4.188 |
| Validação externa | 1.397 |
| Teste | 1.397 |

A divisão utiliza a semente `42`. O MLP reserva ainda uma fração interna do treino para controlar a parada antecipada. O teste externo não participa da escolha da configuração.

### Resultados no teste reservado

| Modelo | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Rede neural selecionada | 8,64 | 19,09 | 0,886 |
| Previsão pela mediana do treino | 25,81 | 59,02 | −0,089 |

O **MAE** mede o erro absoluto médio. O **RMSE** dá maior peso a erros grandes. O **R²** mede o desempenho em relação à previsão pela média; não é uma porcentagem de acerto.

A ANN apresentou menor erro que a previsão pela mediana. As colunas com maior importância por permutação na validação foram `pieces`, `theme` e `themeGroup`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| [PUCSP_CS_AI_10_AI_ML_ANN_LEGO_LAB.ipynb](PUCSP_CS_AI_10_AI_ML_ANN_LEGO_LAB.ipynb) | Notebook executado, com código, explicações, gráficos e conclusão |
| `lego_sets.csv` | Base original |
| `lego_sets_data_dictionary.csv` | Dicionário dos dados |
| `requirements.txt` | Versões das dependências utilizadas |
| `lego_sets_com_precos.csv` | Base completa com preços observados e estimados |
| `lego_precos_estimados.csv` | Apenas os conjuntos que receberam estimativas |
| `metricas_validacao.csv` | Comparação das configurações na validação |
| `metricas_teste.csv` | Avaliação final e baseline |
| `importancia_colunas.csv` | Importância por permutação |
| `modelo_ann_lego.joblib` | Modelo final reajustado com todos os preços conhecidos |
| `metadados_modelo.json` | Configuração, métricas e informações da execução |

Os arquivos de dados e resultados estão no pacote completo. O notebook isolado consegue baixar os dados e gerar as saídas novamente.

## Como executar

### VS Code ou Jupyter

1. Extraia o pacote completo em uma pasta.
2. Utilize Python 3.12 e instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Abra `PUCSP_CS_AI_10_AI_ML_ANN_LEGO_LAB.ipynb`.
4. Selecione o kernel do ambiente Python utilizado.
5. Execute todas as células na ordem.

O notebook reutiliza os CSVs disponíveis junto ao arquivo. Caso não estejam disponíveis, realiza o download automaticamente. As saídas são gravadas na pasta `resultados_lego`, relativa ao diretório de execução.

### Google Colab

1. Abra o [Google Colab](https://colab.research.google.com/).
2. Envie o notebook pela opção de upload.
3. Execute todas as células na ordem.
4. Baixe os arquivos gerados em `resultados_lego` pelo painel de arquivos.

A primeira célula instala bibliotecas ausentes. O treinamento não exige GPU.

## Colunas adicionadas

| Coluna | Significado |
|---|---|
| `price` | Preço final: observado ou estimado pela ANN |
| `price_source` | `observed` para preço original; `estimated_ann` para estimativa |
| `missing_feature_count` | Quantidade de atributos preditores ausentes no registro |
| `prediction_extrapolation` | Indica estimativa com categoria inédita ou atributo numérico fora da faixa dos registros com preço conhecido |
| `prediction_caution` | Sinaliza estimativa com extrapolação ou pelo menos dois atributos ausentes |

As verificações confirmaram que todos os conjuntos receberam um preço positivo e finito e que os preços observados foram preservados. As estimativas são arredondadas para duas casas decimais e recebem um piso operacional de US$ 0,01.

## Limitações

- A avaliação usa registros com preço conhecido. O erro pode ser diferente nos conjuntos originalmente sem preço, pois a cobertura dos dados varia entre categorias, temas e épocas.
- Os indicadores de cautela não são intervalos de confiança.
- Não foi aplicada correção monetária ou conversão para reais.
- Uma estimativa não comprova que o item tenha sido vendido individualmente nos Estados Unidos.
- As métricas de teste pertencem à ANN avaliada antes do reajuste final. O modelo exportado foi reajustado com todos os preços conhecidos para preencher a base.
- Para reutilizar o arquivo `joblib`, mantenha um ambiente compatível e carregue apenas arquivos de fonte confiável.

## Entrega

Preencha os nomes da dupla e publique o notebook neste repositório. Copie o link da página do arquivo `.ipynb` no GitHub e envie-o na atividade do Teams.
