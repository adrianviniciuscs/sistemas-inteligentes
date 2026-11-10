# Lista 1 de Projetos — Sistemas Inteligentes

Implementações dos problemas 1 a 4 de aprendizado de máquina da disciplina PAM0466. Cada projeto reúne o conjunto de dados, um notebook com o experimento reproduzível e um relatório com a análise dos resultados.

## Projetos

| Problema | Técnica | Notebook | Relatório |
|---|---|---|---|
| 1. Produção de energia | Regressão linear | [Código](problema1/problema1.ipynb) | [Análise](problema1/relatorio.md) |
| 2. Doença cardíaca | Regressão logística | [Código](problema2/problema2.ipynb) | [Análise](problema2/relatorio.md) |
| 3. Qualidade do vinho | k-NN | [Código](problema3/problema3.ipynb) | [Análise](problema3/relatorio.md) |
| 4. Risco de crédito | Árvore de decisão | [Código](problema4/problema4.ipynb) | [Análise](problema4/relatorio.md) |

## Enunciado

[Lista 1 de projetos (PDF)](lista1-projeto-si.pdf).

## Como executar

Na pasta desta lista, crie um ambiente Python e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

No Windows, ative o ambiente com `.venv\Scripts\activate`.

Abra o notebook dentro da pasta do problema e execute todas as células, na ordem. Os caminhos dos dados são relativos à pasta de cada notebook. Os arquivos de dados necessários estão incluídos; não é preciso baixá-los novamente.

Os notebooks contêm as saídas, tabelas e gráficos, e os relatórios resumem os resultados. Os arquivos de documentação das bases foram mantidos junto aos dados. O PDF contém o enunciado original completo; esta entrega reúne somente os problemas 1 a 4.
