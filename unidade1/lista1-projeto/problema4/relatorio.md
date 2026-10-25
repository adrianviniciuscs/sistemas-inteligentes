# Problema 4 — Risco de crédito com árvore de decisão


## 1. Dados e preparação

- **Dados:** 1.000 solicitações, sendo 700 de bom risco e 300 de mau risco; nenhum ausente.
- **Atributos:** sete numéricos e 13 categóricos; os categóricos receberam codificação one-hot.
- **Divisão:** 800 casos de treino e 200 de teste, com estratificação.
- **Escala:** árvores dividem por limiares e dispensam padronização.

Prever sempre `bom` já acertaria **70%** dos casos; por isso, analisamos também o recall da classe `mau`. O código e os gráficos estão no [notebook](problema4.ipynb).

## 2. Análise e seleção

Com cinco folds no treino, comparamos profundidades 2, 3, 5, 7, 10 e sem limite para Gini e entropia. A maior acurácia média de validação foi **0,7137** com entropia e profundidade 5; Gini nessa profundidade obteve **0,7012**. A diferença observada é pequena. Árvores rasas podem sofrer **subajuste (underfitting)**. Sem limite, ambas atingiram acurácia **1,0 no treino** sem ganho correspondente de validação, sinal de **sobreajuste (overfitting)**.

## 3. Modelo e resultados

### Interpretação da árvore

As maiores importâncias por redução de impureza na árvore escolhida foram `checking_status_A14` (**0,2793**), `credit_amount` (**0,1845**) e `duration` (**0,0897**). O código `A14` indica **ausência de conta corrente**. A visualização completa está no notebook.

A folha com mais exemplos de mau risco no treino contém **158 maus e 164 bons**. Seu caminho é:

1. Não pertencer à categoria A14 de conta corrente (`checking_status_A14 = 0`).
2. Solicitar prazo superior a 11,5 meses (`duration > 11,5`).
3. Não pertencer à categoria A65 de poupança (`savings_status_A65 = 0`).
4. Solicitar até 10.841,5 unidades monetárias (`credit_amount ≤ 10.841,5`).
5. Não pertencer à categoria A64 de poupança (`savings_status_A64 = 0`).

**Previsão dessa folha: bom risco**, pois os bons são ligeiramente mais numerosos nela. Isso ajuda a explicar os maus pagadores que o modelo deixa passar.

### Avaliação no teste

| Classe real / prevista | Bom | Mau |
|---|---:|---:|
| Bom | 130 | 10 |
| Mau | 53 | 7 |

| Classe | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Bom | 0,7104 | 0,9286 | 0,8050 |
| Mau | 0,4118 | 0,1167 | 0,1818 |

## 4. Interpretação e conclusão

**Resposta ao enunciado.** A acurácia foi **0,6850**, abaixo da linha de base de 0,70. O erro mais frequente é classificar **53 maus como bons**. Conceder crédito a um mau pagador pode causar perda do valor emprestado; recusar um bom pagador tende a causar perda da oportunidade de negócio. O custo relativo precisa ser definido na aplicação. Para reduzir o primeiro erro, testaríamos pesos maiores para a classe `mau` ou outro limiar **na validação**, medindo também quantos bons clientes seriam recusados.
