# Problema 2 — Presença de doença cardíaca


## 1. Dados e preparação

- **Dados:** 303 registros Cleveland e 13 preditores clínicos.
- **Ausentes:** seis registros removidos (`ca`: 4; `thal`: 2), restando 297.
- **Alvo:** 0 = sem doença registrada; valores originais de 1 a 4 = com doença.
- **Divisão:** 237 registros de treino e 60 de teste, com estratificação.
- **Escala:** `StandardScaler` ajustado somente no treino.

Os cálculos e a curva ROC estão no [notebook](problema2.ipynb).

## 2. Análise e seleção

Treinamos `LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)`. O otimizador convergiu em **5 iterações**.

## 3. Modelo e resultados

| Classe no teste | Precisão | Recall | F1 | Amostras |
|---|---:|---:|---:|---:|
| Sem doença (0) | 0,8235 | 0,8750 | 0,8485 | 32 |
| Com doença (1) | 0,8462 | 0,7857 | 0,8148 | 28 |

**Acurácia:** 0,8333. **AUC ROC:** 0,9498. Na matriz de confusão, houve 28 verdadeiros negativos, 4 falsos positivos, 6 falsos negativos e 22 verdadeiros positivos. A curva ROC e a matriz estão no notebook.

## 4. Interpretação e conclusão

**Resposta ao enunciado.** A AUC sugere boa capacidade de ordenar positivos acima de negativos neste teste, enquanto o recall mostra que **6 dos 28 casos positivos** não foram identificados no limiar padrão. O conjunto de teste é pequeno. O resultado descreve o exercício e não constitui validação para uso clínico; mudanças no limiar exigiriam avaliação adicional dos dois tipos de erro.
