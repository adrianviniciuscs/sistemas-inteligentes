# Problema 1 — Previsão da potência de uma usina


## 1. Dados e preparação

- **Dados:** 9.568 observações de `AT`, `V`, `AP`, `RH` e potência `PE` (MW).
- **Limpeza:** nenhum ausente; 41 duplicatas removidas antes da divisão.
- **Divisão:** 7.621 amostras de treino e 1.906 de teste (80/20; semente 42).
- **Escala:** padronização ajustada apenas no treino.

Os cálculos e as saídas estão no [notebook](problema1.ipynb).

## 2. Análise e seleção

Correlações de Pearson calculadas no treino:

| Par com `PE` | `AT` | `V` | `AP` | `RH` |
|---|---:|---:|---:|---:|
| Correlação | −0,948 | −0,868 | +0,519 | +0,386 |

`AT` e `V` têm as associações lineares marginais mais fortes com `PE`. Isso não significa que a correlação estime o efeito causal ou a contribuição isolada de cada variável. Entre os preditores, o maior coeficiente é `AT` × `V` (**+0,840**), sinal de possível colinearidade. Ela pode dificultar a interpretação dos coeficientes individuais, embora não inviabilize a previsão.

## 3. Modelo e resultados

Ajustamos regressão linear múltipla. Convertidos à escala original, os coeficientes dão aproximadamente:

`PE = 453,586 − 1,977·AT − 0,235·V + 0,063·AP − 0,157·RH`.

O coeficiente de `AT`, por exemplo, descreve a variação média prevista em MW para +1 °C **mantidos os demais preditores constantes**. Não deve ser lido como efeito causal.

| Conjunto | R² | RMSE (MW) | MAE (MW) |
|---|---:|---:|---:|
| Teste | 0,9284 | 4,59 | 3,62 |

## 4. Interpretação e conclusão

**Resposta ao enunciado.** O MAE equivale a cerca de 0,8% da potência média do teste (454,33 MW). O desempenho torna o modelo uma referência útil para prever `PE` em condições semelhantes às observadas. Um único teste aleatório não demonstra desempenho em outras usinas ou regimes operacionais.
