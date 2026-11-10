# Problema 3 — Qualidade de vinhos com k-NN

## 1. Dados e preparação

- **Dados:** 1.599 vinhos tintos, 11 atributos e nenhum ausente.
- **Limpeza:** 240 duplicatas removidas; restaram **1.359** amostras.
- **Classes:** `ruim` ≤5 (640), `médio` 6–7 (702), `bom` ≥8 (17).
- **Divisão:** 1.087 amostras de treino e 272 de teste, com estratificação.
- **Escala:** `StandardScaler` dentro de cada fold, ajustado somente à parte de treino.

O k-NN usa distâncias euclidianas: sem padronização, atributos de escala maior dominariam a escolha dos vizinhos. O código e a curva de validação estão no [notebook](problema3.ipynb).

## 2. Análise e seleção

Testamos `k = 1,…,19` em validação cruzada estratificada de cinco folds no treino. O maior valor médio de acurácia foi **0,7057 em k=11**. Valores muito pequenos tornam a decisão sensível a casos individuais; valores altos suavizam as fronteiras e favorecem classes frequentes. A escolha por acurácia beneficia as duas classes maiores e não garante detecção da classe `bom`.

## 3. Modelo e resultados

| Classe real / prevista | Ruim | Médio | Bom |
|---|---:|---:|---:|
| Ruim | 92 | 36 | 0 |
| Médio | 29 | 112 | 0 |
| Bom | 0 | 3 | 0 |

| Classe | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Ruim | 0,7603 | 0,7188 | 0,7390 |
| Médio | 0,7417 | 0,7943 | 0,7671 |
| Bom | 0 | 0 | 0 |

## 4. Interpretação e conclusão

**Resposta ao enunciado.** A acurácia total foi **0,7500**. O par `ruim`–`médio` concentrou **65 erros** (36 em um sentido e 29 no outro). Os três vinhos `bom` do teste foram classificados como `médio`; a classe rara impede uma estimativa estável de seu desempenho e evidencia a limitação da acurácia global.
