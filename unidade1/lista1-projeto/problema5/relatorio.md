# Problema 5 — Segmentação de clientes com k-Means


## 1. Dados e preparação

- **Dados:** 440 clientes, sem ausentes. Usamos apenas seis categorias de gastos; `Channel` e `Region` ficam para validação externa.
- **Descrição:** média, mediana, desvio padrão, extremos e boxplots estão no [notebook](problema5.ipynb).
- **Transformação:** `log1p` reduz a assimetria e o peso de gastos extremos; Z-score equilibra as escalas na distância euclidiana.
- **Correlação:** `Grocery`–`Detergents_Paper` = **0,9246** nos gastos brutos e **0,7971** após `log1p`.

Em tarefas supervisionadas, o teste estima erro sobre rótulos conhecidos. Aqui não há rótulo a prever, então avaliamos coesão, separação e relação com metadados externos, sem um gabarito de grupos.

## 2. Análise e seleção

| k | Inércia | Silhueta |
|---:|---:|---:|
| **2** | **1.844,06** | **0,2903** |
| 3 | 1.553,42 | 0,2594 |
| 4 | 1.386,86 | 0,1885 |
| 5 | 1.266,15 | 0,1916 |
| 6 | 1.174,76 | 0,2009 |
| 7 | 1.086,36 | 0,1958 |
| 8 | 1.028,49 | 0,1852 |
| 9 | 973,14 | 0,1965 |
| 10 | 932,00 | 0,1905 |

Escolhemos **k=2** pela maior silhueta entre os valores testados e pelos dois perfis comerciais interpretáveis. A curva de inércia não tem cotovelo inequívoco. A inércia ótima não aumenta com mais centroides, pois a solução anterior ainda seria possível: com `k=440`, cada cliente poderia ser seu próprio grupo e a inércia seria zero, sem utilidade como segmentação.

O k-Means converge para um **mínimo local**. Com `k=2`, dez sementes e uma inicialização por execução, a inércia final ficou em torno de **1.844,06–1.844,13** para `random` e `k-means++`. Os desvios padrão foram **0,0289** e **0,0308**, respectivamente: não houve vantagem observável do `k-means++` neste experimento. Seu objetivo é reduzir escolhas iniciais de centroides muito próximos, favorecendo novos centros distantes dos já escolhidos.

## 3. Modelo e resultados

Os centroides foram revertidos da escala padronizada e logarítmica. São centros do espaço transformado expressos em unidades monetárias, **não médias aritméticas dos gastos brutos**.

| Categoria | Grupo 0: frescos/congelados | Grupo 1: mercearia/limpeza |
|---|---:|---:|
| Fresh | 8.580,6 | 4.013,9 |
| Milk | 1.753,7 | 8.061,6 |
| Grocery | 2.274,4 | 12.051,1 |
| Frozen | 2.063,8 | 953,5 |
| Detergents_Paper | 287,4 | 4.035,6 |
| Delicassen | 649,4 | 1.022,1 |

O grupo 0 tem 252 clientes e o grupo 1, 188. No cruzamento com `Channel`, **244/298** clientes Horeca estão no grupo 0 e **134/142** clientes Retail no grupo 1; **ARI = 0,5135**, indicando concordância parcial. `Grocery` e `Detergents_Paper` destacam o perfil ligado a Retail, enquanto `Fresh` e `Frozen` destacam o ligado a Horeca. Para `Region`, **ARI = −0,0077**, sem associação clara com os grupos.

## 4. Interpretação e conclusão

**Resposta ao enunciado.** A projeção PCA do notebook ajuda a observar sobreposição e pontos extremos, mas descarta parte da informação das seis dimensões. O k-Means favorece grupos compactos e tende a lidar pior com grupos alongados ou de tamanhos muito diferentes; clientes atípicos também podem deslocar os centroides. O gráfico ilustra essas limitações, mas a projeção em duas dimensões não comprova a geometria completa dos dados.
