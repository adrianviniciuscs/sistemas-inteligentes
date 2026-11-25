# 2ª Lista de Exercícios — Sistemas Inteligentes (Resolução Resumida)

> Resumo direto ao ponto para revisão e memorização, com fórmulas apenas onde é indispensável.

---

## 1. Fundamentos e Arquiteturas de Redes Neurais

### 1. Habilidade de generalização
É a capacidade da rede neural de **acertar em dados novos (nunca vistos no treinamento)** que seguem o mesmo padrão do problema.
- **Ideia-chave:** aprender o padrão geral dos dados, e não apenas "decorar" o conjunto de treino (*overfitting*).
- **Como se mede:** avaliando a rede em um conjunto de **teste** separado.

### 2. Papel da função de ativação e importância da não linearidade
- **Papel:** recebe a soma ponderada das entradas ($u$) e define a saída do neurônio ($y = g(u)$), **limitando a amplitude** da saída (ex.: $[0,1]$, $[-1,1]$) e controlando o "disparo" do neurônio.
- **Por que não linearidade é crucial:** combinar várias funções lineares sempre resulta em **uma única função linear**. Sem função de ativação não linear, uma rede com 10 camadas teria o mesmo poder de uma rede de 1 camada só — não conseguiria aprender fronteiras curvas nem problemas como o XOR.

### 3. Feedforward de camada simples vs. múltiplas camadas
Em ambas, o sinal só vai **para frente** (entrada $\to$ saída, sem realimentação). A diferença é a presença de **camadas ocultas**:
- **Camada simples:** só tem entrada $\to$ saída (1 camada de processamento). Ex.: Perceptron, Adaline. Só separa com **hiperplanos (retas)**.
- **Múltiplas camadas:** tem **uma ou mais camadas ocultas** entre entrada e saída. Ex.: MLP, CNN. Extrai características em etapas e cria **fronteiras não lineares**.

---

## 2. Perceptron e Separabilidade Linear

### 4. Arquitetura do Perceptron e tipos de valores aceitos
Classificador linear de **camada única** formado por: entradas ($x_i$), pesos ($w_i$), limiar/viés ($\theta$ ou $b$), somador linear ($u = \sum w_i x_i - \theta$) e função de ativação **degrau** (unipolar ou bipolar).
- **Entradas ($\mathbf{x}$):** valores **reais contínuos** ou **discretos** (binários $\{0,1\}$ ou bipolares $\{-1,+1\}$).
- **Pesos ($\mathbf{w}$) e limiar ($\theta$):** valores **reais contínuos** (positivos, negativos ou zero).
- **Saída ($y$):** sempre **discreta** ($\{0,1\}$ ou $\{-1,+1\}$).

### 5. Treinamento do Perceptron e atualização de pesos
Treinamento supervisionado por **correção de erros**:
1. Inicializa os pesos com valores pequenos aleatórios (ou zeros) e define a taxa de aprendizado $\eta$.
2. Para cada amostra, calcula $u$ e a saída $y = g(u)$.
3. Compara $y$ com a saída desejada $d$:
   - Se **$y = d$** (acertou): **não muda** os pesos.
   - Se **$y \neq d$** (errou): atualiza os pesos pela regra:
     $$w_i^{\text{novo}} = w_i^{\text{atual}} + \eta (d - y) x_i$$
4. Repete por épocas até passar uma **época inteira sem erros** (ou atingir limite máximo de épocas).

### 6. Teorema da convergência do Perceptron
- **O que diz:** se os dados forem **linearmente separáveis**, o Perceptron **sempre converge** e encontra um hiperplano que separa tudo em um **número finito de iterações** (independente dos pesos iniciais).
- **Se NÃO forem linearmente separáveis:** **nunca converge**; o hiperplano fica oscilando infinitamente (corrigir um ponto faz errar outro). Precisa de um critério de parada externo (ex.: número máximo de épocas).

---

## 3. Rede Adaline e Regra Delta

### 7. Diferença entre regra do Perceptron e regra Delta (Adaline) no cálculo do erro
A diferença está em **onde** o erro é calculado:
- **Perceptron (pós-ativação / discreto):** calcula o erro **depois** da função degrau $\to e = d - y$. Só sabe se errou ou acertou.
- **Adaline / Regra Delta (pré-ativação / contínuo):** calcula o erro **antes** da função degrau, usando o potencial linear $u$ $\to e = d - u$. Por ser contínuo e diferenciável, usa **gradiente descendente** para minimizar o Erro Quadrático Médio (EQM).

### 8. Por que o hiperplano do Adaline é único (ao contrário do Perceptron)?
- **Perceptron:** para assim que acerta todas as amostras. Como existem infinitas retas que separam duas classes linearmente separáveis, ele fica com a **primeira que encontrar** (depende dos pesos iniciais e da ordem das amostras).
- **Adaline:** minimiza o **EQM**, cuja superfície de erro é um **paraboloide convexo** (formato de tigela) com **um único mínimo global**. Assim, sempre converge para o mesmo hiperplano ótimo, mais centralizado e robusto a ruído.

### 9. Critério matemático de parada no Adaline
Como o erro contínuo ($d - u$) raramente chega exatamente a zero, o Adaline para quando o **EQM estabiliza** entre duas épocas seguidas:
$$\text{EQM} = \frac{1}{N}\sum_{k=1}^{N}(d^{(k)} - u^{(k)})^2$$
$$\left|\text{EQM}_{\text{atual}} - \text{EQM}_{\text{anterior}}\right| \le \varepsilon$$
onde $\varepsilon > 0$ é uma tolerância bem pequena (ex.: $10^{-6}$).

### 10. Influência da taxa de aprendizagem ($\eta$)
Controla o **tamanho do passo** na atualização dos pesos:
- **$\eta$ muito pequena:** treinamento **muito lento** (muitas épocas). No Adaline, pode até causar parada prematura por variação mínima do EQM.
- **$\eta$ muito grande:** passos exagerados geram **oscilações** fortes ao redor da solução; no Adaline, pode **ultrapassar o mínimo e divergir** (erro explodindo).

### 11. É possível aplicar o Perceptron neste problema?
Dados: $(0{,}75; 0{,}75) \to A$, $(0{,}25; 0{,}25) \to A$, $(0{,}75; 0{,}25) \to B$, $(0{,}25; 0{,}75) \to B$.

**Resposta: Não é possível**, pois os dados **não são linearmente separáveis** (é equivalente ao problema lógico **XNOR/XOR**).

- **Justificativa geométrica (mais fácil de lembrar):** os 4 pontos formam um quadrado no plano $(x_1, x_2)$. A classe $A$ está em uma diagonal e a classe $B$ na outra diagonal — nenhuma reta única consegue separar as duas diagonais.
- **Justificativa algébrica rápida:** supondo $A > 0$ e $B < 0$ para $w_1 x_1 + w_2 x_2 - \theta$:
  - Somando os dois pontos de $A$: $(0{,}75 + 0{,}25)w_1 + (0{,}75 + 0{,}25)w_2 - 2\theta > 0 \implies w_1 + w_2 > 2\theta$.
  - Somando os dois pontos de $B$: $(0{,}75 + 0{,}25)w_1 + (0{,}25 + 0{,}75)w_2 - 2\theta < 0 \implies w_1 + w_2 < 2\theta$.
  - **Contradição!** Logo, não existe solução linear.

---

## 4. MLP e Backpropagation

### 12. Diferença estrutural entre MLP e Perceptron simples (e por que resolve problemas não lineares)
- **Diferenças estruturais:**
  1. A MLP tem **uma ou mais camadas ocultas** (o Perceptron simples não tem).
  2. A MLP usa **funções de ativação não lineares e diferenciáveis** (sigmoide, tanh, ReLU), e não apenas degrau.
- **Por que resolve problemas não linearmente separáveis:** cada neurônio oculto traça um hiperplano diferente e transforma o espaço original em um novo espaço onde as classes passam a ser separáveis. Combinando esses hiperplanos, a MLP cria fronteiras **curvas ou fechadas** (como no XOR).

### 13. Gradiente local $\delta_j^{(n)}$ na camada de saída vs. camadas ocultas
O **gradiente local** ($\delta_j$) mede o quanto o erro da rede é sensível à variação do potencial de ativação ($u_j$) do neurônio $j$ — ou seja, indica a **"culpa"** daquele neurônio no erro.

- **Na camada de saída:** o erro é conhecido diretamente ($d_j - y_j$):
  $$\delta_j = (d_j - y_j) \cdot g'(u_j)$$
- **Nas camadas ocultas:** não existe saída desejada $d_j$; o erro vem **retropropagado** da camada seguinte (soma dos gradientes seguintes ponderados pelos pesos):
  $$\delta_j = g'(u_j) \cdot \sum_k \left(\delta_k \cdot w_{kj}\right)$$

### 14. Regra de atualização dos pesos no Backpropagation e papel de $\eta$
- **Regra (Regra Delta Generalizada):**
  $$w_{ji}^{\text{novo}} = w_{ji}^{\text{atual}} + \eta \cdot \delta_j \cdot y_i$$
  onde $\eta$ é a taxa de aprendizado, $\delta_j$ é o gradiente local do neurônio destino $j$ e $y_i$ é a entrada vinda do neurônio anterior $i$.
- **Papel de $\eta$:** define o tamanho do passo na descida do gradiente.
  - Muito pequena $\to$ convergência lenta e risco de travar em **mínimos locais** ou platôs.
  - Muito grande $\to$ **oscilações**, divergência ou saturação das funções de ativação.

### 15. Overfitting e Early Stopping
- **Overfitting (sobreajuste):** acontece quando a rede "decora" o conjunto de treino (inclusive ruídos). O erro no **treino fica muito baixo**, mas o erro em **dados novos (teste/validação) fica alto** (perde generalização).
- **Early Stopping (parada antecipada):**
  1. Separa os dados em **treino** e **validação**.
  2. Treina os pesos apenas no treino, mas mede o erro na **validação** a cada época.
  3. Enquanto ambos os erros caem, continua. Quando o erro de **validação começa a subir** (sinal de overfitting), **para o treinamento** e restaura os pesos da época em que o erro de validação foi mínimo.

### 16. Funções de ativação na MLP (ocultas e saída)
- **Camadas ocultas (não lineares e diferenciáveis):** **ReLU** (mais usada em redes profundas, rápida e evita gradiente desvanecente), **Tangente Hiperbólica (tanh)** (centrada em zero) e **Sigmoide/Logística**.
- **Camada de saída (depende da tarefa):**

| Tarefa | Função na saída | Saída |
| :--- | :--- | :--- |
| **Regressão** (valor contínuo) | Linear / Identidade ($g(u)=u$) | $(-\infty, +\infty)$ |
| **Classificação binária** $\{0,1\}$ | Sigmoide (Logística) | $(0, 1)$ (probabilidade) |
| **Classificação binária** $\{-1,+1\}$ | Tangente Hiperbólica (tanh) | $(-1, 1)$ |
| **Classificação multiclasse** | Softmax (ou Sigmoide/tanh com EQM) | Probabilidades que somam 1 |

### 17. Topologia da MLP e complexidade das regiões de decisão
- **Número de camadas ocultas (profundidade):**
  - **0 ocultas:** só 1 hiperplano (reta/plano).
  - **1 oculta:** forma regiões **convexas** (e, com neurônios suficientes, aproxima qualquer fronteira contínua).
  - **2 ou mais ocultas:** combina sub-regiões e representa formatos **arbitrários** (não convexos, "ilhas" desconexas, anéis) com menos neurônios.
- **Número de neurônios por camada (largura):** define quantos hiperplanos/"lados" moldam a fronteira.
  - **Poucos demais:** fronteira simples demais $\to$ *underfitting*.
  - **Muitos demais:** fronteira recortada demais (decora ruído) $\to$ *overfitting*.

### 18. Duas estratégias de codificação de saída multiclasse ($C$ classes)
1. **Saída única (escalar/inteira):** usa **1 neurônio** na saída, atribuindo um número diferente a cada classe (ex.: $1, 2, 3, \dots, C$).
   - *Problema:* cria uma **ordem/distância artificial** entre classes que não têm relação numérica (ex.: classe 2 fica "entre" 1 e 3).
2. **One-Hot / 1-de-C (padrão recomendado):** usa **$C$ neurônios** na saída (um por classe). A classe correta recebe `1` e as demais `0` (ex.: `[1,0,0]`, `[0,1,0]`, `[0,0,1]`).
   - *Decisão:* vence o neurônio com **maior saída** (*winner-takes-all* / `argmax`). Evita ordenação artificial.

---

## 5. Redes Neurais Convolucionais (CNNs)

### 19. Estrutura típica de uma CNN e função de cada camada
A CNN tem dois blocos: **extração de características** + **classificação**.
1. **Entrada:** recebe a imagem como tensor (Altura $\times$ Largura $\times$ Canais, ex.: RGB).
2. **Convolucional (Conv):** aplica filtros (kernels) deslizantes para gerar **mapas de características** (bordas no início $\to$ formas e objetos nas camadas fundas).
3. **Ativação (ReLU):** introduz não linearidade (zera valores negativos).
4. **Pooling (Subamostragem):** reduz altura e largura dos mapas, diminuindo custo computacional, evitando *overfitting* e dando invariância a pequenas translações.
5. **Flattening:** achata os mapas 3D em um único vetor 1D.
6. **Totalmente Conectadas (Dense / MLP):** combinam globalmente as características extraídas.
7. **Saída:** gera a classificação final (geralmente com **Softmax**).

### 20. Comparação: Max Pooling, Average Pooling e Sum Pooling

| Operação | O que faz na janela | Quando é mais indicada |
| :--- | :--- | :--- |
| **Max Pooling** | Pega o **maior valor** | Padrão em CNNs; ideal para destacar **feições fortes/esparsas** (bordas, cantos, texturas) ignorando o fundo. |
| **Average Pooling** | Calcula a **média** | Quando a informação está **distribuída** na janela, para **suavizar ruído** ou no final da rede (*Global Average Pooling*). |
| **Sum Pooling** | Faz a **soma total** | Quando a **energia total ou contagem acumulada** importa (ex.: contagem de objetos/multidões em mapas de densidade). |

### 21. Como os filtros de uma CNN são obtidos vs. processamento de imagens tradicional
- **Na CNN:** os valores dos filtros são **pesos treináveis** — começam aleatórios e são **aprendidos automaticamente** pelos dados via **Backpropagation** e gradiente descendente, otimizados diretamente para a tarefa final.
- **No processamento tradicional:** os filtros são **fixos e projetados manualmente** por humanos com fórmulas pré-definidas (ex.: Sobel para bordas, Gaussiano para suavização), de forma independente do classificador.

### 22. O que é *flattening* e por que é necessário?
- **O que é:** é o "achatamento" (*reshape*) que transforma o tensor 3D final (Altura $\times$ Largura $\times$ Canais) em um **único vetor 1D** — sem alterar valores e sem ter pesos treináveis.
- **Por que é necessário:** porque as camadas totalmente conectadas (**Dense/MLP**) no final da CNN só aceitam **vetores 1D** como entrada para multiplicar pelos pesos sinápticos e combinar todas as características na classificação final.
