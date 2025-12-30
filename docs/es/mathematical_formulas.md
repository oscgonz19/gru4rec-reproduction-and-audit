# Formulas Matematicas y Derivaciones

**Audiencia:** Estadisticos, Investigadores Cuantitativos, Teoricos de ML
**Prerrequisitos:** Algebra lineal, teoria de probabilidad, calculo

---

## Tabla de Contenidos

1. [Notacion](#1-notacion)
2. [Redes Neuronales Recurrentes](#2-redes-neuronales-recurrentes)
3. [Unidades Recurrentes con Compuertas](#3-unidades-recurrentes-con-compuertas)
4. [Funciones de Perdida](#4-funciones-de-perdida)
5. [Metricas de Evaluacion](#5-metricas-de-evaluacion)
6. [Distribuciones de Probabilidad](#6-distribuciones-de-probabilidad)

---

## 1. Notacion

| Simbolo | Descripcion | Dimensiones |
|---------|-------------|-------------|
| $V$ | Tamano del vocabulario (numero de items) | Escalar |
| $D$ | Dimension del embedding | Escalar |
| $H$ | Dimension del estado oculto | Escalar |
| $B$ | Tamano del batch | Escalar |
| $T$ | Longitud de la secuencia | Escalar |
| $\mathbf{E}$ | Matriz de embedding | $V \times D$ |
| $\mathbf{e}_i$ | Embedding del item $i$ | $D \times 1$ |
| $\mathbf{h}_t$ | Estado oculto en el tiempo $t$ | $H \times 1$ |
| $\mathbf{x}_t$ | Entrada en el tiempo $t$ | $D \times 1$ |
| $\mathbf{y}_t$ | Scores de salida en el tiempo $t$ | $V \times 1$ |
| $\sigma(\cdot)$ | Funcion sigmoide | — |
| $\odot$ | Producto elemento a elemento (Hadamard) | — |

---

## 2. Redes Neuronales Recurrentes

### 2.1 RNN Vanilla

La red neuronal recurrente basica calcula estados ocultos como:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

Salida en cada paso temporal:

$$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$

### 2.2 Problema del Gradiente Desvaneciente

Para una secuencia de longitud $T$, el gradiente de la perdida con respecto a estados ocultos tempranos involucra:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$

Si $\|\mathbf{W}_{hh}\| < 1$, los gradientes desaparecen exponencialmente.

---

## 3. Unidades Recurrentes con Compuertas

### 3.1 Ecuaciones GRU

**Compuerta de Actualizacion:**
$$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**Compuerta de Reinicio:**
$$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

**Estado Oculto Candidato:**
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})$$

**Estado Oculto Final:**
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### 3.2 Interpretacion de Compuertas

**Compuerta de Actualizacion ($\mathbf{z}_t$):** Controla el flujo de informacion del estado anterior.
- $z_t \approx 0$: Mantener estado anterior ($\mathbf{h}_t \approx \mathbf{h}_{t-1}$)
- $z_t \approx 1$: Usar nuevo candidato ($\mathbf{h}_t \approx \tilde{\mathbf{h}}_t$)

**Compuerta de Reinicio ($\mathbf{r}_t$):** Controla cuanta informacion pasada olvidar.
- $r_t \approx 0$: Ignorar estado anterior al calcular candidato
- $r_t \approx 1$: Usar estado anterior completo

---

## 4. Funciones de Perdida

### 4.1 Perdida de Entropia Cruzada

Para un item objetivo $i$ entre $V$ items:

$$\mathcal{L}_{CE} = -\log\left(\frac{\exp(r_i)}{\sum_{j=1}^{V}\exp(r_j)}\right) = -r_i + \log\left(\sum_{j=1}^{V}\exp(r_j)\right)$$

Donde $r_j = \mathbf{h}_t^\top \mathbf{e}_j$ es el score para el item $j$.

**Gradiente con respecto a scores:**

$$\frac{\partial \mathcal{L}_{CE}}{\partial r_j} = \text{softmax}(r_j) - \mathbb{1}[j = i] = p_j - \mathbb{1}[j = i]$$

### 4.2 Perdida BPR (Ranking Personalizado Bayesiano)

$$\mathcal{L}_{BPR} = -\log\sigma(r_i - r_j)$$

Donde $i$ es el item positivo, $j$ es una muestra negativa.

**Intuicion:** Maximizar la probabilidad de que el item positivo tenga mayor score que los negativos.

### 4.3 Perdida BPR-Max

$$\mathcal{L}_{BPR-max} = -\log\sigma\left(r_i - \max_{j \in \mathcal{N}}(r_j)\right) + \lambda \sum_{j \in \mathcal{N}} \sigma(r_j)^2 \cdot s_j$$

Donde:
- $s_j$ es el peso softmax del negativo $j$
- $\lambda$ es el coeficiente de regularizacion

---

## 5. Metricas de Evaluacion

### 5.1 Recall@K

$$\text{Recall@K} = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}[\text{rank}(t) \leq K]$$

**Propiedades:**
- Rango: $[0, 1]$
- Mayor es mejor
- Mide "tasa de acierto" en top-K

### 5.2 Rango Reciproco Medio (MRR@K)

$$\text{MRR@K} = \frac{1}{|T|} \sum_{t \in T} \frac{\mathbb{1}[\text{rank}(t) \leq K]}{\text{rank}(t)}$$

**Propiedades:**
- Rango: $[0, 1]$
- Mayor es mejor
- Recompensa rankings mas altos mas que Recall

### 5.3 Ganancia Acumulada Descontada Normalizada (NDCG@K)

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i + 1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Para prediccion de objetivo unico (rel = 1 para objetivo, 0 sino):

$$\text{NDCG@K} = \frac{\mathbb{1}[\text{rank} \leq K]}{\log_2(\text{rank} + 1)}$$

### 5.4 Calculo de Rango (Modo Conservador)

$$\text{rank}(i) = \sum_{j=1}^{V} \mathbb{1}[r_j \geq r_i]$$

Esto cuenta todos los items con score mayor o igual al objetivo, manejando empates conservadoramente.

---

## 6. Distribuciones de Probabilidad

### 6.1 Distribucion de Popularidad de Items (Ley de Zipf)

En datasets del mundo real, la popularidad de items sigue una ley de potencia:

$$P(X = k) \propto \frac{1}{k^s}$$

Donde $k$ es el rango de popularidad y $s \approx 1$ (exponente de Zipf).

**Normalizada:**
$$P(X = k) = \frac{k^{-s}}{\sum_{i=1}^{V} i^{-s}} = \frac{k^{-s}}{H_{V,s}}$$

Donde $H_{V,s} = \sum_{i=1}^{V} i^{-s}$ es el numero armonico generalizado.

### 6.2 Generacion de Datos Sinteticos

Para testing reproducible, generamos datos con:

**Seleccion de item:**
$$P(\text{item} = i) = \frac{w_i}{\sum_j w_j}, \quad w_i = \frac{1}{i^{0.8}}$$

**Longitud de sesion:**
$$L \sim \text{Uniforme}(L_{min}, L_{max})$$

---

## Apendice A: Propiedades de Sigmoide y Softmax

### Funcion Sigmoide

$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{1 + e^x}$$

**Propiedades:**
- Rango: $(0, 1)$
- Simetrica: $\sigma(-x) = 1 - \sigma(x)$
- Derivada: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

### Funcion Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Propiedades:**
- La salida suma 1: $\sum_i \text{softmax}(x_i) = 1$
- Invariante a desplazamiento constante: $\text{softmax}(x + c) = \text{softmax}(x)$

---

*Este documento es parte del proyecto de Estudio de Reproducibilidad de GRU4Rec.*
