# Mathematical Formulas and Derivations

**Audience:** Statisticians, Quantitative Researchers, ML Theorists
**Prerequisites:** Linear algebra, probability theory, calculus

---

## Table of Contents

1. [Notation](#1-notation)
2. [Recurrent Neural Networks](#2-recurrent-neural-networks)
3. [Gated Recurrent Units](#3-gated-recurrent-units)
4. [Loss Functions](#4-loss-functions)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Probability Distributions](#6-probability-distributions)
7. [Gradient Derivations](#7-gradient-derivations)

---

## 1. Notation

| Symbol | Description | Dimensions |
|--------|-------------|------------|
| $V$ | Vocabulary size (number of items) | Scalar |
| $D$ | Embedding dimension | Scalar |
| $H$ | Hidden state dimension | Scalar |
| $B$ | Batch size | Scalar |
| $T$ | Sequence length | Scalar |
| $\mathbf{E}$ | Embedding matrix | $V \times D$ |
| $\mathbf{e}_i$ | Embedding of item $i$ | $D \times 1$ |
| $\mathbf{h}_t$ | Hidden state at time $t$ | $H \times 1$ |
| $\mathbf{x}_t$ | Input at time $t$ | $D \times 1$ |
| $\mathbf{y}_t$ | Output scores at time $t$ | $V \times 1$ |
| $\sigma(\cdot)$ | Sigmoid function | — |
| $\odot$ | Element-wise (Hadamard) product | — |

---

## 2. Recurrent Neural Networks

### 2.1 Vanilla RNN

The basic recurrent neural network computes hidden states as:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

Output at each timestep:

$$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$

### 2.2 Vanishing Gradient Problem

For a sequence of length $T$, the gradient of the loss with respect to early hidden states involves:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$

Where:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}(1 - \mathbf{h}_t^2) \cdot \mathbf{W}_{hh}$$

If $\|\mathbf{W}_{hh}\| < 1$, gradients vanish exponentially: $\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1}\| \to 0$ as $T \to \infty$.

---

## 3. Gated Recurrent Units

### 3.1 GRU Equations

**Update Gate:**
$$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**Reset Gate:**
$$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

**Candidate Hidden State:**
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})$$

**Final Hidden State:**
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### 3.2 Gate Interpretation

**Update Gate ($\mathbf{z}_t$):** Controls information flow from previous state.
- $z_t \approx 0$: Keep previous state ($\mathbf{h}_t \approx \mathbf{h}_{t-1}$)
- $z_t \approx 1$: Use new candidate ($\mathbf{h}_t \approx \tilde{\mathbf{h}}_t$)

**Reset Gate ($\mathbf{r}_t$):** Controls how much past information to forget.
- $r_t \approx 0$: Ignore previous state when computing candidate
- $r_t \approx 1$: Use full previous state

### 3.3 Why GRUs Solve Vanishing Gradients

The gradient flow through time:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = (1 - \mathbf{z}_t) + \mathbf{z}_t \cdot \frac{\partial \tilde{\mathbf{h}}_t}{\partial \mathbf{h}_{t-1}}$$

The $(1 - \mathbf{z}_t)$ term creates a "skip connection" that allows gradients to flow directly, preventing vanishing.

---

## 4. Loss Functions

### 4.1 Cross-Entropy Loss

For a target item $i$ among $V$ items:

$$\mathcal{L}_{CE} = -\log\left(\frac{\exp(r_i)}{\sum_{j=1}^{V}\exp(r_j)}\right) = -r_i + \log\left(\sum_{j=1}^{V}\exp(r_j)\right)$$

Where $r_j = \mathbf{h}_t^\top \mathbf{e}_j$ is the score for item $j$.

**Gradient with respect to scores:**

$$\frac{\partial \mathcal{L}_{CE}}{\partial r_j} = \text{softmax}(r_j) - \mathbb{1}[j = i] = p_j - \mathbb{1}[j = i]$$

### 4.2 Cross-Entropy with Negative Sampling

For computational efficiency, approximate with $n$ negative samples:

$$\mathcal{L}_{CE-NS} = -\log\left(\frac{\exp(r_i)}{\exp(r_i) + \sum_{j \in \mathcal{N}}\exp(r_j)}\right)$$

Where $\mathcal{N}$ is the set of $n$ sampled negative items.

**Sampling Distribution:**
Items are sampled proportionally to popularity:

$$P(j) \propto c_j^\alpha$$

Where $c_j$ is the count of item $j$ and $\alpha \in [0, 1]$ (typically $\alpha = 0.75$).

### 4.3 BPR Loss (Bayesian Personalized Ranking)

$$\mathcal{L}_{BPR} = -\log\sigma(r_i - r_j)$$

Where $i$ is the positive item, $j$ is a negative sample, and $\sigma$ is the sigmoid function.

**Intuition:** Maximize the probability that the positive item scores higher than negative items.

### 4.4 BPR-Max Loss

$$\mathcal{L}_{BPR-max} = -\log\sigma\left(r_i - \max_{j \in \mathcal{N}}(r_j)\right) + \lambda \sum_{j \in \mathcal{N}} \sigma(r_j)^2 \cdot s_j$$

Where:
- $s_j$ is the softmax weight of negative $j$: $s_j = \frac{\exp(r_j)}{\sum_{k \in \mathcal{N}}\exp(r_k)}$
- $\lambda$ is the regularization coefficient

**Interpretation:**
- First term: Push positive score above the hardest negative
- Second term: Regularize high-scoring negatives toward zero

---

## 5. Evaluation Metrics

### 5.1 Recall@K

$$\text{Recall@K} = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}[\text{rank}(t) \leq K]$$

Where:
- $T$ is the set of test predictions
- $\text{rank}(t)$ is the position of the target item in the sorted score list

**Properties:**
- Range: $[0, 1]$
- Higher is better
- Measures "hit rate" in top-K

### 5.2 Mean Reciprocal Rank (MRR@K)

$$\text{MRR@K} = \frac{1}{|T|} \sum_{t \in T} \frac{\mathbb{1}[\text{rank}(t) \leq K]}{\text{rank}(t)}$$

**Properties:**
- Range: $[0, 1]$
- Higher is better
- Rewards higher rankings more than Recall

### 5.3 Normalized Discounted Cumulative Gain (NDCG@K)

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i + 1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

For single-target prediction (rel = 1 for target, 0 otherwise):

$$\text{NDCG@K} = \frac{\mathbb{1}[\text{rank} \leq K]}{\log_2(\text{rank} + 1)}$$

### 5.4 Rank Computation (Conservative Mode)

$$\text{rank}(i) = \sum_{j=1}^{V} \mathbb{1}[r_j \geq r_i]$$

This counts all items with score greater than or equal to the target, handling ties conservatively.

---

## 6. Probability Distributions

### 6.1 Item Popularity Distribution (Zipf's Law)

In real-world datasets, item popularity follows a power law:

$$P(X = k) \propto \frac{1}{k^s}$$

Where $k$ is the popularity rank and $s \approx 1$ (Zipf's exponent).

**Normalized:**
$$P(X = k) = \frac{k^{-s}}{\sum_{i=1}^{V} i^{-s}} = \frac{k^{-s}}{H_{V,s}}$$

Where $H_{V,s} = \sum_{i=1}^{V} i^{-s}$ is the generalized harmonic number.

### 6.2 Session Length Distribution

Session lengths often follow a geometric or negative binomial distribution:

**Geometric:**
$$P(L = k) = (1-p)^{k-1} p$$

**Mean session length:** $E[L] = \frac{1}{p}$

### 6.3 Synthetic Data Generation

For reproducible testing, we generate data with:

**Item selection:**
$$P(\text{item} = i) = \frac{w_i}{\sum_j w_j}, \quad w_i = \frac{1}{i^{0.8}}$$

**Session length:**
$$L \sim \text{Uniform}(L_{min}, L_{max})$$

---

## 7. Gradient Derivations

### 7.1 Embedding Gradient

For the cross-entropy loss with target $i$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{e}_j} = \frac{\partial \mathcal{L}}{\partial r_j} \cdot \frac{\partial r_j}{\partial \mathbf{e}_j} = (p_j - \mathbb{1}[j=i]) \cdot \mathbf{h}_t$$

### 7.2 Hidden State Gradient (Output Layer)

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \sum_{j=1}^{V} \frac{\partial \mathcal{L}}{\partial r_j} \cdot \frac{\partial r_j}{\partial \mathbf{h}_t} = \sum_{j=1}^{V} (p_j - \mathbb{1}[j=i]) \cdot \mathbf{e}_j = \mathbf{E}^\top (\mathbf{p} - \mathbf{y})$$

Where $\mathbf{y}$ is the one-hot target vector.

### 7.3 GRU Weight Gradients

For the update gate weights:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_z} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} \cdot \frac{\partial \mathbf{z}_t}{\partial \mathbf{W}_z}$$

Where:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \odot (\tilde{\mathbf{h}}_t - \mathbf{h}_{t-1})$$

$$\frac{\partial \mathbf{z}_t}{\partial \mathbf{W}_z} = \mathbf{z}_t \odot (1 - \mathbf{z}_t) \otimes [\mathbf{h}_{t-1}, \mathbf{x}_t]^\top$$

### 7.4 Backpropagation Through Time (BPTT)

For sequence of length $T$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}} + \sum_{t=1}^{T} \sum_{k=1}^{t-1} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_k} \cdot \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}}$$

The first term is the direct contribution at each timestep. The second term captures dependencies through time.

---

## 8. Complexity Analysis

### 8.1 Time Complexity

| Operation | Complexity |
|-----------|------------|
| Forward pass (one timestep) | $O(D \cdot H + H^2)$ |
| Full sequence forward | $O(T \cdot (D \cdot H + H^2))$ |
| Score all items | $O(H \cdot V)$ |
| Training batch | $O(B \cdot T \cdot (D \cdot H + H^2) + B \cdot H \cdot n)$ |
| Full ranking evaluation | $O(|T_{test}| \cdot H \cdot V)$ |

Where $n$ is the number of negative samples.

### 8.2 Space Complexity

| Component | Space |
|-----------|-------|
| Embedding matrix | $O(V \cdot D)$ |
| GRU weights | $O(3 \cdot (D + H) \cdot H)$ |
| Hidden states (batch) | $O(B \cdot H)$ |
| Output scores | $O(B \cdot V)$ |

### 8.3 Comparison

| Model | Training | Inference |
|-------|----------|-----------|
| Popularity | $O(N)$ | $O(1)$ |
| Markov | $O(N)$ | $O(K \log K)$ |
| GRU4Rec | $O(N \cdot T \cdot H^2)$ | $O(T \cdot H^2 + H \cdot V)$ |

Where $N$ is the number of training events.

---

## Appendix A: Sigmoid and Softmax Properties

### Sigmoid Function

$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{1 + e^x}$$

**Properties:**
- Range: $(0, 1)$
- Symmetric: $\sigma(-x) = 1 - \sigma(x)$
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

### Softmax Function

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Properties:**
- Output sums to 1: $\sum_i \text{softmax}(x_i) = 1$
- Invariant to constant shift: $\text{softmax}(x + c) = \text{softmax}(x)$
- Jacobian: $\frac{\partial \text{softmax}(x_i)}{\partial x_j} = \text{softmax}(x_i)(\mathbb{1}[i=j] - \text{softmax}(x_j))$

---

## Appendix B: Matrix Calculus Identities

$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$$

$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$

$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{A}\mathbf{X}^\top) = \mathbf{A}$$

$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}\mathbf{A}\mathbf{X}^\top\mathbf{B}) = \mathbf{B}^\top\mathbf{X}\mathbf{A}^\top + \mathbf{B}\mathbf{X}\mathbf{A}$$

---

*This document is part of the GRU4Rec Reproduction Study project.*
