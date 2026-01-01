# Pipeline Explained: Session-Based Recommendation System

**Audience:** Data Scientists, ML Engineers
**Prerequisites:** Basic understanding of neural networks and recommendation systems

---

## Overview

This document explains how the GRU4Rec prediction pipeline works, from raw session data to ranked recommendations.

<p align="center">
  <img src="../../figures/pipeline.png" alt="Pipeline Overview" width="100%">
</p>

---

## 1. Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        End-to-End Data Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STAGE 1: Data Ingestion                                               │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Raw Sessions (TSV)                                              │   │
│   │  ┌─────────┬─────────┬─────────────┐                            │   │
│   │  │SessionId│ ItemId  │    Time     │                            │   │
│   │  ├─────────┼─────────┼─────────────┤                            │   │
│   │  │    1    │   42    │ 1609459200  │                            │   │
│   │  │    1    │   17    │ 1609459210  │                            │   │
│   │  │    1    │   89    │ 1609459220  │                            │   │
│   │  │    2    │   42    │ 1609459300  │                            │   │
│   │  │   ...   │   ...   │    ...      │                            │   │
│   │  └─────────┴─────────┴─────────────┘                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   STAGE 2: Preprocessing                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  a) Sort by SessionId, Time                                      │   │
│   │  b) Compute session start times                                  │   │
│   │  c) Temporal split (80/20 by session start time)                │   │
│   │  d) Filter test items not in training vocabulary                │   │
│   │  e) Remove sessions with <2 items                               │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                    ┌─────────┴─────────┐                                │
│                    ▼                   ▼                                │
│              ┌──────────┐        ┌──────────┐                           │
│              │  Train   │        │   Test   │                           │
│              │  (80%)   │        │  (20%)   │                           │
│              └──────────┘        └──────────┘                           │
│                    │                   │                                │
│                    ▼                   │                                │
│   STAGE 3: Model Training              │                                │
│   ┌─────────────────────┐              │                                │
│   │  For each epoch:    │              │                                │
│   │    For each batch:  │              │                                │
│   │      1. Get session │              │                                │
│   │      2. Forward GRU │              │                                │
│   │      3. Compute loss│              │                                │
│   │      4. Backprop    │              │                                │
│   │      5. Update      │              │                                │
│   └─────────────────────┘              │                                │
│              │                         │                                │
│              ▼                         ▼                                │
│   STAGE 4: Evaluation                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  For each test session [i1, i2, ..., in]:                       │   │
│   │    For t = 1 to n-1:                                            │   │
│   │      history = [i1, ..., it]                                    │   │
│   │      target = i(t+1)                                            │   │
│   │      scores = model.predict(history)  # ALL items               │   │
│   │      rank = position of target in sorted scores                 │   │
│   │      update Recall@K, MRR@K                                     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   STAGE 5: Results                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Recall@5:  0.1867    MRR@5:  0.1172                            │   │
│   │  Recall@10: 0.2632    MRR@10: 0.1271                            │   │
│   │  Recall@20: 0.3428    MRR@20: 0.1324                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. GRU4Rec Forward Pass

<p align="center">
  <img src="../../figures/architecture.png" alt="GRU4Rec Architecture" width="70%">
</p>

### 2.1 Session Encoding

```
Input Session: [shoe_42, sock_17, bottle_89]

Step 1: Item Embedding Lookup
┌─────────────────────────────────────────┐
│  Embedding Matrix E (V × D)             │
│  V = vocabulary size (number of items)  │
│  D = embedding dimension                │
│                                         │
│  shoe_42  → e_42  = [0.2, -0.1, ...]   │
│  sock_17  → e_17  = [0.5,  0.3, ...]   │
│  bottle_89→ e_89  = [-0.1, 0.4, ...]   │
└─────────────────────────────────────────┘

Step 2: Sequential GRU Processing
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  t=0: h_0 = zeros(H)                                        │
│        ↓                                                     │
│  t=1: h_1 = GRU(h_0, e_42)  ← Process shoe                  │
│        ↓                                                     │
│  t=2: h_2 = GRU(h_1, e_17)  ← Process sock                  │
│        ↓                                                     │
│  t=3: h_3 = GRU(h_2, e_89)  ← Process bottle                │
│        ↓                                                     │
│       h_3 = final session representation                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Step 3: Score All Items
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  scores = h_3 @ E.T   (H × 1) @ (D × V).T = (V,)           │
│                                                              │
│  Result: score for each of V items                          │
│  [score_0, score_1, ..., score_V-1]                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Step 4: Ranking
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Top-K items = argsort(scores, descending)[:K]              │
│                                                              │
│  Recommendation: [item_234, item_89, item_42, ...]          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 GRU Cell Computation

```
┌─────────────────────────────────────────────────────────────┐
│                    GRU Cell Detail                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Inputs:                                                     │
│    h_{t-1} : previous hidden state (H,)                     │
│    x_t     : current item embedding (D,)                    │
│                                                              │
│  Computations:                                               │
│                                                              │
│    ┌─────────────────────────────────────────────┐          │
│    │  z_t = σ(W_z · [h_{t-1}, x_t] + b_z)       │  Update  │
│    │        ↓                                    │  Gate    │
│    │  Determines how much of h_{t-1} to keep    │          │
│    └─────────────────────────────────────────────┘          │
│                                                              │
│    ┌─────────────────────────────────────────────┐          │
│    │  r_t = σ(W_r · [h_{t-1}, x_t] + b_r)       │  Reset   │
│    │        ↓                                    │  Gate    │
│    │  Determines how much of h_{t-1} to forget  │          │
│    └─────────────────────────────────────────────┘          │
│                                                              │
│    ┌─────────────────────────────────────────────┐          │
│    │  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b) │ Candidate│
│    │        ↓                                    │  State   │
│    │  New candidate hidden state                │          │
│    └─────────────────────────────────────────────┘          │
│                                                              │
│    ┌─────────────────────────────────────────────┐          │
│    │  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t   │  Output  │
│    │        ↓                                    │          │
│    │  Blend old state with candidate            │          │
│    └─────────────────────────────────────────────┘          │
│                                                              │
│  Output:                                                     │
│    h_t : new hidden state (H,)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Training Pipeline

### 3.1 Mini-Batch Construction

```
┌─────────────────────────────────────────────────────────────┐
│                 Session-Parallel Mini-Batches                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Sessions in batch (batch_size=4):                          │
│                                                              │
│  Session 1: [A, B, C, D, E]                                 │
│  Session 2: [F, G, H]                                       │
│  Session 3: [I, J, K, L]                                    │
│  Session 4: [M, N]                                          │
│                                                              │
│  Time step 0:                                               │
│    Input:  [A, F, I, M]                                     │
│    Target: [B, G, J, N]                                     │
│                                                              │
│  Time step 1:                                               │
│    Input:  [B, G, J, N]                                     │
│    Target: [C, H, K, -]  ← Session 4 finished              │
│                                                              │
│  Time step 2:                                               │
│    Input:  [C, -, K, -]  ← Replace finished with new       │
│    Target: [D, -, L, -]                                     │
│                                                              │
│  ... continues until all sessions processed                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Loss Computation

```
┌─────────────────────────────────────────────────────────────┐
│                    Cross-Entropy Loss                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For each (input, target) pair:                             │
│                                                              │
│  1. Forward pass: scores = model(input)  # shape: (V,)      │
│                                                              │
│  2. Softmax: probs = softmax(scores)                        │
│              probs_i = exp(s_i) / Σ_j exp(s_j)              │
│                                                              │
│  3. Loss: L = -log(probs[target])                           │
│           = -scores[target] + log(Σ_j exp(scores_j))        │
│                                                              │
│  4. With negative sampling (efficiency):                    │
│     - Sample n_sample negative items                        │
│     - Compute loss only over target + negatives             │
│     - Much faster than full softmax over V items            │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    BPR-Max Loss (Alternative)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L = -log(σ(s_target - max(s_negatives)))                   │
│      + λ · Σ_j (σ(s_j)² · sample_weight_j)                  │
│                                                              │
│  Intuition:                                                  │
│  - Push target score above hardest negative                 │
│  - Regularize negative scores toward zero                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Evaluation Pipeline

### 4.1 Full Ranking Protocol

```
┌─────────────────────────────────────────────────────────────┐
│                Full Ranking Evaluation                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  WHY full ranking?                                          │
│  - Sampled evaluation inflates metrics 2-3x                 │
│  - Production systems rank ALL items                        │
│  - More realistic performance estimates                     │
│                                                              │
│  Algorithm:                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  for session in test_sessions:                      │    │
│  │      items = session.items                          │    │
│  │      for t in range(len(items) - 1):                │    │
│  │          history = items[:t+1]                      │    │
│  │          target = items[t+1]                        │    │
│  │                                                     │    │
│  │          # Score ALL V items                        │    │
│  │          scores = model.score_all(history)          │    │
│  │                                                     │    │
│  │          # Find rank of target                      │    │
│  │          rank = (scores >= scores[target]).sum()    │    │
│  │                                                     │    │
│  │          # Update metrics                           │    │
│  │          for k in [5, 10, 20]:                      │    │
│  │              if rank <= k:                          │    │
│  │                  recall[k] += 1                     │    │
│  │                  mrr[k] += 1/rank                   │    │
│  │          total += 1                                 │    │
│  │                                                     │    │
│  │  # Average                                          │    │
│  │  recall = {k: v/total for k,v in recall.items()}   │    │
│  │  mrr = {k: v/total for k,v in mrr.items()}         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Metrics Explained

```
┌─────────────────────────────────────────────────────────────┐
│                      Recall@K                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Definition: Was the target item in the top-K?              │
│                                                              │
│  Recall@K = (1 if rank ≤ K else 0)                          │
│                                                              │
│  Example:                                                    │
│    Predictions: [A, B, C, D, E, F, G, H, I, J]              │
│    Target: D (rank = 4)                                     │
│                                                              │
│    Recall@3 = 0  (D not in top 3)                           │
│    Recall@5 = 1  (D is in top 5)                            │
│    Recall@10 = 1 (D is in top 10)                           │
│                                                              │
│  Interpretation:                                             │
│    Recall@20 = 0.35 means 35% of targets are in top 20     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        MRR@K                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Definition: Mean Reciprocal Rank                           │
│                                                              │
│  MRR@K = (1/rank if rank ≤ K else 0)                        │
│                                                              │
│  Example:                                                    │
│    Predictions: [A, B, C, D, E, F, G, H, I, J]              │
│    Target: D (rank = 4)                                     │
│                                                              │
│    MRR@3 = 0      (rank > 3)                                │
│    MRR@5 = 0.25   (1/4 = 0.25)                              │
│    MRR@10 = 0.25  (1/4 = 0.25)                              │
│                                                              │
│  Interpretation:                                             │
│    MRR@20 = 0.13 means average rank ≈ 7.7 when found       │
│    Higher MRR = target appears higher in ranking            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Baseline Pipelines

### 5.1 Popularity Baseline

```
┌─────────────────────────────────────────────────────────────┐
│                   Popularity Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Training:                                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  item_counts = Counter(train_df['ItemId'])          │    │
│  │  top_items = sorted by count, descending            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Prediction:                                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  def predict(session, k):                           │    │
│  │      return top_items[:k]  # Ignore session!        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Complexity: O(1) prediction                                │
│                                                              │
│  Why it works:                                               │
│  - Power law distribution of item popularity                │
│  - Popular items are often relevant                         │
│  - Strong baseline, especially with sparse data             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Markov Chain Baseline

```
┌─────────────────────────────────────────────────────────────┐
│                   Markov Chain Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Training:                                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  transitions = defaultdict(Counter)                 │    │
│  │  for session in sessions:                           │    │
│  │      for i in range(len(session) - 1):              │    │
│  │          transitions[session[i]][session[i+1]] += 1 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Transition matrix example:                                  │
│  ┌────────┬──────┬──────┬──────┬──────┐                    │
│  │ From\To│ shoe │ sock │bottle│ bag  │                    │
│  ├────────┼──────┼──────┼──────┼──────┤                    │
│  │ shoe   │  5   │  20  │  10  │  3   │                    │
│  │ sock   │  8   │  2   │  15  │  5   │                    │
│  │ bottle │  3   │  7   │  1   │  12  │                    │
│  │ bag    │  10  │  5   │  8   │  2   │                    │
│  └────────┴──────┴──────┴──────┴──────┘                    │
│                                                              │
│  Prediction:                                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  def predict(session, k):                           │    │
│  │      last_item = session[-1]                        │    │
│  │      candidates = transitions[last_item]            │    │
│  │      return top_k(candidates, k)                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Example: session = [shoe, sock, ?]                         │
│    → Look up transitions from 'sock'                        │
│    → bottle(15) > shoe(8) > bag(5) > sock(2)               │
│    → Predict: [bottle, shoe, bag, ...]                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Make Targets Reference

```bash
# Data Pipeline
make synth_data    # Generate synthetic sessions
make preprocess    # Temporal train/test split

# Model Pipeline
make fetch         # Clone official GRU4Rec
make train_tiny    # Train small model for testing
make eval_tiny     # Evaluate trained model

# Baseline Pipeline
make baselines     # Run popularity + Markov

# Full Pipeline
make ci            # Complete reproducibility check

# Utilities
make test          # Run pytest
make clean         # Remove generated files
make help          # Show all targets
```

---

*This document is part of the GRU4Rec Reproduction Study project.*
