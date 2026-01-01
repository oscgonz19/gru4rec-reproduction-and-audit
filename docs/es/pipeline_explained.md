# Pipeline Explicado: Sistema de Recomendacion Basado en Sesiones

**Audiencia:** Data Scientists, Ingenieros de ML
**Prerrequisitos:** Comprension basica de redes neuronales y sistemas de recomendacion

---

## Vision General

Este documento explica como funciona el pipeline de prediccion de GRU4Rec, desde datos crudos de sesiones hasta recomendaciones ordenadas.

<p align="center">
  <img src="../../figures/pipeline.png" alt="Vista General del Pipeline" width="100%">
</p>

---

## 1. Pipeline de Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Pipeline de Datos End-to-End                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ETAPA 1: Ingestion de Datos                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Sesiones Crudas (TSV)                                          │   │
│   │  ┌─────────┬─────────┬─────────────┐                            │   │
│   │  │SessionId│ ItemId  │    Time     │                            │   │
│   │  ├─────────┼─────────┼─────────────┤                            │   │
│   │  │    1    │   42    │ 1609459200  │                            │   │
│   │  │    1    │   17    │ 1609459210  │                            │   │
│   │  │    1    │   89    │ 1609459220  │                            │   │
│   │  │    2    │   42    │ 1609459300  │                            │   │
│   │  └─────────┴─────────┴─────────────┘                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ETAPA 2: Preprocesamiento                                             │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  a) Ordenar por SessionId, Time                                 │   │
│   │  b) Calcular tiempos de inicio de sesion                        │   │
│   │  c) Division temporal (80/20 por tiempo de inicio)              │   │
│   │  d) Filtrar items de test no en vocabulario de entrenamiento   │   │
│   │  e) Eliminar sesiones con <2 items                              │   │
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
│   ETAPA 3: Entrenamiento de Modelo     │                                │
│   ┌─────────────────────┐              │                                │
│   │  Para cada epoca:   │              │                                │
│   │    Para cada batch: │              │                                │
│   │      1. Obtener ses │              │                                │
│   │      2. Forward GRU │              │                                │
│   │      3. Calcular loss│             │                                │
│   │      4. Backprop    │              │                                │
│   │      5. Actualizar  │              │                                │
│   └─────────────────────┘              │                                │
│              │                         │                                │
│              ▼                         ▼                                │
│   ETAPA 4: Evaluacion                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Para cada sesion de test [i1, i2, ..., in]:                   │   │
│   │    Para t = 1 hasta n-1:                                        │   │
│   │      historia = [i1, ..., it]                                   │   │
│   │      objetivo = i(t+1)                                          │   │
│   │      scores = modelo.predecir(historia)  # TODOS los items      │   │
│   │      rank = posicion del objetivo en scores ordenados           │   │
│   │      actualizar Recall@K, MRR@K                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pasada Forward de GRU4Rec

<p align="center">
  <img src="../../figures/architecture.png" alt="Arquitectura GRU4Rec" width="70%">
</p>

### 2.1 Codificacion de Sesion

```
Sesion de Entrada: [zapato_42, calcetin_17, botella_89]

Paso 1: Busqueda de Embedding de Item
┌─────────────────────────────────────────┐
│  Matriz de Embedding E (V × D)          │
│  V = tamano del vocabulario (items)     │
│  D = dimension del embedding            │
│                                         │
│  zapato_42  → e_42  = [0.2, -0.1, ...]  │
│  calcetin_17→ e_17  = [0.5,  0.3, ...]  │
│  botella_89 → e_89  = [-0.1, 0.4, ...]  │
└─────────────────────────────────────────┘

Paso 2: Procesamiento GRU Secuencial
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  t=0: h_0 = zeros(H)                                        │
│        ↓                                                     │
│  t=1: h_1 = GRU(h_0, e_42)  ← Procesar zapato               │
│        ↓                                                     │
│  t=2: h_2 = GRU(h_1, e_17)  ← Procesar calcetin             │
│        ↓                                                     │
│  t=3: h_3 = GRU(h_2, e_89)  ← Procesar botella              │
│        ↓                                                     │
│       h_3 = representacion final de la sesion               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Paso 3: Puntuar Todos los Items
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  scores = h_3 @ E.T   (H × 1) @ (D × V).T = (V,)           │
│                                                              │
│  Resultado: score para cada uno de los V items              │
│  [score_0, score_1, ..., score_V-1]                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Paso 4: Ranking
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Top-K items = argsort(scores, descendente)[:K]             │
│                                                              │
│  Recomendacion: [item_234, item_89, item_42, ...]           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Metricas Explicadas

### Recall@K

```
┌─────────────────────────────────────────────────────────────┐
│                      Recall@K                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Definicion: ¿Estaba el item objetivo en el top-K?          │
│                                                              │
│  Recall@K = (1 si rank ≤ K sino 0)                          │
│                                                              │
│  Ejemplo:                                                    │
│    Predicciones: [A, B, C, D, E, F, G, H, I, J]             │
│    Objetivo: D (rank = 4)                                   │
│                                                              │
│    Recall@3 = 0  (D no esta en top 3)                       │
│    Recall@5 = 1  (D esta en top 5)                          │
│    Recall@10 = 1 (D esta en top 10)                         │
│                                                              │
│  Interpretacion:                                             │
│    Recall@20 = 0.35 significa que 35% de objetivos          │
│    estan en el top 20                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### MRR@K (Rango Reciproco Medio)

```
┌─────────────────────────────────────────────────────────────┐
│                        MRR@K                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Definicion: Rango Reciproco Medio                          │
│                                                              │
│  MRR@K = (1/rango si rango ≤ K sino 0)                      │
│                                                              │
│  Ejemplo:                                                    │
│    Predicciones: [A, B, C, D, E, F, G, H, I, J]             │
│    Objetivo: D (rango = 4)                                  │
│                                                              │
│    MRR@3 = 0      (rango > 3)                               │
│    MRR@5 = 0.25   (1/4 = 0.25)                              │
│    MRR@10 = 0.25  (1/4 = 0.25)                              │
│                                                              │
│  Interpretacion:                                             │
│    MRR@20 = 0.13 significa que el rango promedio ≈ 7.7     │
│    Mayor MRR = objetivo aparece mas arriba en el ranking    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Referencia de Comandos Make

```bash
# Pipeline de Datos
make synth_data    # Generar sesiones sinteticas
make preprocess    # Division temporal train/test

# Pipeline de Modelo
make fetch         # Clonar GRU4Rec oficial
make train_tiny    # Entrenar modelo pequeno para testing
make eval_tiny     # Evaluar modelo entrenado

# Pipeline de Baselines
make baselines     # Ejecutar popularidad + Markov

# Pipeline Completo
make ci            # Verificacion completa de reproducibilidad

# Utilidades
make test          # Ejecutar pytest
make clean         # Eliminar archivos generados
make help          # Mostrar todos los targets
```

---

*Este documento es parte del proyecto de Estudio de Reproducibilidad de GRU4Rec.*
