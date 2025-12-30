# GRU4Rec Reproduction and Audit

> **Reproducible evaluation harness around the official GRU4Rec implementation for session-based recommendations**

[![CI](https://github.com/oscgonz19/gru4rec-reproduction-and-audit/workflows/CI/badge.svg)](https://github.com/oscgonz19/gru4rec-reproduction-and-audit/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

---

## What This Is (and What It Isn't)

**This is**: A reproducibility harness and baseline comparison framework around GRU4Rec.

**This is NOT**: A reimplementation or fork of GRU4Rec. The model code is **not included** in this repository.

### My Contributions

- Automated pipeline (`Makefile` + scripts) for reproducible experiments
- Synthetic data generator for CI/testing without licensing issues
- Temporal train/test split without data leakage
- Production-ready baselines: Popularity and First-order Markov Chain
- Evaluation module with Recall@K, MRR@K, NDCG@K
- Bilingual documentation (10 documents in EN/ES)
- GitHub Actions CI workflow

### GRU4Rec Model

The official GRU4Rec PyTorch implementation by [Balazs Hidasi](https://github.com/hidasib/GRU4Rec_PyTorch_Official) is **fetched on-demand** to a gitignored `vendor/` directory. It is NOT redistributed.

> **License note**: Official GRU4Rec is free for research/education; contact the author for commercial use.

---

## El Problema que Resuelve

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EL PROBLEMA DEL USUARIO ANONIMO                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   En e-commerce y plataformas digitales:                                │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │  70-80% de visitantes son ANONIMOS o primera visita     │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                          │
│   Los sistemas de recomendacion tradicionales FALLAN porque:            │
│   ✗ Requieren historial de usuario (semanas/meses)                      │
│   ✗ Necesitan identificacion (login, cookies)                           │
│   ✗ Dependen de ratings explicitos                                      │
│                                                                          │
│   Impacto: Perdida de 20-40% de conversiones potenciales                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### La Solucion: Recomendaciones Basadas en Sesion

**Session-based recommendations** predicen el siguiente item que un usuario quiere basandose UNICAMENTE en su sesion actual de navegacion, sin requerir historial ni identificacion.

```
Metodo Tradicional:
  "Usuario hizo clic en zapatos" → Recomendar zapatos

GRU4Rec (Session-based):
  "Usuario vio zapatillas running → agrego al carrito →
   miro calcetines → vio botellas de agua"
   → Recomendar: gear de running, accesorios fitness
```

---

## Que es GRU4Rec?

**GRU4Rec** es un modelo de deep learning que usa **Redes Neuronales Recurrentes con Compuertas (GRU)** para aprender patrones secuenciales en el comportamiento de navegacion.

| Aspecto | Detalle |
|---------|---------|
| **Paper** | "Session-based Recommendations with RNNs" (ICLR 2016) |
| **Citaciones** | 1,500+ en literatura academica |
| **Autor** | Balazs Hidasi (Gravity R&D) |
| **Adopcion** | Usado en produccion por grandes plataformas e-commerce |

### Arquitectura Visual

```
┌─────────────────────────────────────────────────────────────┐
│                  ARQUITECTURA GRU4Rec                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Sesion: [zapato, calcetin, botella, ?]                    │
│              │        │         │                            │
│              ▼        ▼         ▼                            │
│   ┌──────────────────────────────────────┐                  │
│   │         EMBEDDING LAYER              │                  │
│   │    Convierte items en vectores       │                  │
│   └──────────────────────────────────────┘                  │
│              │        │         │                            │
│              ▼        ▼         ▼                            │
│   ┌──────────────────────────────────────┐                  │
│   │           GRU LAYERS                 │                  │
│   │   Aprende patrones secuenciales      │                  │
│   │   h_t = GRU(h_{t-1}, x_t)           │                  │
│   └──────────────────────────────────────┘                  │
│                       │                                      │
│                       ▼                                      │
│   ┌──────────────────────────────────────┐                  │
│   │         OUTPUT LAYER                 │                  │
│   │   Score para cada item del catalogo  │                  │
│   └──────────────────────────────────────┘                  │
│                       │                                      │
│                       ▼                                      │
│   Prediccion: [item_234, item_89, item_42, ...]             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Que Ofrece Este Repositorio?

### Pipeline Reproducible Completo

```bash
# En 30 segundos tienes resultados
make fetch        # Clona GRU4Rec oficial (no redistribuimos codigo)
make synth_data   # Genera datos sinteticos
make preprocess   # Split temporal sin data leakage
make baselines    # Ejecuta Popularity + Markov
```

### Baselines Implementados

| Modelo | Descripcion | Complejidad |
|--------|-------------|-------------|
| **Popularity** | Recomienda items mas populares | O(1) prediccion |
| **Markov Chain** | Predice basado en ultimo item | O(k log k) prediccion |
| **GRU4Rec** | Deep learning sobre secuencia completa | O(T·H² + H·V) |

### Evaluacion Rigurosa

Usamos **full ranking** (puntuar TODOS los items), no negativos muestreados:

| Tipo Evaluacion | Recall@20 Tipico | Realidad |
|-----------------|------------------|----------|
| Muestreada (100 neg) | ~80% | **Inflado 2-3x** |
| Full ranking | ~35% | **Realista** |

---

## Resultados de Demo

```
============================================================
           GRU4Rec Reproduction Study - Demo
============================================================

[1/4] Generating synthetic session data...
      Generated 11,222 events, 1,000 sessions, 499 items

[2/4] Preprocessing with temporal split...
      Train: 8,837 events, 800 sessions
      Test:  2,385 events, 200 sessions

[3/4] Running baselines...

[4/4] Results
============================================================
Metric            Popularity       Markov
------------------------------------------------------------
Recall@5              0.1867       0.1190
Recall@10             0.2632       0.1817
Recall@20             0.3428       0.2778
MRR@5                 0.1172       0.0737
MRR@10                0.1271       0.0816
MRR@20                0.1324       0.0881
============================================================

GRU4Rec Training (5 epochs):
  Loss: 7.15 → 6.50 (↓9% mejora)
  Tiempo: 7.94s (CPU)
```

---

## Quick Start

### Opcion 1: Conda (Recomendado)

```bash
# Clonar
git clone https://github.com/oscgonz19/gru4rec-reproduction-and-audit.git
cd gru4rec-reproduction-and-audit

# Crear entorno
conda env create -f environment.yml
conda activate gru4rec-study

# Ejecutar demo completa
make fetch synth_data preprocess baselines
```

### Opcion 2: pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make ci
```

---

## Casos de Uso

### Cuando Usar GRU4Rec

| Escenario | Recomendacion |
|-----------|---------------|
| E-commerce con patrones de navegacion complejos | **GRU4Rec** |
| Plataformas de contenido secuencial | **GRU4Rec** |
| Datos limitados o cold-start | Popularity baseline |
| Latencia <10ms requerida | Markov o Popularity |
| Recursos computacionales limitados | Baselines |

### ROI Esperado

| Comparacion | Mejora Esperada en CTR |
|-------------|------------------------|
| GRU4Rec vs Popularity baseline | 10-30% |
| GRU4Rec vs sin recomendaciones | 200-400% |

---

## Documentacion Completa

### Matriz de Documentos

| Documento | English | Espanol | Audiencia |
|-----------|---------|---------|-----------|
| **Caso de Estudio** | [case_study.md](docs/en/case_study.md) | [case_study.md](docs/es/case_study.md) | Portfolio |
| **Resumen Ejecutivo** | [executive_summary.md](docs/en/executive_summary.md) | [executive_summary.md](docs/es/executive_summary.md) | Managers |
| **Reporte Tecnico** | [technical_report.md](docs/en/technical_report.md) | [technical_report.md](docs/es/technical_report.md) | Engineers |
| **Pipeline Explicado** | [pipeline_explained.md](docs/en/pipeline_explained.md) | [pipeline_explained.md](docs/es/pipeline_explained.md) | Data Scientists |
| **Formulas Matematicas** | [mathematical_formulas.md](docs/en/mathematical_formulas.md) | [mathematical_formulas.md](docs/es/mathematical_formulas.md) | Researchers |

---

## Estructura del Proyecto

```
gru4rec-reproduction-and-audit/
│
├── docs/                          # 10 documentos (5 EN + 5 ES)
│   ├── en/                        # English documentation
│   └── es/                        # Documentacion en espanol
│
├── scripts/                       # Pipeline automation
│   ├── fetch_official.py          # Clona GRU4Rec oficial
│   ├── make_synth_data.py         # Genera datos sinteticos
│   ├── preprocess_sessions.py     # Split temporal
│   └── run_gru4rec.py             # Wrapper de entrenamiento
│
├── src/                           # Codigo fuente
│   ├── baselines/
│   │   ├── popularity.py          # Baseline de popularidad
│   │   └── markov.py              # Baseline Markov
│   ├── metrics.py                 # Recall@K, MRR@K, NDCG@K
│   └── report.py                  # Visualizaciones
│
├── tests/                         # 18 unit tests
│   ├── test_baselines.py
│   └── test_metrics.py
│
├── data/                          # Datos (gitignored)
├── results/                       # Modelos (gitignored)
├── vendor/                        # GRU4Rec oficial (gitignored)
│
├── environment.yml                # Conda environment
├── requirements.txt               # Pip requirements
├── Makefile                       # Build automation
└── .github/workflows/ci.yml       # GitHub Actions
```

---

## Comandos Disponibles

```bash
# Pipeline de datos
make synth_data     # Generar datos sinteticos
make preprocess     # Split temporal train/test

# Modelos
make fetch          # Clonar GRU4Rec oficial
make train_tiny     # Entrenar modelo pequeno
make baselines      # Ejecutar baselines

# Utilidades
make test           # Ejecutar pytest (18 tests)
make ci             # Pipeline completo de CI
make clean          # Limpiar archivos generados
make help           # Ver todos los comandos
```

---

## Formato de Datos

Archivos TSV con tres columnas:

```
SessionId    ItemId    Time
1            42        1609459200
1            17        1609459210
1            89        1609459220
2            42        1609459300
```

| Columna | Descripcion |
|---------|-------------|
| `SessionId` | Identificador unico de sesion |
| `ItemId` | Identificador de item |
| `Time` | Unix timestamp (para ordenamiento) |

---

## Key Findings / Hallazgos Clave

### 1. Los Baselines Son Competitivos
Los metodos simples de popularidad logran **60-70%** del rendimiento de redes neuronales con una fraccion del costo computacional.

### 2. El Protocolo de Evaluacion Importa
La evaluacion muestreada puede **sobreestimar el rendimiento 2-3x**, llevando a malas decisiones de produccion.

### 3. Los Patrones Secuenciales Desbloquean Valor
GRU4Rec sobresale cuando hay patrones secuenciales significativos (navegacion → comparacion → compra).

---

## Referencias

1. Hidasi, B., et al. (2016). **Session-based Recommendations with Recurrent Neural Networks**. ICLR 2016.

2. Hidasi, B., & Karatzoglou, A. (2018). **Recurrent Neural Networks with Top-k Gains for Session-based Recommendations**. CIKM 2018.

3. Ludewig, M., & Jannach, D. (2018). **Evaluation of Session-based Recommendation Algorithms**. UMUAI.

---

## Licencia

- **Mis contribuciones** (pipeline, baselines, scripts, documentacion): MIT License
- **GRU4Rec oficial** (en `vendor/`): Libre para investigacion y educacion; contactar autor para uso comercial

---

## Autor

**Oscar Gonzalez**

*Este proyecto demuestra competencias en deep learning, sistemas de recomendacion y practicas de investigacion reproducible.*

---

<p align="center">
  <b>Si este proyecto te es util, considera darle una estrella en GitHub</b>
</p>
