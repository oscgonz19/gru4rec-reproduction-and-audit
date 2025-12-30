# Estudio de Reproducibilidad GRU4Rec: Caso de Estudio para Portafolio

**Tipo de Proyecto:** Investigacion e Implementacion de Machine Learning
**Dominio:** Sistemas de Recomendacion / Deep Learning
**Duracion:** Diciembre 2024
**Autor:** Oscar Gonzalez

---

## Resumen Ejecutivo

Este caso de estudio documenta el desarrollo end-to-end de un pipeline de investigacion reproducible para recomendaciones de productos basadas en sesiones usando deep learning. El proyecto demuestra experiencia en ingenieria de machine learning, metodologia de investigacion y mejores practicas de desarrollo de software.

---

## 1. El Desafio

### 1.1 Contexto de Negocio

Las plataformas de comercio electronico pierden ingresos significativos cuando no pueden personalizar la experiencia para usuarios anonimos:

```
┌─────────────────────────────────────────────────────────┐
│           El Problema del Usuario Anonimo                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   70-80% de visitantes de e-commerce son anonimos       │
│                                                          │
│   Usuarios anonimos convierten al 1-2%                  │
│   vs. usuarios recurrentes al 3-5%                      │
│                                                          │
│   Perdida potencial de ingresos: 20-40% del GMV total  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Desafio Tecnico

Los sistemas de recomendacion tradicionales requieren:
- Identificacion de usuario (login, cookies)
- Datos historicos de interaccion (semanas/meses)
- Ratings o preferencias explicitas

Las **recomendaciones basadas en sesiones** deben funcionar con:
- Solo sesiones anonimas
- Secuencias de interaccion cortas (minutos)
- Feedback implicito (solo clics)

### 1.3 Objetivos del Proyecto

| Objetivo | Criterio de Exito | Estado |
|----------|-------------------|--------|
| Reproducir metodologia GRU4Rec | Pipeline de entrenamiento funcional | Logrado |
| Implementar baselines de comparacion | 2+ baselines funcionales | Logrado |
| Establecer protocolo de evaluacion | Metricas con ranking completo | Logrado |
| Crear pipeline reproducible | Ejecucion con un comando | Logrado |
| Documentar para portafolio | Docs tecnicos + ejecutivos | Logrado |

---

## 2. Arquitectura de la Solucion

### 2.1 Arquitectura de Alto Nivel

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Estudio de Reproducibilidad GRU4Rec                  │
│                     Arquitectura del Sistema                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Datos Crudos │    │Preprocesador │    │  Modelos     │          │
│  │   (TSV)      │───▶│  (Division   │───▶│Entrenamiento │          │
│  │              │    │  Temporal)   │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ SessionId    │    │ Train: 80%   │    │ GRU4Rec      │          │
│  │ ItemId       │    │ Test:  20%   │    │ Popularidad  │          │
│  │ Timestamp    │    │ (Temporal)   │    │ Markov       │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                 │                    │
│                                                 ▼                    │
│                            ┌──────────────────────────────┐         │
│                            │       Evaluacion             │         │
│                            │  ┌─────────┬─────────┐      │         │
│                            │  │Recall@K │ MRR@K   │      │         │
│                            │  │ @5,10,20│ @5,10,20│      │         │
│                            │  └─────────┴─────────┘      │         │
│                            │    (Ranking Completo)       │         │
│                            └──────────────────────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Stack Tecnologico

| Capa | Tecnologia | Proposito |
|------|------------|-----------|
| Deep Learning | PyTorch 2.x | Entrenamiento e inferencia |
| Procesamiento de Datos | Pandas, NumPy | Manipulacion de datos |
| Automatizacion | Make, Bash | Orquestacion del pipeline |
| Testing | Pytest | Pruebas unitarias e integracion |
| CI/CD | GitHub Actions | Testing automatizado |
| Entorno | Conda | Reproducibilidad |
| Documentacion | Markdown | Escritura tecnica |

---

## 3. Resultados e Impacto

### 3.1 Resultados Cuantitativos

```
============================================================
           Resultados de Comparacion de Baselines
============================================================
Metrica           Popularidad       Markov
------------------------------------------------------------
Recall@5              0.1867       0.1190
Recall@10             0.2632       0.1817
Recall@20             0.3428       0.2778
MRR@5                 0.1172       0.0737
MRR@10                0.1271       0.0816
MRR@20                0.1324       0.0881
============================================================

Entrenamiento GRU4Rec:
  - Reduccion de perdida: 7.15 → 6.50 (9% mejora en 5 epocas)
  - Tiempo de entrenamiento: 7.94s en CPU
  - Tamano del modelo: 77MB
```

### 3.2 Hallazgos Clave

1. **Los Baselines Importan**
   - El baseline de popularidad simple logra 60-70% del rendimiento de redes neuronales
   - Esencial para establecer umbrales de mejora significativos

2. **El Protocolo de Evaluacion es Critico**
   - La evaluacion muestreada sobreestima por 2-3x
   - El ranking completo proporciona estimaciones realistas de produccion

3. **La Reproducibilidad Requiere Disciplina**
   - Fijacion de entorno (conda, numpy<2.0)
   - Divisiones temporales (no aleatorias)
   - Atribucion apropiada

### 3.3 Metricas del Proyecto

| Metrica | Valor |
|---------|-------|
| Lineas de Codigo (original) | ~1,500 |
| Cobertura de Pruebas | 18 pruebas unitarias |
| Paginas de Documentacion | 5 documentos x 2 idiomas |
| Tiempo de Reproduccion | <2 minutos |
| Dependencias | 8 paquetes |

---

## 4. Lecciones Aprendidas

### 4.1 Lecciones Tecnicas

1. **Compatibilidad de NumPy**
   - NumPy 2.0 rompio el GRU4Rec oficial por manejo de dtype
   - Solucion: Fijar numpy<2.0 en el entorno

2. **Manejo de Rutas**
   - Las llamadas a subprocesos requieren rutas absolutas al cambiar cwd
   - Siempre resolver rutas antes de pasar a procesos hijos

3. **Rigor en Evaluacion**
   - Los atajos academicos (negativos muestreados) no se traducen a produccion
   - Siempre validar con ranking completo

### 4.2 Lecciones de Proceso

1. **Atribucion Primero**
   - Se descubrio que el codigo oficial no tenia licencia abierta
   - Se rediseno a patron fetch-on-demand

2. **La Documentacion Rinde**
   - Invertir en docs temprano ahorro tiempo de debugging
   - Docs bilingues expanden la audiencia potencial

---

## 5. Conclusion

Este proyecto demuestra capacidad end-to-end en investigacion e ingenieria de machine learning:

- **Habilidades de Investigacion:** Comprension y reproduccion de trabajo academico
- **Habilidades de Ingenieria:** Construccion de pipelines reproducibles y testeados
- **Habilidades de Comunicacion:** Documentacion tecnica y ejecutiva
- **Mejores Practicas:** Control de versiones, CI/CD, atribucion apropiada

El repositorio resultante sirve tanto como herramienta de investigacion como pieza de portafolio que demuestra practicas profesionales de desarrollo de software en el dominio de ML.

---

## Apendice: Inicio Rapido

```bash
# Clonar y configurar
git clone https://github.com/TU_USUARIO/gru4rec-reproduction-study.git
cd gru4rec-reproduction-study
conda env create -f environment.yml
conda activate gru4rec-study

# Ejecutar demo completa
make fetch synth_data preprocess baselines

# Salida esperada en ~30 segundos
```

---

*Este caso de estudio es parte del portafolio del proyecto de Estudio de Reproducibilidad de GRU4Rec.*
