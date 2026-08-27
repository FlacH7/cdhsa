# Referencia Completa de Parámetros del Pipeline CD-HSA

## Descripción General

El pipeline CD-HSA (*Common Directions — Hierarchical Subspace Analysis*) analiza la estructura de subespacios comunes y específicos por condición en datos EEG agrupados. Consta de dos componentes principales:

| Componente | Archivo | Rol |
|---|---|---|
| **Pipeline individual** | `run_cdhsa.py` | Construye matrices de Hankel desde EEG, ejecuta CD-HSA y guarda resultados. Se invoca desde la CLI o como módulo Python. |
| **Batch runner** | `run_batch_cdhsa.py` | Orquesta múltiples ejecuciones del pipeline (una por super-sujeto) leyendo toda la configuración desde un JSON. Incluye checkpoint, log CSV y post-procesamiento comparativo. |

El flujo de datos es:

```
Raw EEG (sujetos individuales)
    ↓  load_super_subject_eeg() — concatenación temporal
Super-sujeto(s)
    ↓  extract_filtered_data_matrix() — filtrado pasa-banda
Datos filtrados (p, T)
    ↓  _build_multivariate_hankel() — embedding de retardo
Matriz de Hankel (p·depth, T-depth+1)
    ↓  run_cdhsa(X, L, config) — análisis CD-HSA completo
CDHSAResult (r₀, modos comunes, modos específicos, testeo estadístico)
```

CD-HSA internamente aplica un **segundo nivel de embedding** (block-Hankel) en el Step A, que escala la dimensión de p·depth a p·depth·L. Este es el cuello de botella de memoria.

---

## Tabla de Contenidos

1. [Formas de Ejecución](#1-formas-de-ejecución)
2. [Parámetros del JSON del Batch Runner](#2-parámetros-del-json-del-batch-runner)
    - 2.1 [Bloque `super_subjects`](#21-bloque-super_subjects)
    - 2.2 [Bloque `sessions`](#22-bloque-sessions)
    - 2.3 [Bloque `tasks`](#23-bloque-tasks)
    - 2.4 [Bloque `time_window`](#24-bloque-time_window)
    - 2.5 [Bloque `cdhsa_params`](#25-bloque-cdhsa_params)
    - 2.6 [Bloque `execution`](#26-bloque-execution)
3. [Parámetros de la CLI (`run_cdhsa.py`)](#3-parámetros-de-la-cli-run_cdhsapy)
    - 3.1 [Modo Multi-Super-Subject](#31-modo-multi-super-subject)
    - 3.2 [Modo Single-Super-Subject](#32-modo-single-super-subject)
    - 3.3 [Parámetros Comunes](#33-parámetros-comunes)
    - 3.4 [Parámetros CD-HSA](#34-parámetros-cd-hsa)
4. [CDHSAConfig — Parámetros Completos (API Python)](#4-cdhsaconfig--parámetros-completos-api-python)
    - 4.1 [Rank y Subespacio Común (Step A)](#41-rank-y-subespacio-común-step-a)
    - 4.2 [Test de Rango Común (Step A6)](#42-test-de-rango-común-step-a6)
    - 4.3 [Test de Condición — Energía y Geometría (Steps B/C)](#43-test-de-condición--energía-y-geometría-steps-bc)
    - 4.4 [Test de Geometría Tangente](#44-test-de-geometría-tangente)
    - 4.5 [Modos Específicos por Condición (Step D)](#45-modos-específicos-por-condición-step-d)
    - 4.6 [Flags de Salteo](#46-flags-de-salteo)
5. [Mapeo JSON → CLI → CDHSAConfig](#5-mapeo-json--cli--cdhsaconfig)
6. [Cuándo Usar Cada Modo](#6-cuándo-usar-cada-modo)
7. [Ejemplos de Configuración JSON](#7-ejemplos-de-configuración-json)
8. [Estructura de Directorios de Salida](#8-estructura-de-directorios-de-salida)
9. [Archivos Generados por Job](#9-archivos-generados-por-job)
10. [Memoria — Estimación y Límites](#10-memoria--estimación-y-límites)

---

## 1. Formas de Ejecución

El pipeline CD-HSA puede ejecutarse de **dos formas mutuamente excluyentes**, cada una con sus propios parámetros de entrada:

### 1.1 Pipeline Directo (CLI)

Ejecutar `run_cdhsa.py` directamente desde la terminal. Soporta **dos sub-modos** según cómo se especifiquen los super-sujetos:

| Sub-modo | Flag | S (en CD-HSA) | Use case |
|---|---|---|---|
| **Multi-SS** | `--n-super-subjects N` | N (≥ 1) | Analizar subespacios comunes entre N super-sujetos a la vez. Requiere más memoria porque CD-HSA recibe S > 1. |
| **Single-SS** | `--super-subject-id N` | 1 | Analizar un solo super-sujeto. Usado por el batch runner. Menor consumo de memoria. |

Los dos flags son **mutuamente excluyentes**; exactamente uno es requerido.

```bash
# Multi-SS: 3 super-sujetos de 20 sujetos, todo junto
python run_cdhsa.py \
  --n-super-subjects 3 \
  --session session1 \
  --tasks eyesclosed music \
  --L 27 --fixed-rank 25 \
  --t-start 0.0 --t-end 30.0

# Single-SS: solo el super-sujeto 2
python run_cdhsa.py \
  --super-subject-id 2 \
  --session session1 \
  --tasks eyesclosed music \
  --L 27 --fixed-rank 25 \
  --t-start 0.0 --t-end 60.0
```

### 1.2 Batch Runner (JSON)

Ejecutar `run_batch_cdhsa.py` pasando un JSON de configuración. El batch runner **siempre** usa el modo Single-SS internamente: genera un job por cada combinación (super-sujeto × sesión) y despacha cada uno como un subprocess de `run_cdhsa.py --super-subject-id N`.

```bash
python run_batch_cdhsa.py --params-json mi_experimento.json
```

Ventajas del batch runner:
- **Checkpoint**: los jobs completados no se re-ejecutan si se interrumpe.
- **Log CSV**: registro de todos los jobs con timestamp, éxito/fallo y tiempo.
- **Post-procesamiento**: compara resultados entre super-sujetos.
- **Ejecución paralela** (opcional): con `max_workers > 1`.

### 1.3 Cuadro Comparativo

| Aspecto | Pipeline Directo | Batch Runner |
|---|---|---|
| **Entrada** | Flags de CLI | Archivo JSON |
| **Super-sujetos** | 1 (single) o N (multi) | Siempre 1 por job |
| **Memoria por ejecución** | Alta si multi-SS | Baja (S=1 siempre) |
| **Múltiples SS** | Uno solo (multi-SS los procesa juntos) | N jobs independientes |
| **Checkpoint** | No | Sí (JSON) |
| **Post-procesamiento** | No | Sí (comparación cruzada) |
| **Paralelización** | No | Sí (`max_workers`) |
| **Control fino de CDHSAConfig** | Parcial (solo los flags CLI) | Parcial (solo los campos del JSON) |

---

## 2. Parámetros del JSON del Batch Runner

Todo se controla desde un archivo JSON pasado vía `--params-json` (default: `cdhsa_batch_params.json` en el mismo directorio que el script).

```json
{
  "experiment_label": "...",
  "super_subjects": { ... },
  "sessions": [ ... ],
  "tasks": [ ... ],
  "time_window": { ... },
  "cdhsa_params": { ... },
  "execution": { ... }
}
```

### 2.1 Bloque `super_subjects`

Define cómo se construyen los super-sujetos (agrupaciones de sujetos individuales cuyos datos se concatenan temporalmente).

| Campo | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `selected` | `list[int]` | **Sí** | — | Lista de IDs de super-sujetetos a procesar. Cada ID genera un job independiente. Los IDs son 1-indexed (1, 2, 3, ...). |
| `subjects_per_super_subject` | `int` | Sí* | `20` | Cantidad de sujetos individuales por super-sujeto cuando se usa la resolución automática. *No requerido si se usa `groups`. Los `total_subjects` se reparten equitativamente: `total_subjects / len(selected)` da el tamaño si no se especifica. |
| `subject_start_offset` | `int` | No | `1` | Índice del primer sujeto individual. Con offset=1 y `subjects_per_super_subject=20`, el SS1 tiene sujetos [1..20], SS2 tiene [21..40], etc. |
| `total_subjects` | `int` | No | `60` | Total de sujetos disponibles en el dataset. Informativo; el batch runner no lo usa directamente — lo usa `run_cdhsa.py` en modo multi-SS. |
| `groups` | `dict` | No | `null` | Mapeo explícito de `super_subject_id → list[int]` de subject IDs. Si está presente, **toma precedencia** sobre `subjects_per_super_subject`. Permite particiones no-contiguas o de distinto tamaño. Ver nota sobre limitación actual abajo. |

**Resolución automática (sin `groups`):**

Con `selected=[1,2,3]`, `subjects_per_super_subject=20`, `subject_start_offset=1`:

| Super-sujeto | Sujetos |
|---|---|
| 1 | [1, 2, 3, ..., 20] |
| 2 | [21, 22, 23, ..., 40] |
| 3 | [41, 42, 43, ..., 60] |

Con `selected=[1,2,3,4,5]`, `subjects_per_super_subject=12`, `subject_start_offset=1`:

| Super-sujeto | Sujetos |
|---|---|
| 1 | [1, 2, ..., 12] |
| 2 | [13, 14, ..., 24] |
| 3 | [25, 26, ..., 36] |
| 4 | [37, 38, ..., 48] |
| 5 | [49, 50, ..., 60] |

**Resolución explícita (con `groups`):**

```json
"groups": {
  "1": [1, 2, 3, ..., 20],
  "2": [41, 42, 43, ..., 60]
}
```

> **⚠️ Limitación actual (v1):** El campo `groups` en el JSON es leído por el batch runner, pero `run_cdhsa.py` no tiene un flag `--subject-ids`. Los subject IDs explícitos no se pasan al subprocess. Para usar `groups` correctamente, es necesario agregar soporte en `run_cdhsa.py` (flag `--subject-ids`). Con la versión actual, solo funcionan las particiones contiguas (resolución automática).

---

### 2.2 Bloque `sessions`

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| *(valor)* | `list[str]` | **Sí** | Lista de session IDs a procesar. Cada sesión se combina con cada super-sujeto seleccionado, generando `len(selected) × len(sessions)` jobs. |

Ejemplo: `"sessions": ["session1"]` → 3 jobs (uno por super-sujeto).
Ejemplo: `"sessions": ["session1", "session2"]` → 6 jobs.

---

### 2.3 Bloque `tasks`

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| *(valor)* | `list[str]` | **Sí** | Lista de condiciones/tareas a incluir en CD-HSA. Todas las tareas se pasan juntas a una misma invocación de CD-HSA, que analiza la estructura cruzada entre condiciones. |

**Mínimo 2 tareas.** CD-HSA requiere al menos 2 condiciones para los tests estadísticos (Steps B/C/D). Con una sola tarea, los tests de condición no tienen sentido.

Ejemplos: `"tasks": ["eyesclosed", "music"]`, `"tasks": ["eyesclosed", "eyesopen", "music"]`.

---

### 2.4 Bloque `time_window`

Define la ventana temporal **por sujeto individual** antes de la concatenación en super-sujeto.

| Campo | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `t_start` | `float` | No | `null` (desde el inicio) | Tiempo de inicio en segundos. Se aplica a cada Raw individual antes de concatenar. |
| `t_end` | `float` | No | `null` (hasta el final) | Tiempo de fin en segundos. **Parámetro clave para controlar la memoria.** Con t_end más chico, menos columnas tiene la Hankel y menos memoria consume el block-Hankel de CD-HSA. |

**Efecto en memoria:** La cantidad de muestras por sujeto es `t_end × sfreq` (con sfreq=500 Hz y t_end=60, son 30,000 muestras). El super-sujeto las concatena, así que con 20 sujetos × 60s × 500Hz = 600,000 muestras totales. Reducir `t_end` es la forma más directa de controlar el uso de memoria.

---

### 2.5 Bloque `cdhsa_params`

Parámetros del algoritmo CD-HSA y del preprocesamiento. Todos son opcionales salvo `L`.

| Campo | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `L` | `int` | **Sí** | — | Dimensión del subespacio de truncamiento para CD-HSA (Step A). Determina el tamaño del block-Hankel interno: la dimensión de las filas pasa de `p·depth` a `p·depth·L`. Es el parámetro que más impacto tiene en el uso de memoria y en la resolución del análisis. |
| `hankel_depth` | `int \| null` | No | `null` (auto) | Profundidad del embedding de Hankel (cantidad de retardos temporales). Si es `null`, se calcula automáticamente con `_auto_embedding_depth(sfreq, n_times)`. Típico: 10. |
| `l_freq` | `float` | No | `1.0` | Frecuencia de corte inferior del filtro pasa-banda (Hz). |
| `h_freq` | `float` | No | `40.0` | Frecuencia de corte superior del filtro pasa-banda (Hz). |
| `fixed_rank` | `int` | No | `10` | Rango máximo para el Step A cuando `rank_method="fixed"`. Es el número de direcciones comunes estimadas. |
| `rank_method` | `str` | No | `"fixed"` | Método para determinar el rango del subespacio común. Valores: `"fixed"` (usa `fixed_rank` directamente), `"reproducibility"` (usa reproducibilidad cruzada con `rmax`, `n_blocks`, `repro_threshold`). |
| `a6_n_null` | `int` | No | `100` | Número de permutaciones nulas para el test de rango común (Step A6). Más permutaciones = p-valor más estable pero más lento. |
| `bc_n_perm` | `int` | No | `5000` | Número de permutaciones de labels para los tests de condición (Steps B/C). Más permutaciones = más preciso pero más lento. |
| `skip_bc` | `bool` | No | `false` | Si es `true`, saltea los Steps B/C (tests de energía y geometría por condición). |
| `skip_tangent` | `bool` | No | `false` | Si es `true`, saltea el test de geometría tangente. |
| `skip_d` | `bool` | No | `false` | Si es `true`, saltea el Step D (extracción de modos específicos por condición). |

**Nota sobre parámetros no expuestos:** La `CDHSAConfig` tiene muchos más campos (ver [Sección 4](#4-cdhsaconfig--parámetros-completos-api-python)) que no están disponibles ni en el JSON ni en la CLI. Para usarlos, es necesario invocar `run_cdhsa()` directamente desde Python.

---

### 2.6 Bloque `execution`

Controla cómo el batch runner orquesta la ejecución.

| Campo | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `max_workers` | `int` | No | `1` | Número de workers para ejecución paralela (via `ProcessPoolExecutor`). `1` = secuencial. > 1 = paralelo. Cuidado: cada worker ejecuta un subprocess que consume memoria. Con `max_workers=3` y 256 GB, cada job debe usar < 85 GB. |
| `delay` | `float` | No | `2.0` | Pausa en segundos entre jobs consecutivos (solo en modo secuencial). Permite que el sistema libere recursos. |
| `run_comparison` | `bool` | No | `true` | Si es `true`, ejecuta el post-procesamiento comparativo después de todos los jobs. Lee los `cdhsa_summary.txt` de cada job y genera un reporte consolidado. |

---

## 3. Parámetros de la CLI (`run_cdhsa.py`)

Estos son los flags que se pasan a `run_cdhsa.py` desde la terminal. El batch runner construye estos comandos automáticamente a partir del JSON.

### 3.1 Modo Multi-Super-Subject

```bash
python run_cdhsa.py --n-super-subjects N [otros params...]
```

| Flag | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `--n-super-subjects` | `int` | **Sí** *(grupo)* | — | Cantidad de super-sujetos. Mutuamente excluyente con `--super-subject-id`. Los `total_subjects` se reparten equitativamente. |
| `--total-subjects` | `int` | No | `60` | Total de sujetos disponibles. |
| `--subject-start-offset` | `int` | No | `1` | Índice del primer sujeto. |

En este modo, el pipeline ejecuta un **two-pass**: primero carga todos los super-sujetos para calcular la intersección global de canales (asegurando que p sea consistente), luego construye las Hankel con los canales comunes. El resultado `X` tiene forma `list[list[ndarray]]` con S elementos (uno por super-sujeto), cada uno con C matrices (una por tarea).

### 3.2 Modo Single-Super-Subject

```bash
python run_cdhsa.py --super-subject-id N [otros params...]
```

| Flag | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `--super-subject-id` | `int` | **Sí** *(grupo)* | — | ID del super-sujeto (1-indexed). Mutuamente excluyente con `--n-super-subjects`. |
| `--subjects-per-super-subject` | `int` | No | `20` | Sujetos por super-sujeto. No tiene efecto en modo multi-SS. |
| `--subject-start-offset` | `int` | No | `1` | Índice del primer sujeto. |

En este modo, no hay intersección global de canales — el super-sujeto usa sus propios canales (ya que S=1, no hay inconsistencia posible). La resolución de subject IDs se delega a `resolve_super_subject_subject_ids()`.

### 3.3 Parámetros Comunes

| Flag | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `--session` | `str` | **Sí** | — | Session ID (ej: `session1`). |
| `--tasks` | `str` (nargs `+`) | **Sí** | — | Lista de tareas/condiciones. Mínimo 2 para que CD-HSA tenga sentido. |
| `--db-path` | `str` | No | `None` (usa `src.utils.config`) | Ruta raíz del dataset Gedai. |
| `--t-start` | `float` | No | `None` (desde inicio) | Tiempo de inicio por sujeto (segundos). |
| `--t-end` | `float` | No | `None` (hasta final) | Tiempo de fin por sujeto (segundos). **Clave para memoria.** |
| `--out-dir` | `str` | No | `None` (usa `BASE_RESULTS_PATH`) | Directorio raíz para resultados. |
| `--l-freq` | `float` | No | `1.0` | Corte inferior del filtro (Hz). |
| `--h-freq` | `float` | No | `40.0` | Corte superior del filtro (Hz). |
| `--hankel-depth` | `int` | No | `None` (auto) | Profundidad Hankel. |
| `--verbose` | *flag* | No | `True` | Activar logging verboso. |
| `--no-save` | *flag* | No | `False` | No guardar resultados a disco (solo imprimir). |

### 3.4 Parámetros CD-HSA

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--L` | `int` | *(req)* | Dimensión del subespacio (requerido). |
| `--fixed-rank` | `int` | `10` | Rango máximo para Step A. |
| `--rank-method` | `str` | `"fixed"` | `"fixed"` o `"reproducibility"`. |
| `--a6-n-null` | `int` | `100` | Permutaciones nulas para A6. |
| `--bc-n-perm` | `int` | `5000` | Permutaciones para B/C. |
| `--skip-bc` | *flag* | `False` | Saltear Steps B/C. |
| `--skip-tangent` | *flag* | `False` | Saltear test tangente. |
| `--skip-d` | *flag* | `False` | Saltear Step D. |

---

## 4. CDHSAConfig — Parámetros Completos (API Python)

La dataclass `CDHSAConfig` (en `run_cdhsa.py` y en `src.cdhsa`) contiene **todos** los parámetros configurables del algoritmo CD-HSA. Solo un subconjunto está expuesto vía CLI y JSON. Para acceso completo, usar la API Python directamente.

### 4.1 Rank y Subespacio Común (Step A)

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `fixed_rank` | `int` | `10` | Número de direcciones comunes a estimar cuando `rank_method="fixed"`. Es la dimensión del subespacio común resultante del Step A. Valores típicos: 10–30. |
| `rank_method` | `str` | `"fixed"` | Estrategia para determinar el rango. `"fixed"` usa `fixed_rank` directamente. `"reproducibility"` determina el rango por estabilidad cruzada (ver parámetros abajo). |
| `rmax` | `int` | `20` | Rango máximo a explorar en el método de reproducibilidad. Solo usado cuando `rank_method="reproducibility"`. |
| `n_blocks` | `int` | `4` | Número de bloques temporales para la validación cruzada de reproducibilidad. |
| `repro_threshold` | `float` | `0.80` | Umbral de reproducibilidad (0–1). Direcciones con reproducibilidad ≥ este valor se consideran estables. |
| `repro_strategy` | `str` | `"consecutive"` | Estrategia de partición de bloques. `"consecutive"` divide el tiempo en bloques contiguos. |
| `max_common` | `int` | `30` | Límite superior para la cantidad de direcciones comunes. |

### 4.2 Test de Rango Común (Step A6)

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `a6_max_common` | `int` | `0` | Máximo de direcciones comunes a testear. `0` = automático: `min(20, len(lambda_))`. |
| `a6_n_folds` | `int` | `5` | Número de folds para la validación cruzada interna del test A6. |
| `a6_n_null` | `int` | `100` | Número de permutaciones nulas. Cada una genera una distribución nula del estadístico bajo H₀ (no hay direcciones comunes). Más permutaciones → p-valores más estables. |
| `a6_alpha` | `float` | `0.05` | Nivel de significación para el test A6. Se rechaza H₀ si el p-valor ≤ alpha. |
| `a6_seed` | `int` | `1234` | Semilla para el generador de permutaciones nulas. Asegura reproducibilidad. |

### 4.3 Test de Condición — Energía y Geometría (Steps B/C)

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `bc_blocks` | `list \| None` | `None` | Partición explícita de bloques temporales para los tests. Si es `None`, se usan los bloques del Step A. |
| `bc_energy_metric` | `str` | `"log_absolute"` | Métrica de energía. `"log_absolute"` usa el logaritmo de los valores absolutos de las componentes. |
| `bc_geometry_metric` | `str` | `"adjusted"` | Métrica de geometría. `"adjusted"` usa la métrica ajustada del paper de CD-HSA. |
| `bc_n_perm` | `int` | `5000` | Número de permutaciones de labels de condición. Más permutaciones → distribución nula más precisa → p-valores más estables. Costo computacional: O(bc_n_perm × S × C × d). |
| `bc_seed` | `int` | `20260812` | Semilla para las permutaciones de labels. |
| `bc_alpha` | `float` | `0.05` | Nivel de significación. Se marca con `*` las direcciones donde el estadístico excede el percentil (1-alpha) de la distribución nula. |
| `bc_condition_names` | `list[str]` | `[]` | Nombres de las condiciones para el reporte. Se llena automáticamente con los nombres de las tareas cuando se usa la CLI. |

### 4.4 Test de Geometría Tangente

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `tangent_blocks` | `list \| None` | `None` | Partición explícita de bloques. Si es `None`, se usan los del Step A. |
| `tangent_n_perm` | `int` | `5000` | Permutaciones para el test de geometría tangente. |
| `tangent_seed` | `int` | `9999` | Semilla para las permutaciones tangentes. |
| `tangent_alpha` | `float` | `0.05` | Nivel de significación. |

### 4.5 Modos Específicos por Condición (Step D)

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `d_max_specific` | `int` | `10` | Máximo de modos específicos por condición a extraer. |
| `d_residual_rank_method` | `str` | `"local_gap"` | Método para estimar el rango del espacio residual (la parte no común). `"local_gap"` busca un quiebre en los valores singulares. `"fixed"` usa `d_fixed_residual_rank`. |
| `d_residual_rank_threshold` | `float` | `0.1` | Umbral para el método `"local_gap"`. |
| `d_fixed_residual_rank` | `int` | `5` | Rango residual fijo cuando `d_residual_rank_method="fixed"`. |
| `prevalence_quantile` | `float` | `0.10` | Cuantil de prevalencia para determinar si un modo es específico de una condición. |

### 4.6 Flags de Salteo

| Parámetro | Tipo | Default | Efecto |
|---|---|---|---|
| `skip_bc` | `bool` | `False` | Saltea Steps B/C. Útil para explorar solo la estructura común (Step A + A6) sin testear diferencias por condición. |
| `skip_tangent` | `bool` | `False` | Saltea el test de geometría tangente. |
| `skip_d` | `bool` | `False` | Saltea Step D (modos específicos). Si solo te interesa el rango común y los tests de condición, pero no los modos específicos. |
| `seed` | `int` | `42` | Semilla general para el pipeline. |

---

## 5. Mapeo JSON → CLI → CDHSAConfig

La siguiente tabla muestra cómo cada parámetro fluye desde el JSON del batch runner hacia la CLI de `run_cdhsa.py` y finalmente hacia `CDHSAConfig`:

| JSON (`cdhsa_params`) | Flag CLI | Campo CDHSAConfig | ¿Expuesto? |
|---|---|---|---|
| `L` | `--L` | *(se pasa como argumento a `run_cdhsa()`)* | ✅ JSON + CLI |
| `hankel_depth` | `--hankel-depth` | *(se usa en `build_hankel_*`)* | ✅ JSON + CLI |
| `l_freq` | `--l-freq` | *(se usa en filtrado)* | ✅ JSON + CLI |
| `h_freq` | `--h-freq` | *(se usa en filtrado)* | ✅ JSON + CLI |
| `fixed_rank` | `--fixed-rank` | `fixed_rank` | ✅ JSON + CLI |
| `rank_method` | `--rank-method` | `rank_method` | ✅ JSON + CLI |
| `a6_n_null` | `--a6-n-null` | `a6_n_null` | ✅ JSON + CLI |
| `bc_n_perm` | `--bc-n-perm` | `bc_n_perm` | ✅ JSON + CLI |
| `skip_bc` | `--skip-bc` | `skip_bc` | ✅ JSON + CLI |
| `skip_tangent` | `--skip-tangent` | `skip_tangent` | ✅ JSON + CLI |
| `skip_d` | `--skip-d` | `skip_d` | ✅ JSON + CLI |
| — | — | `rmax` | ❌ Solo API |
| — | — | `n_blocks` | ❌ Solo API |
| — | — | `repro_threshold` | ❌ Solo API |
| — | — | `repro_strategy` | ❌ Solo API |
| — | — | `max_common` | ❌ Solo API |
| — | — | `prevalence_quantile` | ❌ Solo API |
| — | — | `a6_max_common` | ❌ Solo API |
| — | — | `a6_n_folds` | ❌ Solo API |
| — | — | `a6_alpha` | ❌ Solo API |
| — | — | `a6_seed` | ❌ Solo API |
| — | — | `bc_blocks` | ❌ Solo API |
| — | — | `bc_energy_metric` | ❌ Solo API |
| — | — | `bc_geometry_metric` | ❌ Solo API |
| — | — | `bc_seed` | ❌ Solo API |
| — | — | `bc_alpha` | ❌ Solo API |
| — | — | `tangent_blocks` | ❌ Solo API |
| — | — | `tangent_n_perm` | ❌ Solo API |
| — | — | `tangent_seed` | ❌ Solo API |
| — | — | `tangent_alpha` | ❌ Solo API |
| — | — | `d_max_specific` | ❌ Solo API |
| — | — | `d_residual_rank_method` | ❌ Solo API |
| — | — | `d_residual_rank_threshold` | ❌ Solo API |
| — | — | `d_fixed_residual_rank` | ❌ Solo API |
| — | — | `seed` | ❌ Solo API |

---

## 6. Cuándo Usar Cada Modo

### 6.1 Pipeline Directo — Multi-SS (`--n-super-subjects`)

**Usar cuando:**
- Querés analizar la estructura de subespacios **comunes entre** múltiples super-sujetos en una sola ejecución.
- Tenés suficiente memoria RAM para el block-Hankel de un super-sujeto completo (ver [Sección 10](#10-memoria--estimación-y-límites)).
- La pregunta es: "¿Existe un subespacio común compartido por estos N grupos de sujetos?"

**No usar cuando:**
- El dataset es grande y no cabe en memoria. Usar el batch runner en su lugar.

**Ejemplo:**

```bash
python run_cdhsa.py \
  --n-super-subjects 3 \
  --total-subjects 60 \
  --session session1 \
  --tasks eyesclosed music \
  --L 27 --fixed-rank 25 \
  --t-start 0.0 --t-end 30.0
```

### 6.2 Pipeline Directo — Single-SS (`--super-subject-id`)

**Usar cuando:**
- Querés analizar un solo super-sujeto (debugging, exploración rápida, validación).
- No necesitás comparación entre super-sujetos.

**Ejemplo:**

```bash
python run_cdhsa.py \
  --super-subject-id 2 \
  --session session1 \
  --tasks eyesclosed music \
  --L 27 --fixed-rank 25 \
  --subjects-per-super-subject 20 \
  --t-start 0.0 --t-end 60.0
```

### 6.3 Batch Runner (JSON)

**Usar cuando:**
- Querés correr CD-HSA en **cada super-sujeto por separado** y luego comparar resultados.
- Necesitás **checkpoint** (los jobs se retoman si se interrumpen).
- Necesitás **log estructurado** (CSV) de todas las ejecuciones.
- Querés el **post-procesamiento comparativo** automático.
- El dataset completo no cabe en memoria en modo multi-SS.

**No usar cuando:**
- Querés analizar la estructura común entre super-sujetos en una sola ejecución (usar `--n-super-subjects`).

**Ejemplo:**

```bash
python run_batch_cdhsa.py --params-json cdhsa_batch_params.json
```

---

## 7. Ejemplos de Configuración JSON

### 7.1 Configuración Estándar (3 SS de 20, 2 tareas)

```json
{
  "experiment_label": "per_ss_condition_modes_comparison",
  "super_subjects": {
    "selected": [1, 2, 3],
    "subjects_per_super_subject": 20,
    "subject_start_offset": 1,
    "total_subjects": 60
  },
  "sessions": ["session1"],
  "tasks": ["eyesclosed", "music"],
  "time_window": { "t_start": 0.0, "t_end": 60.0 },
  "cdhsa_params": {
    "L": 27, "hankel_depth": 10,
    "l_freq": 1.0, "h_freq": 40.0,
    "fixed_rank": 25, "rank_method": "reproducibility",
    "a6_n_null": 500, "bc_n_perm": 5000
  },
  "execution": { "max_workers": 1, "delay": 2.0, "run_comparison": true }
}
```

### 7.2 Cinco Super-sujetos (12 sujetos c/u, menos memoria)

```json
{
  "experiment_label": "5ss_12subs_each",
  "super_subjects": {
    "selected": [1, 2, 3, 4, 5],
    "subjects_per_super_subject": 12,
    "subject_start_offset": 1,
    "total_subjects": 60
  },
  "sessions": ["session1"],
  "tasks": ["eyesclosed", "music"],
  "time_window": { "t_start": 0.0, "t_end": 60.0 },
  "cdhsa_params": {
    "L": 27, "hankel_depth": 10,
    "l_freq": 1.0, "h_freq": 40.0,
    "fixed_rank": 25, "a6_n_null": 500, "bc_n_perm": 5000
  },
  "execution": { "max_workers": 1, "delay": 2.0, "run_comparison": true }
}
```

### 7.3 Dos Sesiones (cross-session)

```json
{
  "experiment_label": "cross_session_comparison",
  "super_subjects": {
    "selected": [1, 2, 3],
    "subjects_per_super_subject": 20,
    "subject_start_offset": 1,
    "total_subjects": 60
  },
  "sessions": ["session1", "session2"],
  "tasks": ["eyesclosed", "music"],
  "time_window": { "t_start": 0.0, "t_end": 60.0 },
  "cdhsa_params": {
    "L": 27, "hankel_depth": 10,
    "l_freq": 1.0, "h_freq": 40.0,
    "fixed_rank": 25, "a6_n_null": 500, "bc_n_perm": 5000
  },
  "execution": { "max_workers": 1, "delay": 2.0, "run_comparison": true }
}
```

Genera 6 jobs: (SS1×s1, SS1×s2, SS2×s1, SS2×s2, SS3×s1, SS3×s2).

### 7.4 Tres Tareas (eyesclosed, eyesopen, music)

```json
{
  "experiment_label": "3task_comparison",
  "super_subjects": {
    "selected": [1, 2, 3],
    "subjects_per_super_subject": 20,
    "subject_start_offset": 1,
    "total_subjects": 60
  },
  "sessions": ["session1"],
  "tasks": ["eyesclosed", "eyesopen", "music"],
  "time_window": { "t_start": 0.0, "t_end": 60.0 },
  "cdhsa_params": {
    "L": 27, "hankel_depth": 10,
    "l_freq": 1.0, "h_freq": 40.0,
    "fixed_rank": 25, "a6_n_null": 500, "bc_n_perm": 5000
  },
  "execution": { "max_workers": 1, "delay": 2.0, "run_comparison": true }
}
```

### 7.5 Solo Subespacio Común (saltear tests B/C/D)

Para una ejecución rápida explorando solo la estructura común (Steps A + A6):

```json
{
  "experiment_label": "common_subspace_only",
  "super_subjects": {
    "selected": [1],
    "subjects_per_super_subject": 20,
    "subject_start_offset": 1,
    "total_subjects": 60
  },
  "sessions": ["session1"],
  "tasks": ["eyesclosed", "music"],
  "time_window": { "t_start": 0.0, "t_end": 60.0 },
  "cdhsa_params": {
    "L": 27, "hankel_depth": 10,
    "l_freq": 1.0, "h_freq": 40.0,
    "fixed_rank": 25, "a6_n_null": 500, "bc_n_perm": 5000,
    "skip_bc": true, "skip_tangent": true, "skip_d": true
  },
  "execution": { "max_workers": 1, "delay": 2.0, "run_comparison": false }
}
```

### 7.6 Ejecución Paralela (2 workers)

```json
{
  "experiment_label": "parallel_2workers",
  "super_subjects": {
    "selected": [1, 2, 3],
    "subjects_per_super_subject": 20,
    "subject_start_offset": 1,
    "total_subjects": 60
  },
  "sessions": ["session1"],
  "tasks": ["eyesclosed", "music"],
  "time_window": { "t_start": 0.0, "t_end": 60.0 },
  "cdhsa_params": {
    "L": 27, "hankel_depth": 10,
    "l_freq": 1.0, "h_freq": 40.0,
    "fixed_rank": 25, "a6_n_null": 500, "bc_n_perm": 5000
  },
  "execution": { "max_workers": 2, "delay": 0.0, "run_comparison": true }
}
```

> **Advertencia:** Con 2 workers en paralelo, cada job consume memoria simultáneamente. Verificar que `memoria_por_job × max_workers < RAM_disponible`.

---

## 8. Estructura de Directorios de Salida

### 8.1 Pipeline Directo — Multi-SS

```
{BASE_RESULTS_PATH}/cdhsa/{session}/
    nSS{N}_L{L}_fr{fr}_a6n{a6}_bcn{bc}/
        {l_freq}-{h_freq}Hz_depth{d}/
            from{t_start}s_to{t_end}s_{tasks}/
                hankel_info.json
                config.json
                characterization.txt
                cdhsa_summary.txt
                hankel_matrices.npz
                cdhsa_arrays.npz
```

Ejemplo: `results/cdhsa/session1/nSS3_L27_fr25_a6n500_bcn5000/1.0-40.0Hz_depth10/from0.0s_to60.0s_eyesclosed_music/`

### 8.2 Pipeline Directo — Single-SS (y Batch Runner)

```
{BASE_RESULTS_PATH}/cdhsa/{session}/
    SS{ss_id}_L{L}_fr{fr}_a6n{a6}_bcn{bc}/
        {l_freq}-{h_freq}Hz_depth{d}/
            from{t_start}s_to{t_end}s_{tasks}/
                hankel_info.json
                config.json
                characterization.txt
                cdhsa_summary.txt
                hankel_matrices.npz
                cdhsa_arrays.npz
```

Ejemplo: `results/cdhsa/session1/SS1_L27_fr25_a6n500_bcn5000/1.0-40.0Hz_depth10/from0.0s_to60.0s_eyesclosed_music/`

La diferencia es `nSS{N}` vs `SS{id}` en el segundo nivel.

### 8.3 Archivos del Batch Runner

```
{BASE_RESULTS_PATH}/
    batch_logs/
        batch_cdhsa_{label}_{timestamp}.csv
    cdhsa/{session}/
        batch_comparison/
            comparison_{tasks_tag}.txt
    cache/
        batch_checkpoint_cdhsa.json
```

---

## 9. Archivos Generados por Job

Cada ejecución exitosa (ya sea vía CLI directa o batch) genera 6 archivos:

| Archivo | Formato | Contenido |
|---|---|---|
| `hankel_info.json` | JSON | Metadatos de la construcción de Hankel: formas de cada matriz, canales usados, profundidad, sfreq, sujetos por SS. |
| `config.json` | JSON | Configuración CDHSA completa usada en la ejecución (todos los campos de `CDHSAConfig` + `L`). |
| `characterization.txt` | Texto | Tabla de caracterización de las matrices de Hankel: forma, rank estimado, compresión, top-5 valores singulares para cada (SS, tarea). |
| `cdhsa_summary.txt` | Texto | Resumen textual del resultado CD-HSA: r₀ (rango común), valores λ, tests de condición con significancia, modos específicos por condición con prevalence contrast. |
| `hankel_matrices.npz` | NPZ | Matrices de Hankel `X[s][c]` con claves `H_ss{s}_c{c}`. Forma: `(p·depth, T-depth+1)`. |
| `cdhsa_arrays.npz` | NPZ | Arrays numéricos del resultado CD-HSA organizados por prefijo: `R__*` (Step A), `A6__*` (test rango), `BC__*` (tests condición), `G__*` (tangente), `D__*` (modos específicos). |

---

## 10. Memoria — Estimación y Límites

### 10.1 Cuello de Botella: Block-Hankel Interno

El paso que consume más memoria es el **Step A** de CD-HSA, que internamente construye un block-Hankel `build_block_hankel(Xi, L)`. Si la Hankel de entrada tiene forma `(d, K)`, el block-Hankel tiene forma `(d × L, K)`. Con los parámetros típicos del dataset Gedai:

- `p = 53` canales (después de intersección global)
- `depth = 10`
- `L = 27`
- `d = p × depth = 530`
- Block-Hankel: `(530 × 27, K) = (14,310, K)`
- Memoria por block-Hankel: `14,310 × K × 8 bytes`

### 10.2 Tabla de Memoria por t_end (1 SS de 20 sujetos)

| t_end (s) | Muestras por sujeto | Columnas K (aprox) | Memoria block-Hankel (1 condición) | Total estimado (2 condiciones + overhead) |
|---|---|---|---|---|
| 1 | 500 | ~10,000 | ~1.1 GB | ~3 GB |
| 10 | 5,000 | ~100,000 | ~11.5 GB | ~25 GB |
| 20 | 10,000 | ~200,000 | ~22.9 GB | ~50 GB |
| 30 | 15,000 | ~300,000 | ~34.3 GB | ~75 GB |
| 60 | 30,000 | ~600,000 | ~68.7 GB | ~150 GB |
| 120 | 60,000 | ~1,200,000 | ~137 GB | ~300 GB |

### 10.3 Regla Práctica

- **256 GB RAM, L=27, depth=10, 20 sujetos/SS:** `t_end` máximo seguro ≈ **30–40 s** por sujeto.
- **256 GB RAM, L=27, depth=10, 12 sujetos/SS (5 SS):** `t_end` máximo ≈ **50–60 s**.
- **256 GB RAM, L=27, depth=10, 1 sujeto/SS (60 SS):** Cualquier `t_end` (muy pocos datos por SS).

### 10.4 Multi-SS vs Batch

| Modo | Memoria pico | Trade-off |
|---|---|---|
| `--n-super-subjects 3` | Block-Hankel de 1 SS (CD-HSA los procesa secuencialmente internamente) + las Hankel de los 3 SS en RAM para la intersección de canales | Más memoria en el preprocesamiento (2-pass), pero una sola ejecución. |
| Batch runner (3 jobs × S=1) | Block-Hankel de 1 SS por job | Menos memoria por ejecución, pero 3 ejecuciones independientes. Resultados comparables pero no idénticos (cada SS se analiza individualmente). |

---
*AI生成*
