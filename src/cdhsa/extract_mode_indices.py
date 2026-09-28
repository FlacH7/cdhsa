"""
extract_mode_indices.py
=======================
Lee los archivos de salida de CD-HSA (hankel_info.json, config.json,
cdhsa_arrays.npz) y genera un JSON que mapea cada par
(super-sujeto, condicion) a los indices de los modos especificos
mas relevantes de esa condicion.

Uso::

    python -m src.cdhsa.extract_mode_indices \
        --results-dir results/cdhsa/session1/nSS5_.../.../eyesclosed_music
    python -m src.cdhsa.extract_mode_indices --results-dir <PATH> --top-n 2

El JSON de salida (mode_map.json) contiene, para cada condicion,
los indices de los modos especificos ordenados por lambda, y para
cada (super-sujeto, condicion) la clave de la Hankel asociada y la
formula de proyeccion.

Contexto matematico (v4 — corregido al metodo real)
----------------------------------------------------
Step D de CD-HSA opera sobre BASES ortonormales, no sobre
covarianzas de datos:

  1. Residuos: U_res(s,c) = columnas ortonormales de
     (I - W0 W0^T) U(s,c)  — la parte de la base local Hankel
     ortogonal al backbone comun (D1).
  2. Pooling por condicion: B_c = [U_res(1,c) ... U_res(S,c)]/√S
     y W(c), lambda(c) = SVD izquierda de B_c (D2). Los modos
     especificos W(c) viven en el espacio de nivel 2
     (dimension d = p * L, donde p = canales*depth del nivel 1
     y L = profundidad del nivel 2).
  3. Alineaciones propia/cruzada de cada residuo con W(c) (D3) y
     contraste de prevalencia Delta(c) (D4).

CRITICO — proyeccion correcta: W(c) vive en el espacio DOBLEMENTE
embebido (d = p*L filas), mientras que hankel_matrices.npz guarda
la Hankel de NIVEL 1 (p filas). La serie temporal del modo NO es
``W_sel.T @ H`` (solo tendria sentido con L=1); hay que reconstruir
el embedding de nivel 2 de H y proyectar sobre el::

    alpha(t) = W_sel^T @ x2(t),   x2(t) = [H[:, t+L-1]; ...; H[:, t]]

(ver ``project_level2``). La funcion ``project_level2`` lo hace por
bloques sin materializar la matriz de nivel 2 (que seria L veces mas
grande que H), y aplica la mascara de fronteras de concatenacion si
esta guardada en cdhsa_arrays.npz (columnas que cruzan fronteras
entre sujetos -> alpha = 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _safe_scalar(x) -> int | float | None:
    """Convertir un valor numpy (de cualquier dimensionalidad) a escalar Python.

    Maneja: None, 0-d arrays, 1D arrays de un elemento, y arrays
    multidimensionales (devuelve la lista conversion via tolist).
    """
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.flat[0].item()
    return arr.tolist()


def _json_safe(obj):
    """Convertir tipos numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ============================================================================
# Proyeccion correcta: nivel 1 -> nivel 2 -> alpha
# ============================================================================

def project_level2(
    H1: np.ndarray,
    W: np.ndarray,
    L: int,
    col_mask: np.ndarray | None = None,
    chunk: int = 4096,
) -> np.ndarray:
    """
    alpha = W^T @ build_block_hankel(H1, L) sin materializar la nivel 2.

    La columna t de la matriz de nivel 2 es::

        x2(t) = [H1[:, t+L-1]; H1[:, t+L-2]; ...; H1[:, t]]

    (muestra mas reciente primero — la MISMA convencion de
    ``a_common_subspace.build_block_hankel`` con la que se estimaron
    W0 y W_specific). Se procesa por bloques de ``chunk`` columnas,
    asi que el pico de memoria es O(p * L * chunk) en vez de
    O(p * L * K).

    Parameters
    ----------
    H1 : ndarray, shape (p, T)
        Hankel de nivel 1 (la guardada en hankel_matrices.npz).
    W : ndarray, shape (p*L, n_modes)
        Modos en el espacio de nivel 2 (columnas de W_specific).
    L : int
        Profundidad del embedding de nivel 2 (el ``L``/``L_bh`` de la
        configuracion de CD-HSA).
    col_mask : ndarray of bool, shape (T - L + 1,), optional
        Mascara de columnas de nivel 2 (ver ``level2_boundary_mask``).
        Las columnas False (ventanas que cruzan fronteras de
        concatenacion entre sujetos) devuelven alpha = 0.
    chunk : int
        Numero de columnas procesadas por bloque.

    Returns
    -------
    alpha : ndarray, shape (n_modes, T - L + 1)
        Series temporales de los modos. alpha[t] = 0 en columnas
        enmascaradas.
    """
    H1 = np.asarray(H1, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    p, T = H1.shape
    if T < L:
        raise ValueError(f"T={T} < L={L}: no se puede construir el nivel 2.")
    K = T - L + 1
    d = p * L
    if W.shape[0] != d:
        raise ValueError(
            f"W tiene {W.shape[0]} filas pero el embedding de nivel 2 de "
            f"H1 (p={p}, L={L}) tiene d={d}. W_specific vive en el espacio "
            f"DOBLEMENTE embebido; ver el docstring de este modulo."
        )

    if col_mask is not None:
        m = np.asarray(col_mask).astype(bool).ravel()
        if m.shape[0] != K:
            raise ValueError(
                f"col_mask tiene longitud {m.shape[0]} pero K={K}."
            )
    else:
        m = None

    n_modes = W.shape[1]
    alpha = np.zeros((n_modes, K), dtype=np.float64)

    for t0 in range(0, K, chunk):
        t1 = min(t0 + chunk, K)
        # Ventanas [t0, t1): el bloque de fila ell contiene las columnas
        # H1[:, t+L-1-ell] para t en [t0, t1) — un slice CONTIGUO.
        X2 = np.empty((d, t1 - t0), dtype=np.float64)
        for ell in range(L):
            X2[ell * p:(ell + 1) * p, :] = H1[:, t0 + L - 1 - ell:
                                              t1 + L - 1 - ell]
        alpha[:, t0:t1] = W.T @ X2

    if m is not None:
        alpha[:, ~m] = 0.0

    return alpha


# ============================================================================
# Extraccion de W(c) / lambda(c) por condicion
# ============================================================================

def _extract_per_condition(
    W: np.ndarray,
    lam: np.ndarray | None,
    r_specific: np.ndarray,
    C: int,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Extraer W(c) y lambda(c) por condicion desde los arrays planos.

    El array W_specific guardado por _save_result_arrays puede tener
    distintas formas dependiendo de como lo devuelva el pipeline:

    - 3D (C, p, r_max): stacked, cada condicion ocupa W[c, :, :r_c]
    - 2D (p, r_total): concatenado, se divide por sumas acumuladas de r_specific

    Parameters
    ----------
    W : ndarray
        Array de modos especificos (W_specific).
    lam : ndarray or None
        Array de eigenvalores (lambda_specific).
    r_specific : ndarray, shape (C,)
        Numero de modos por condicion.
    C : int
        Numero de condiciones.

    Returns
    -------
    list of (W_c, lam_c) tuples
        W_c has shape (p, r_c), lam_c has shape (r_c,) or None.
    """
    results = []

    if W.ndim == 3 and W.shape[0] == C:
        # --- Caso 3D: (C, p, r_max) ---
        for c in range(C):
            rc = int(r_specific[c])
            W_c = W[c, :, :rc]
            results.append((W_c, None))  # lam se procesa abajo

        if lam is not None:
            if lam.ndim == 2 and lam.shape[0] == C:
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[c, :rc])
            elif lam.ndim == 1:
                # Flat concatenado igual que W
                offset = 0
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[offset:offset + rc])
                    offset += rc

    elif W.ndim == 2:
        # --- Caso 2D: (p, r_total) concatenado ---
        offset = 0
        for c in range(C):
            rc = int(r_specific[c])
            W_c = W[:, offset:offset + rc]
            results.append((W_c, None))
            offset += rc

        if lam is not None:
            if lam.ndim == 1:
                offset = 0
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[offset:offset + rc])
                    offset += rc
            elif lam.ndim == 2 and lam.shape[0] == C:
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[c, :rc])

    else:
        raise ValueError(
            f'Shape de W_specific no reconocido: {W.shape}. '
            f'Se esperaba 2D (p, r_total) o 3D (C, p, r_max).'
        )

    return results


def build_mode_map(
    results_dir: str | Path,
    top_n: int = 2,
    L_override: int | None = None,
) -> dict:
    """Construir el mapeo de modos especificos desde resultados de CD-HSA.

    Parameters
    ----------
    results_dir : path
        Directorio que contiene hankel_info.json, config.json,
        y cdhsa_arrays.npz.
    top_n : int
        Cuantos modos especificos conservar por condicion (ordenados
        por eigenvalor descendente).
    L_override : int or None
        Profundidad de nivel 2 a usar en las formulas de proyeccion.
        Default: la del config.json guardado (clave 'L').

    Returns
    -------
    mode_map : dict
        Estructura JSON completa.
    """
    results_dir = Path(results_dir)

    # ------------------------------------------------------------------
    # 1. Cargar metadata
    # ------------------------------------------------------------------
    with open(results_dir / 'hankel_info.json') as f:
        hankel_info = json.load(f)

    with open(results_dir / 'config.json') as f:
        config = json.load(f)

    tasks = hankel_info.get('tasks', [])
    S = hankel_info.get('n_super_subjects',
                        hankel_info.get('S', 1))
    C = len(tasks)
    p_ch = hankel_info.get('n_channels_common',
                          hankel_info.get('n_global_channels', None))
    depth = hankel_info.get('depth_common',
                            hankel_info.get('hankel_depth_requested', None))
    p_hankel = (p_ch * depth) if (p_ch is not None and depth is not None) else None

    # L de nivel 2 (necesario para la proyeccion correcta)
    L2 = L_override if L_override is not None else config.get('L')
    if L2 is None:
        raise ValueError(
            'No se encontro la profundidad de nivel 2 (clave "L" del '
            'config.json). Use --L para indicarla.'
        )
    L2 = int(L2)

    # ------------------------------------------------------------------
    # 2. Cargar cdhsa_arrays.npz
    # ------------------------------------------------------------------
    npz_path = results_dir / 'cdhsa_arrays.npz'
    npz_data = np.load(npz_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # 3. Extraer arrays de Step D (keys flat D__*)
    # ------------------------------------------------------------------
    d_keys = [k for k in npz_data.files if k.startswith('D__')]

    # --- r_specific: (C,) numero de modos por condicion ---
    if 'D__r_specific' not in npz_data:
        raise ValueError(
            'No se encontro D__r_specific en cdhsa_arrays.npz. '
            'Step D puede no haberse ejecutado.\n'
            f'Keys D__ disponibles: {d_keys}'
        )
    r_specific = npz_data['D__r_specific']
    C_npz = len(r_specific)
    if C_npz != C:
        raise ValueError(
            f'Inconsistencia: hankel_info dice C={C} pero '
            f'D__r_specific tiene {C_npz} elementos'
        )

    # --- prevalence_contrast: (C,) ---
    prev_contrast = (npz_data['D__prevalence_contrast']
                     if 'D__prevalence_contrast' in npz_data else None)

    # --- prevalence_own: (C,) ---
    prev_own = (npz_data['D__prevalence_own']
                if 'D__prevalence_own' in npz_data else None)

    # --- W_specific: modos especificos (3D, 2D o ragged-indexed) ---
    lam_all = (npz_data['D__lambda_specific']
               if 'D__lambda_specific' in npz_data else None)

    ragged = 'D__W_specific__0' in npz_data and 'D__W_specific' not in npz_data

    if 'D__W_specific' in npz_data:
        W_all = npz_data['D__W_specific']
        per_condition = _extract_per_condition(W_all, lam_all, r_specific, C)
        w_npz_key = 'D__W_specific'
        w_shape_full = list(W_all.shape)
    elif 'D__W_specific__0' in npz_data:
        # Ragged (r_c distinto por condicion, p.ej. rank adaptativo):
        # _save_result_arrays guarda cada condicion como D__W_specific__{c}
        per_condition = []
        for c_idx in range(C):
            W_c = npz_data[f'D__W_specific__{c_idx}']
            lam_c = (npz_data[f'D__lambda_specific__{c_idx}']
                     if f'D__lambda_specific__{c_idx}' in npz_data
                     else None)
            per_condition.append((W_c, lam_c))
        w_npz_key = 'D__W_specific__{c}'
        w_shape_full = None
    else:
        raise ValueError(
            'No se encontro D__W_specific (ni D__W_specific__0) en '
            'cdhsa_arrays.npz. Step D puede no haberse ejecutado.\n'
            f'Keys D__ disponibles: {d_keys}'
        )

    # --- residual_rank: (S, C) o escalar ---
    residual_rank = (npz_data['D__residual_rank']
                     if 'D__residual_rank' in npz_data else None)

    # --- Mascara de fronteras por (s, c) si se guardo ---
    colmasks: list[np.ndarray | None] = [None] * (S * C)
    has_masks = False
    for s_idx in range(S):
        for c_idx in range(C):
            key = f'colmask_ss{s_idx + 1}_c{c_idx + 1}'
            if key in npz_data.files:
                colmasks[s_idx * C + c_idx] = npz_data[key]
                has_masks = True

    # ------------------------------------------------------------------
    # 5. Para cada condicion: ordenar por eigenvalor, seleccionar top-N
    # ------------------------------------------------------------------
    conditions_data = {}

    for c_idx in range(C):
        task_name = tasks[c_idx]
        W_c, lam_c = per_condition[c_idx]
        r_c = W_c.shape[1]

        # Ordenar por eigenvalor descendente
        if lam_c is not None and len(lam_c) == r_c:
            order = np.argsort(lam_c)[::-1]
        else:
            order = np.arange(r_c)

        n_actual = min(top_n, r_c)
        top_indices = [int(order[i]) for i in range(n_actual)]

        cond_info = {
            'task': task_name,
            'condition_index': c_idx,
            'total_specific_modes': int(r_c),
            'top_n_requested': top_n,
            'top_n_actual': n_actual,
            'mode_indices_in_W': top_indices,
            'W_npz_key': (w_npz_key.format(c=c_idx)
                          if '{c}' in w_npz_key else w_npz_key),
            'W_shape_full': (list(w_shape_full) if w_shape_full is not None
                             else list(W_c.shape)),
            'W_c_shape': list(W_c.shape),
            'eigenvalues_npz_key': ('D__lambda_specific'
                                   if lam_all is not None else None),
            'eigenvalues': _json_safe(lam_c) if lam_c is not None else None,
            'eigenvalues_sorted': (_json_safe(lam_c[order])
                                   if lam_c is not None else None),
            'prevalence_contrast': (float(prev_contrast[c_idx])
                                    if prev_contrast is not None else None),
            'prevalence_own': (float(prev_own[c_idx])
                               if prev_own is not None else None),
            'r_specific_from_pipeline': int(r_specific[c_idx]),
            'residual_rank': _safe_scalar(residual_rank),
        }

        # Info detallada de cada modo seleccionado
        cond_info['modes'] = []
        for rank_pos, mode_idx in enumerate(top_indices):
            mode_info = {
                'rank_position': rank_pos + 1,
                'column_index_in_W_c': mode_idx,
                'eigenvalue': (float(lam_c[mode_idx])
                               if lam_c is not None else None),
                'eigenvalue_rank': rank_pos + 1,
            }
            cond_info['modes'].append(mode_info)

        conditions_data[task_name] = cond_info

    # ------------------------------------------------------------------
    # 6. Para cada (super-sujeto, condicion): mapeo a Hankel + proyeccion
    # ------------------------------------------------------------------
    per_pair = []

    for s_idx in range(S):
        for c_idx in range(C):
            task_name = tasks[c_idx]
            hankel_key = f'H_ss{s_idx + 1}_c{c_idx + 1}'
            mask_key = f'colmask_ss{s_idx + 1}_c{c_idx + 1}'

            cond = conditions_data[task_name]
            mode_indices = cond['mode_indices_in_W']
            n_actual_pair = cond['top_n_actual']

            # Construir la formula de proyeccion concreta (v4: nivel 2)
            if ragged:
                w_sel_expr = (f"cdhsa[f'D__W_specific__{c_idx}']"
                              f"[:, {mode_indices}]")
            elif W_all.ndim == 3:
                w_sel_expr = (f"W_specific[{c_idx}, :, :"
                              f"{cond['total_specific_modes']}][:, "
                              f"{mode_indices}]")
            else:
                rc = cond['total_specific_modes']
                offset = int(np.sum(r_specific[:c_idx]))
                w_sel_expr = (f"W_specific[:, {offset}:{offset + rc}]"
                              f"[:, {mode_indices}]")

            mask_line = (
                f"mask = cdhsa.get('{mask_key}')  "
                f"# columnas de frontera (opcional)\n"
                if has_masks else ""
            )

            entry = {
                'super_subject': s_idx + 1,
                'super_subject_label': f'SS{s_idx + 1}',
                'task': task_name,
                'task_index': c_idx,
                'condition_index_in_W': c_idx if (not ragged and W_all.ndim == 3) else None,
                'hankel_npz_key': hankel_key,
                'hankel_npz_file': 'hankel_matrices.npz',
                'W_npz_key': w_npz_key.format(c=c_idx) if '{c}' in w_npz_key else w_npz_key,
                'W_npz_file': 'cdhsa_arrays.npz',
                'columns_to_extract': mode_indices,
                'n_output_dimensions': n_actual_pair,
                'level2_depth_L': L2,
                'colmask_npz_key': mask_key if has_masks else None,
                'boundary_masked_columns': (
                    int(np.sum(~colmasks[s_idx * C + c_idx].astype(bool)))
                    if colmasks[s_idx * C + c_idx] is not None else 0
                ),
                'projection_code': (
                    f"L = {L2}  # profundidad de nivel 2 (config['L'])\n"
                    f"W_sel = {w_sel_expr}  # (p*L, {n_actual_pair})\n"
                    f"H = hankel['{hankel_key}']   # (p, T) Hankel de NIVEL 1\n"
                    f"{mask_line}"
                    f"alpha = project_level2(H, W_sel, L"
                    f"{', col_mask=mask' if has_masks else ''})  "
                    f"# ({n_actual_pair}, T-L+1)"
                ),
            }
            per_pair.append(entry)

    # ------------------------------------------------------------------
    # 7. Info del espacio comun (A6)
    # ------------------------------------------------------------------
    common_info = {}
    for k in npz_data.files:
        if k.startswith('A6__'):
            short_name = k.replace('A6__', '')
            common_info[short_name] = _json_safe(npz_data[k])

    # ------------------------------------------------------------------
    # 8. Ensamblar JSON final
    # ------------------------------------------------------------------
    mode_map = {
        'metadata': {
            'source_dir': str(results_dir),
            'n_super_subjects': S,
            'n_conditions': C,
            'tasks': tasks,
            'p_hankel': p_hankel,
            'n_channels': p_ch,
            'hankel_depth': depth,
            'level2_depth_L': L2,
            'projection_space': (
                'level2 (d = p*L); W_specific vive en el espacio '
                'doblemente embebido — usar project_level2, NO '
                'W_sel.T @ H'
            ),
            'top_n_requested': top_n,
            'pipeline_config': {
                'fixed_rank': config.get('fixed_rank'),
                'L': config.get('L'),
                'rank_method': config.get('rank_method'),
                'd_max_specific': config.get('d_max_specific', 10),
                'a6_n_null': config.get('a6_n_null'),
                'bc_n_perm': config.get('bc_n_perm'),
            },
        },
        'common_subspace': common_info,
        'conditions': conditions_data,
        'per_super_subject_task': per_pair,
        'npz_structure': {
            'W_specific_shape': (list(w_shape_full) if w_shape_full is not None
                                 else [f'D__W_specific__{c}' for c in range(C)]),
            'W_specific_ragged': ragged,
            'lambda_specific_shape': (list(lam_all.shape)
                                      if lam_all is not None else None),
            'r_specific': _json_safe(r_specific),
            'boundary_masks_saved': has_masks,
            'all_D_keys': d_keys,
        },
        'usage': {
            'description': (
                'Para obtener las series temporales de dimension top_n del '
                'par (super_subject, task): reconstruir el embedding de '
                'nivel 2 de la Hankel de nivel 1 y proyectar sobre los '
                'modos seleccionados. La columna enmascarada (fronteras '
                'entre sujetos) da alpha = 0.'
            ),
            'python_example': (
                'import numpy as np\n'
                'from src.cdhsa.extract_mode_indices import project_level2\n'
                '\n'
                'hankel = np.load("hankel_matrices.npz")\n'
                'cdhsa = np.load("cdhsa_arrays.npz")\n'
                '\n'
                'L = 10  # level2_depth_L de este JSON (config["L"])\n'
                'W_sel = cdhsa["D__W_specific"][:, [0, 1]]  # (p*L, 2)\n'
                'H = hankel["H_ss1_c1"]  # (p, T) Hankel de NIVEL 1\n'
                'mask = cdhsa.get("colmask_ss1_c1")  # opcional (fronteras)\n'
                '\n'
                'alpha = project_level2(H, W_sel, L, col_mask=mask)\n'
                '# alpha: (2, T-L+1); alpha[0] = serie del modo top-1\n'
                '# NO usar "W_sel.T @ H": W_sel vive en el espacio p*L,\n'
                '# H en el espacio p (solo coincide si L=1).'
            ),
        },
    }

    return mode_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Extraer indices de modos especificos de CD-HSA y generar '
            'mode_map.json'
        ),
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directorio con los archivos de salida de CD-HSA',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=2,
        help=(
            'Cuantos modos especificos conservar por condicion '
            '(default: 2)'
        ),
    )
    parser.add_argument(
        '-L', '--level2-depth',
        type=int,
        default=None,
        dest='L',
        help=(
            'Profundidad del embedding de nivel 2 para la proyeccion. '
            'Default: la clave "L" del config.json de results-dir.'
        ),
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=(
            'Ruta del JSON de salida. Default: mode_map.json '
            'en el mismo results-dir'
        ),
    )

    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f'[ERROR] Directorio no encontrado: {results_dir}',
              file=sys.stderr)
        return 1

    # Verificar archivos necesarios
    required = ['hankel_info.json', 'config.json', 'cdhsa_arrays.npz']
    for fname in required:
        if not (results_dir / fname).exists():
            print(f'[ERROR] Archivo requerido no encontrado: {fname}',
                  file=sys.stderr)
            return 1

    print(f'Leyendo resultados de: {results_dir}')
    print(f'Top-N modos por condicion: {args.top_n}')
    print()

    mode_map = build_mode_map(results_dir, top_n=args.top_n,
                              L_override=args.L)

    # Guardar
    out_path = (Path(args.output) if args.output
                else results_dir / 'mode_map.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mode_map, f, indent=2, ensure_ascii=False, default=str)

    print(f'[OK] mode_map.json guardado en: {out_path}')
    print()

    # Resumen
    meta = mode_map['metadata']
    print(f'  Super-sujetos: {meta["n_super_subjects"]}')
    print(f'  Condiciones:   {meta["n_conditions"]} ({meta["tasks"]})')
    print(f'  p (Hankel):    {meta["p_hankel"]}')
    print(f'  L (nivel 2):   {meta["level2_depth_L"]}')
    print()

    print('  Modos especificos por condicion:')
    for task_name, cond in mode_map['conditions'].items():
        r_c = cond['total_specific_modes']
        n_actual = cond['top_n_actual']
        indices = cond['mode_indices_in_W']
        pc = cond['prevalence_contrast']
        pc_str = f'{pc:.6f}' if pc is not None else 'N/A'
        print(f'    {task_name}: {r_c} modos, '
              f'top-{n_actual} = cols {indices}, '
              f'prev_contrast={pc_str}')
        for m in cond['modes']:
            ev = m['eigenvalue']
            ev_str = f'ev={ev:.6f}' if ev is not None else 'ev=N/A'
            print(f'      rank {m["rank_position"]}: col {m["column_index_in_W_c"]} ({ev_str})')

    print()
    print('  Mapeo (SS, task) -> Hankel key -> cols:')
    for entry in mode_map['per_super_subject_task']:
        bm = entry['boundary_masked_columns']
        bm_str = f' [frontera: {bm} cols]' if bm else ''
        print(f'    SS{entry["super_subject"]}/{entry["task"]}: '
              f'{entry["hankel_npz_key"]} -> {entry["columns_to_extract"]}'
              f'{bm_str}')

    print()
    npz_struct = mode_map['npz_structure']
    if not npz_struct['W_specific_ragged']:
        print(f'  W_specific shape: {npz_struct["W_specific_shape"]}')
    else:
        print(f'  W_specific: ragged ({npz_struct["W_specific_shape"]})')
    if npz_struct['lambda_specific_shape']:
        print(f'  lambda_specific shape: {npz_struct["lambda_specific_shape"]}')
    print(f'  r_specific: {npz_struct["r_specific"]}')
    print(f'  boundary_masks_saved: {npz_struct["boundary_masks_saved"]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
