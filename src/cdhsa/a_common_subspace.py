"""
cdhsa/a_common_subspace.py — Steps A1-A5 of CD-HSA (memory-optimized: LinearOperator, H never materialized)
==========================================================================

Estima el subespacio Hankel común poblacional a partir de grabaciones
EEG multicanal de múltiples sujetos y condiciones.

**Memory Optimization**: This version uses `scipy.sparse.linalg.LinearOperator`
instead of materializing the block-Hankel matrix H of shape (p*L, K), which can
exceed 12 GB in realistic EEG datasets. The LinearOperator computes matrix-vector
products on-the-fly, so `svds` never needs the full matrix in memory.

NOTA CRÍTICA DE DISEÑO
------------------------
Este módulo traduce fielmente el algoritmo MATLAB cdhsa_A1_A5.m del tutor,
con las siguientes mejoras y correcciones identificadas:

1. **Normalización geométrica (A1)**: El MATLAB normaliza H por ||H||_F.
   Esto NO cambia U_sc (la base izquierda de la SVD es invariante a escalado
   escalar). Lo conservamos por compatibilidad con B/C (que sí necesita
   la norma original para calcular energía), pero documentamos que es
   irrelevante para A1-A5.

2. **Selección de rank por reproducibilidad (A2)**: El MATLAB usa un
   criterio "consecutivo desde r=1" — si R(3) cae por debajo del umbral,
   se detiene en r=2 aun cuando R(4) y R(5) puedan ser altos. Esto puede
   subestimar el rank en presencia de un componente débil temprano.
   Implementamos el mismo criterio por fidelidad, pero añadimos el
   parámetro ``gap_based=True`` que selecciona por gaps en R(r).

3. **SVD de B para M₀ (A3)**: La concatenación B = [U₁...U_N]/√N y su
   SVD izquierda es equivalente a eigendecompose M₀ = (1/N)Σ U_i U_i^T.
   El MATLAB lo hace bien. Aquí usamos np.linalg.svd en vez de svds porque
   B típicamente tiene dimensiones (pL, Σr_sc) donde Σr_sc << pL.
   NOTA (v4): λ_j = σ_j(B) — VALOR SINGULAR, no su cuadrado — para ser
   homogéneo con Step D (lambda_specific) y con la Def. 3.13 del paper.
   El cambio es monótono, así que el criterio (i) de A6 no se altera;
   solo cambian los valores REPORTADOS de λ.

4. **Verificación de consistencia de canales**: Añadimos chequeo explícito
   de que todos los X[s][c] tengan el mismo número de canales p.

5. **Tipado y documentación**: Docstrings completos con tipos, shape
   annotations, y referencias a las ecuaciones del marco teórico.

6. **Optimización de memoria (v2)**: En lugar de materializar la matriz
   block-Hankel H de shape (p*L, K) — que puede exceder 12 GB — se usa
   `scipy.sparse.linalg.LinearOperator` para que `svds` nunca necesite
   la matriz completa. Se añaden ``make_block_hankel_linop``,
   ``_make_scaled_linop``, y ``compute_block_hankel_fro``.

7. **Máscara de fronteras de concatenación (v4)**: cuando un
   super-sujeto es la concatenación temporal de varios sujetos, las
   columnas de nivel 2 cuya ventana completa de muestras
   [t, t+L+depth-2] cruza una frontera interna mezclan dinámicas de
   individuos distintos. El paper promete descartarlas (a lo sumo
   L+depth-2 columnas por frontera). ``level2_boundary_mask`` computa
   la máscara y ``make_block_hankel_linop``/``compute_block_hankel_fro``
   la aplican (columnas enmascaradas = columnas cero; equivalente
   matemático a eliminarlas) sin materializar nada extra.

Dependencias: numpy, scipy.sparse.linalg (solo para SVD truncada grande).
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import svds


# ============================================================================
# 0. Block Hankel construction
# ============================================================================

def build_block_hankel(X: NDArray[np.floating], L: int) -> NDArray[np.floating]:
    """
    Construye la matriz de Hankel multicanal por bloques.

    Para X de shape (p, T), la columna t-ésima de H es::

        H[:, t] = [x(t), x(t-1), ..., x(t-L+1)]^T

    Es decir, muestra actual primero, luego muestras previas.
    El resultado tiene shape (p*L, T-L+1).

    Parameters
    ----------
    X : array, shape (p, T)
        Datos EEG canales × tiempo.
    L : int
        Número de retardos (embedding depth).

    Returns
    -------
    H : array, shape (p*L, K)  donde K = T - L + 1

    Raises
    ------
    ValueError
        Si T < L.

    CRÍTICA vs MATLAB
    ----------------
    El MATLAB (build_block_hankel) usa la misma convolución:
        H(rows,:) = X(:, L-ell+1:T-ell+1)
    con ell=1..L. Esto produce x(t) primero. Nuestra implementación
    es equivalente.

    NOTA sobre convención del proyecto existente:
    El proyecto usa ``_build_multivariate_hankel`` de hankel_dmd_extractor.
    Si esa función usa una convención diferente (ej. x(t-L+1) primero),
    los subespacios U serán los mismos (las columnas solo se reordenan
    dentro del bloque), pero los patrones espaciales-temporales reshaped
    W_j(e,τ) tendrán el eje τ invertido. Verificar antes de interpretar.

    NOTA (v2): Esta función se conserva para compatibilidad hacia atrás
    y pruebas unitarias. El camino optimizado en memoria usa
    ``make_block_hankel_linop`` en su lugar.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X debe ser 2-D (p × T), got shape {X.shape}")
    p, T = X.shape
    if T < L:
        raise ValueError(
            f"T ({T}) debe ser >= L ({L}) para construir Hankel"
        )
    K = T - L + 1
    H = np.empty((p * L, K), dtype=np.float64)
    for ell in range(L):
        # ell=0 → muestras más recientes (x(t)     hasta x(t-L+1)  en la primera columna)
        # ell=L-1 → muestras más antiguas  (x(t-L+1) hasta x(t-2L+2) en la primera columna)
        H[ell * p : (ell + 1) * p, :] = X[:, L - 1 - ell : T - ell]
    return H


# ============================================================================
# 0c. Boundary masking for concatenated super-subjects (v4)
# ============================================================================

def level2_boundary_mask(
    member_n_times: list[int] | NDArray[np.integer],
    depth: int,
    L: int,
) -> NDArray[np.bool_]:
    """
    Máscara de columnas de nivel 2 libres de mezcla entre sujetos.

    Un super-sujeto es la concatenación temporal de varios sujetos.
    X[s][c] es su Hankel de nivel 1 (profundidad ``depth``), de modo que
    la columna t de la matriz de nivel 2 (construida con
    ``build_block_hankel(X, L)``) cubre las muestras
    [t, t + L + depth - 2] del flujo concatenado. Esa columna mezcla
    dinámicas de dos sujetos distintos si y solo si su ventana cruza
    una frontera interna de concatenación.

    El paper (Sección de construcción de datos) promete descartar esas
    columnas: a lo sumo ``L + depth - 2`` columnas por frontera interna.
    Esta función devuelve la máscara booleana (True = columna limpia) de
    longitud ``K2 = T_total - (depth-1) - (L-1)``.

    Parameters
    ----------
    member_n_times : list of int
        Número de muestras (ya recortadas y filtradas) de CADA sujeto
        miembro, en orden de concatenación. Un solo miembro => sin
        fronteras internas => máscara toda True.
    depth : int
        Profundidad del embedding de nivel 1 (filas de X = canales*depth).
    L : int
        Profundidad del embedding de nivel 2.

    Returns
    -------
    mask : ndarray of bool, shape (K2,)
        True para las columnas cuya ventana completa de muestras no
        cruza ninguna frontera interna.
    """
    member_n_times = np.asarray(member_n_times, dtype=np.int64).ravel()
    if member_n_times.size == 0:
        raise ValueError("member_n_times está vacío.")
    if np.any(member_n_times <= 0):
        raise ValueError("member_n_times debe ser > 0 para cada miembro.")
    if depth < 1 or L < 1:
        raise ValueError(f"depth y L deben ser >= 1, got depth={depth}, L={L}")

    T_total = int(member_n_times.sum())
    K2 = T_total - (depth - 1) - (L - 1)
    if K2 < 1:
        raise ValueError(
            f"Grabación demasiado corta para el embedding de dos niveles: "
            f"T={T_total}, depth={depth}, L={L} => K2={K2} < 1."
        )

    # Fronteras internas: primera muestra de cada miembro excepto el primero.
    boundaries = np.cumsum(member_n_times)[:-1]

    mask = np.ones(K2, dtype=bool)
    t = np.arange(K2)
    # La columna t cruza la frontera b si t < b <= t + (L + depth - 2).
    for b in boundaries:
        crossing = (t < b) & (t + L + depth - 2 >= b)
        mask &= ~crossing
    return mask


def _bad_level2_columns(
    col_mask: NDArray[np.bool_] | None,
    K: int,
) -> NDArray[np.integer] | None:
    """Índices de columnas enmascaradas, o None si no hay enmascaramiento.

    Valida la longitud de la máscara contra ``K`` (número de columnas de
    la matriz de nivel 2). Devuelve un array vacío si la máscara es toda
    True (equivalente a no enmascarar; los caminos rápidos lo detectan
    con ``bad is None or bad.size == 0``).
    """
    if col_mask is None:
        return None
    m = np.asarray(col_mask).astype(bool).ravel()
    if m.shape[0] != K:
        raise ValueError(
            f"col_mask tiene longitud {m.shape[0]} pero la matriz de "
            f"nivel 2 tiene {K} columnas. ¿La máscara se calculó con el "
            f"mismo depth/L de esta grabación?"
        )
    bad = np.flatnonzero(~m)
    return bad if bad.size > 0 else None


# ============================================================================
# 0b. Memory-efficient block-Hankel helpers (v2)
# ============================================================================

def make_block_hankel_linop(
    X: NDArray[np.floating],
    L: int,
    col_mask: NDArray[np.bool_] | None = None,
) -> ScipyLinearOperator:
    """
    Create a LinearOperator that computes build_block_hankel(X, L) @ v
    and build_block_hankel(X, L).T @ w WITHOUT materializing the matrix.

    The block-Hankel H of shape (p*L, K) where K = T - L + 1 has::

        H[ell*p:(ell+1)*p, :] = X[:, L-1-ell : T-ell]  for ell = 0..L-1

    So H @ v is computed by stacking L matrix-vector products of (p, K) slices.
    And H.T @ w is computed by summing L transposed products.

    Parameters
    ----------
    X : array, shape (p, T)
        Datos EEG canales × tiempo.
    L : int
        Número de retardos (embedding depth).
    col_mask : ndarray of bool, shape (K,), optional
        Máscara de columnas de nivel 2 (ver ``level2_boundary_mask``).
        Las columnas con False se tratan como CEROS: la acción del
        operador es exactamente la de ``build_block_hankel(X, L)`` con
        esas columnas eliminadas (las columnas cero no afectan a la SVD
        izquierda, a la norma Frobenius ni a H^T @ W). Solo se
        materializan las pocas columnas enmascaradas (d × n_bad), así
        que el coste extra es despreciable.

    Returns
    -------
    linop : ScipyLinearOperator, shape (p*L, K)
        LinearOperator compatible con ``scipy.sparse.linalg.svds``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X debe ser 2-D (p × T), got shape {X.shape}")
    p, T = X.shape
    if T < L:
        raise ValueError(
            f"T ({T}) debe ser >= L ({L}) para construir Hankel"
        )
    K = T - L + 1
    d = p * L

    bad = _bad_level2_columns(col_mask, K)

    # Columnas enmascaradas de H (d, n_bad), materializadas UNA sola vez.
    Hbad: NDArray[np.floating] | None = None
    if bad is not None:
        Hbad = np.empty((d, bad.size), dtype=np.float64)
        for ell in range(L):
            # Columna t de H: bloque de fila ell = X[:, t + L - 1 - ell].
            Hbad[ell * p : (ell + 1) * p, :] = X[:, bad + (L - 1 - ell)]

    def matvec_full(v):
        v = np.asarray(v).ravel()
        out = np.zeros(d, dtype=np.float64)
        for ell in range(L):
            block = X[:, L - 1 - ell : T - ell]
            out[ell * p : (ell + 1) * p] = block @ v
        return out

    def rmatvec_full(w):
        w = np.asarray(w).ravel()
        out = np.zeros(K, dtype=np.float64)
        for ell in range(L):
            block = X[:, L - 1 - ell : T - ell]
            out += block.T @ w[ell * p : (ell + 1) * p]
        return out

    def _matmat_full(V):
        """H @ V for V of shape (K, nvec) -> (d, nvec)."""
        V = np.asarray(V)
        if V.ndim == 1:
            return matvec_full(V)
        nvec = V.shape[1]
        out = np.zeros((d, nvec), dtype=np.float64)
        for ell in range(L):
            block = X[:, L - 1 - ell : T - ell]  # (p, K)
            out[ell * p : (ell + 1) * p, :] = block @ V  # (p, nvec)
        return out

    def _rmatmat_full(W):
        """H.T @ W for W of shape (d, nvec) -> (K, nvec)."""
        W = np.asarray(W)
        if W.ndim == 1:
            return rmatvec_full(W)
        nvec = W.shape[1]
        out = np.zeros((K, nvec), dtype=np.float64)
        for ell in range(L):
            block = X[:, L - 1 - ell : T - ell]  # (p, K)
            out += block.T @ W[ell * p : (ell + 1) * p, :]  # (K, nvec)
        return out

    if Hbad is None:
        matvec, rmatvec, _matmat, _rmatmat = (
            matvec_full, rmatvec_full, _matmat_full, _rmatmat_full,
        )
    else:
        bad_idx = bad  # cierre explícito

        def matvec(v):
            # H_masked @ v = H @ v - H[:, bad] @ v[bad]
            out = matvec_full(v)
            out -= Hbad @ (np.asarray(v).ravel()[bad_idx])
            return out

        def rmatvec(w):
            # H_masked.T @ w = mask * (H.T @ w)
            out = rmatvec_full(w)
            out[bad_idx] = 0.0
            return out

        def _matmat(V):
            V = np.asarray(V)
            if V.ndim == 1:
                return matvec(V)
            out = _matmat_full(V)
            out -= Hbad @ V[bad_idx, :]
            return out

        def _rmatmat(W):
            W = np.asarray(W)
            if W.ndim == 1:
                return rmatvec(W)
            out = _rmatmat_full(W)
            out[bad_idx, :] = 0.0
            return out

    linop = ScipyLinearOperator((d, K), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
    linop._matmat = _matmat
    linop._rmatmat = _rmatmat
    return linop


def _make_scaled_linop(linop: ScipyLinearOperator, scale: float) -> ScipyLinearOperator:
    """Return a new LinearOperator that computes linop @ v / scale.

    This is the memory-efficient equivalent of H / hnorm where H is the
    block-Hankel matrix and hnorm is its Frobenius norm.

    Parameters
    ----------
    linop : ScipyLinearOperator
        LinearOperator representando la matriz a escalar.
    scale : float
        Escalar divisor (debe ser > 0).

    Returns
    -------
    scaled : ScipyLinearOperator
        LinearOperator que computa linop @ v / scale.
    """
    def matvec(v):
        return linop.matvec(v) / scale

    def rmatvec(w):
        return linop.rmatvec(w) / scale

    def _matmat(V):
        return linop.matmat(V) / scale

    def _rmatmat(W):
        return linop.rmatmat(W) / scale

    scaled = ScipyLinearOperator(linop.shape, matvec=matvec, rmatvec=rmatvec, dtype=linop.dtype)
    scaled._matmat = _matmat
    scaled._rmatmat = _rmatmat
    return scaled


def compute_block_hankel_fro(
    X: NDArray[np.floating],
    L: int,
    col_mask: NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute ||build_block_hankel(X, L)||_F without materializing the matrix.

    Uses: ||H||_F^2 = sum_ell ||X[:, L-1-ell : T-ell]||_F^2
    Each block is a VIEW into X, so peak memory = O(p * K) for the dot product.
    Uses np.dot(block.ravel(), block.ravel()) to avoid creating block**2 temporary.

    With ``col_mask`` (ver ``level2_boundary_mask``), the norm is that of
    the matrix with the masked columns REMOVED: se resta la energía de
    las pocas columnas enmascaradas (coste O((n_bad + L) * p)).

    Parameters
    ----------
    X : array, shape (p, T)
        Datos EEG canales × tiempo.
    L : int
        Número de retardos (embedding depth).
    col_mask : ndarray of bool, shape (T - L + 1,), optional
        Máscara de columnas de nivel 2.

    Returns
    -------
    hnorm : float
        Norma Frobenius de la matriz block-Hankel (con columnas
        enmascaradas eliminadas si se provee ``col_mask``).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X debe ser 2-D (p × T), got shape {X.shape}")
    p, T = X.shape
    if T < L:
        raise ValueError(
            f"T ({T}) debe ser >= L ({L}) para construir Hankel"
        )
    fro_sq = 0.0
    for ell in range(L):
        block = X[:, L - 1 - ell : T - ell]
        fro_sq += np.dot(block.ravel(), block.ravel())

    bad = _bad_level2_columns(col_mask, T - L + 1)
    if bad is not None:
        # ||H[:, t]||^2 = sum_ell ||X[:, t + L - 1 - ell]||^2
        # (suma de normas de L columnas individuales de X).
        # Solo se tocan las columnas de X implicadas (rango contiguo pequeño).
        lo = max(int(bad.min()) - 1, 0)
        hi = min(int(bad.max()) + L, T)
        seg = X[:, lo:hi]
        cn2 = np.einsum("ij,ij->j", seg, seg)  # (hi-lo,)
        idx = (bad[:, None] + (L - 1) - np.arange(L)[None, :]) - lo
        fro_sq -= float(cn2[idx].sum())

    return np.sqrt(max(fro_sq, 0.0))


# ============================================================================
# 1. Truncated SVD helpers
# ============================================================================

def truncated_left_svd(
    H: Union[NDArray[np.floating], ScipyLinearOperator],
    r: int,
) -> NDArray[np.floating]:
    """
    Primeros ``r`` vectores singulares izquierdos de H.

    Usa ``scipy.sparse.linalg.svds`` cuando r < min(m,n)-1 para eficiencia,
    con fallback a SVD densa.

    Cuando H es un ``ScipyLinearOperator``, solo se puede usar ``svds``
    (no hay acceso a la matriz densa para el fallback).

    Parameters
    ----------
    H : array, shape (m, n)  o  ScipyLinearOperator, shape (m, n)
        Matriz o LinearOperator.
    r : int
        Número de componentes a retener.

    Returns
    -------
    U : array, shape (m, r)
        Columnas ortonormales.

    CRÍTICA vs MATLAB
    ----------------
    ``scipy.sparse.linalg.svds`` devuelve valores singulares en orden
    **ascendente**, a diferencia del MATLAB ``svds(...,'largest')`` que
    devuelve descendente. Ordenamos explícitamente.

    Además, ``svds`` puede fallar con ARNOLDI para matrices con valores
    singulares degenerados (común en oscilaciones Hankel). El fallback
    a SVD densa cubre ese caso (solo disponible si H es densa).

    NOTA (v2): Cuando H es un ScipyLinearOperator, no se puede hacer
    fallback a SVD densa. Si ``svds`` falla, se propaga la excepción.
    """
    m, n = H.shape
    r = min(r, m, n)
    if r < 1:
        raise ValueError(f"r debe ser >= 1, got {r}")

    if isinstance(H, ScipyLinearOperator):
        # LinearOperator: can only use svds, no dense fallback
        if r >= min(m, n):
            raise ValueError(
                f"r={r} too large for LinearOperator with shape {H.shape}. "
                f"Need r < min({m}, {n}) = {min(m, n)}"
            )
        U, s_vals, _ = svds(H, k=r)
        # scipy svds → orden ascendente, corregir
        idx = np.argsort(s_vals)[::-1]
        return U[:, idx]

    # Original dense path (unchanged)
    # Usar svds solo cuando es seguro y ventajoso
    if r < min(m, n) - 1:
        try:
            U, s_vals, _ = svds(H, k=r)
            # scipy svds → orden ascendente, corregir
            idx = np.argsort(s_vals)[::-1]
            return U[:, idx]
        except Exception:
            pass  # fallback

    # SVD densa
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    return U[:, :r]


def truncated_left_svd_with_values(
    H: Union[NDArray[np.floating], ScipyLinearOperator],
    r: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Primeros ``r`` vectores singulares izquierdos y sus valores.

    Cuando H es un ``ScipyLinearOperator``, solo se puede usar ``svds``
    (no hay acceso a la matriz densa para el fallback).

    Parameters
    ----------
    H : array, shape (m, n)  o  ScipyLinearOperator, shape (m, n)
        Matriz o LinearOperator.
    r : int
        Número de componentes a retener.

    Returns
    -------
    U : array, shape (m, r)
    s : array, shape (r,)
        Valores singulares en orden descendente.

    NOTA (v2): Cuando H es un ScipyLinearOperator, no se puede hacer
    fallback a SVD densa.
    """
    m, n = H.shape
    r = min(r, m, n)
    if r < 1:
        raise ValueError(f"r debe ser >= 1, got {r}")

    if isinstance(H, ScipyLinearOperator):
        # LinearOperator: can only use svds, no dense fallback
        if r >= min(m, n):
            raise ValueError(
                f"r={r} too large for LinearOperator with shape {H.shape}. "
                f"Need r < min({m}, {n}) = {min(m, n)}"
            )
        U, s_vals, _ = svds(H, k=r)
        idx = np.argsort(s_vals)[::-1]
        return U[:, idx], s_vals[idx]

    # Original dense path (unchanged)
    if r < min(m, n) - 1:
        try:
            U, s_vals, _ = svds(H, k=r)
            idx = np.argsort(s_vals)[::-1]
            return U[:, idx], s_vals[idx]
        except Exception:
            pass

    U_full, s_full, _ = np.linalg.svd(H, full_matrices=False)
    return U_full[:, :r], s_full[:r]


def select_rank_variance(
    H: Union[NDArray[np.floating], ScipyLinearOperator],
    var_explained: float = 0.99,
    var_max_rank: int = 50,
    var_min_rank: int = 1,
) -> tuple[NDArray[np.floating], int, NDArray[np.floating], NDArray[np.floating], bool]:
    """
    Selección de rango local por varianza explicada (criterio X%).

    Selecciona el rango ``r`` como el menor número de componentes que
    explican una fracción ``var_explained`` de la energía total de la
    matriz (Hankel) de referencia::

        r = min{ r' : sum_{i<=r'} s_i^2 / ||H||_F^2 >= var_explained }

    La matriz ``H`` debe estar normalizada por su norma Frobenius
    (p.ej. vía ``_make_scaled_linop``), de modo que la energía total es
    exactamente 1 y ``cumsum(s_i^2)`` es la fracción de varianza
    explicada acumulada.

    Parameters
    ----------
    H : array (m, n) o ScipyLinearOperator
        Matriz (u operador) YA normalizada por ||H||_F.
    var_explained : float
        Fracción de varianza a explicar (0, 1]. P.ej. 0.99 para X=99%.
    var_max_rank : int
        Tope de candidatos a examinar. Si la varianza objetivo no se
        alcanza dentro de este tope, se retorna el tope y
        ``censored=True`` (el rango quedó censurado por el cap).
    var_min_rank : int
        Rango mínimo a retornar (piso).

    Returns
    -------
    U : array, shape (m, r)
        Primeros r vectores singulares izquierdos (descendente).
    r : int
        Rango seleccionado.
    s_vals : array, shape (k,)
        Valores singulares computados (descendente), k <= var_max_rank.
    cum_var : array, shape (k,)
        Varianza explicada acumulada (fracción de ||H||_F^2).
    censored : bool
        True si var_explained no se alcanzó con var_max_rank componentes.

    Notes
    -----
    Este método reviva el test de rango del Step B (Remark 3.6 del paper):
    con ``rank_method='fixed'`` los rangos locales son constantes y el
    test es degenerado; con criterio de varianza cada grabación tiene su
    propio r(s,c) y el test recupera su significado como sonda de
    dimensionalidad dependiente de condición.

    Coste: un único ``svds`` con k = var_max_rank candidatos (líneal en
    el tope, no se re-ejecuta el SVD por candidato).
    """
    m, n = H.shape
    k = int(min(var_max_rank, m - 1, n - 1))
    if k < 1:
        raise ValueError(
            f"var_max_rank demasiado pequeño para H de shape {H.shape}"
        )
    if not (0.0 < var_explained <= 1.0):
        raise ValueError(
            f"var_explained debe estar en (0, 1], got {var_explained}"
        )

    U_k, s_vals = truncated_left_svd_with_values(H, k)

    # H está normalizada: energia total = ||H||_F^2 / ||H||_F^2 = 1
    cum_var = np.cumsum(np.clip(s_vals ** 2, 0.0, None))
    cum_var = np.minimum(cum_var, 1.0)

    hits = np.flatnonzero(cum_var >= var_explained - 1e-12)
    if len(hits) > 0:
        r = int(hits[0]) + 1
        censored = False
    else:
        r = k
        censored = True

    r = max(r, int(var_min_rank))
    r = min(r, k)

    return U_k[:, :r], r, s_vals, cum_var, censored


# ============================================================================
# 2. Rank selection by temporal-block reproducibility (A2 helper)
# ============================================================================

def select_rank_reproducibility(
    X: NDArray[np.floating],
    L: int,
    rmax: int = 20,
    n_blocks: int = 4,
    threshold: float = 0.80,
    strategy: Literal["consecutive", "gap"] = "consecutive",
) -> tuple[int, NDArray[np.floating]]:
    """
    Estima el rank de señal Hankel local via reproducibilidad
    entre bloques temporales.

    Para cada par de bloques (a, b) y cada candidato r::

        R(r) = (1 / n_pairs) * Σ_{a<b} ||U_a(r)^T U_b(r)||_F² / r

    R(r) = 1 ⟹ subespacios idénticos de dimensión r.
    R(r) ≈ 0 ⟹ sin estructura compartida.

    Parameters
    ----------
    X : array, shape (p, T)
    L : int
        Retardos Hankel.
    rmax : int
        Máximo rank candidato.
    n_blocks : int
        Número de bloques temporales.
    threshold : float
        Umbral mínimo de reproducibilidad.
    strategy : {'consecutive', 'gap'}
        'consecutive': el MATLAB original — mayor r consecutivo desde 1.
        'gap': seleccionar por el mayor gap en R(r), más robusto ante
        un componente temprano débil.

    Returns
    -------
    r_selected : int
    R : array, shape (r_allowed,)
        Curva de reproducibilidad.

    CRÍTICA vs MATLAB
    ----------------
    1. **Criterio consecutivo**: El MATLAB selecciona el mayor r tal que
       R(1:r) >= threshold para todo r. Si R(3) = 0.79 (justo debajo),
       se para en r=2 aunque R(4) = 0.95. Esto subestima sistemáticamente
       cuando hay un componente débil interleaved. El parámetro
       ``strategy='gap'`` corrige esto.

    2. **edges con round()**: El MATLAB usa ``round(linspace(...))``, que
       puede producir bloques desiguales o vacíos. Usamos
       ``np.linspace`` con int() que es más predecible.

    3. **Bloques demasiado cortos**: Si T/n_blocks < L + 2, los bloques
       no tienen suficientes columnas Hankel. El MATLAB lanza error; aquí
       damos un warning y reducimos n_blocks automáticamente.

    NOTA (v2): Usa ``compute_block_hankel_fro`` y ``make_block_hankel_linop``
    para evitar materializar la matriz block-Hankel completa.
    """
    X = np.asarray(X, dtype=np.float64)
    p, T = X.shape

    # Validar que los bloques sean factibles
    min_block_len = L + 2
    max_feasible_blocks = T // min_block_len
    if max_feasible_blocks < 2:
        raise ValueError(
            f"Grabación demasiado corta (T={T}) para L={L} con >= 2 bloques. "
            f"Se necesitan al menos {2 * min_block_len} muestras."
        )

    effective_blocks = min(n_blocks, max_feasible_blocks)
    if effective_blocks < n_blocks:
        import warnings

        warnings.warn(
            f"Reduciendo n_blocks de {n_blocks} a {effective_blocks} "
            f"(T={T}, L={L}, se necesitan >={min_block_len} muestras por bloque)",
            stacklevel=2,
        )

    # Dividir en bloques temporales
    edges = np.linspace(0, T, effective_blocks + 1, dtype=int)

    bases: list[NDArray[np.floating]] = []
    r_allowed = rmax

    for b in range(effective_blocks):
        Xb = X[:, edges[b] : edges[b + 1]]
        if Xb.shape[1] <= L:
            raise ValueError(
                f"Bloque {b} tiene solo {Xb.shape[1]} columnas, "
                f"necesitas > {L} para Hankel con L={L}."
            )
        # v2: compute Frobenius norm without materializing H
        hnorm = compute_block_hankel_fro(Xb, L)
        if hnorm < np.finfo(np.float64).eps:
            raise ValueError(f"Bloque {b} tiene Hankel casi cero.")
        # v2: use LinearOperator instead of dense matrix
        linop_b = make_block_hankel_linop(Xb, L)
        linop_b_norm = _make_scaled_linop(linop_b, hnorm)
        rb = min(rmax, linop_b_norm.shape[0], linop_b_norm.shape[1])
        bases.append(truncated_left_svd(linop_b_norm, rb))
        r_allowed = min(r_allowed, rb)

    # Calcular reproducibilidad
    R = np.zeros(r_allowed, dtype=np.float64)
    n_pairs = 0

    for a in range(effective_blocks - 1):
        for b in range(a + 1, effective_blocks):
            n_pairs += 1
            Ua = bases[a]
            Ub = bases[b]
            for r in range(1, r_allowed + 1):
                G = Ua[:, :r].T @ Ub[:, :r]
                R[r - 1] += np.linalg.norm(G, "fro") ** 2 / r

    R /= n_pairs

    # Selección de rank
    if strategy == "consecutive":
        # MATLAB original: mayor r consecutivo desde 1
        r_selected = 1
        for r in range(1, r_allowed + 1):
            if np.all(R[:r] >= threshold):
                r_selected = r
            else:
                break
    elif strategy == "gap":
        # Mayor gap en R(r), buscando el primer descenso pronunciado
        # Añadimos R(0)=1.0 y R(r_allowed+1)=0 como boundary
        R_padded = np.concatenate([[1.0], R, [0.0]])
        drops = -np.diff(R_padded)
        # Excluir el primer drop (de 1.0 a R(1)) que es artificial
        drops[0] = 0.0
        best_gap_idx = np.argmax(drops[1:]) + 1  # +1 por el padding
        r_selected = max(1, best_gap_idx)
    else:
        raise ValueError(f"strategy debe ser 'consecutive' o 'gap', got '{strategy}'")

    return r_selected, R


def _effective_rank_from_spectrum(
    s_vals: NDArray[np.floating],
    var_explained: float,
    r_min: int = 1,
) -> tuple[int, bool]:
    """
    Rango efectivo de un espectro de valores singulares (criterio X%).

    r = min{ r' : sum_{i<=r'} s_i^2 / sum_i s_i^2 >= var_explained }

    Igual que el criterio de ``select_rank_variance`` pero sobre un
    espectro ya computado (usado por el estimador de rango efectivo de
    ``cdhsa_A1_A5`` cuando ``effective_rank=True``).

    Returns
    -------
    (r, censored) : (int, bool)
        censored=True si var_explained no se alcanza con el espectro
        disponible (r queda en la longitud del espectro).
    """
    s2 = np.clip(np.asarray(s_vals, dtype=np.float64) ** 2, 0.0, None)
    total = float(s2.sum())
    if total <= 0.0 or s2.size == 0:
        return max(1, int(r_min)), False
    cum = np.cumsum(s2) / total
    hits = np.flatnonzero(cum >= var_explained - 1e-12)
    if len(hits) > 0:
        return max(int(hits[0]) + 1, int(r_min)), False
    return max(int(s2.size), int(r_min)), True


# ============================================================================
# 3. Main: Steps A1-A5
# ============================================================================

def cdhsa_A1_A5(
    X: list[list[NDArray[np.floating]]],
    L: int,
    *,
    rank_method: Literal["fixed", "reproducibility", "variance"] = "fixed",
    fixed_rank: int = 10,
    rmax: int = 20,
    n_blocks: int = 4,
    repro_threshold: float = 0.80,
    repro_strategy: Literal["consecutive", "gap"] = "consecutive",
    max_common: int = 30,
    prevalence_quantile: float = 0.10,
    var_explained: float = 0.99,
    var_max_rank: int = 50,
    var_min_rank: int = 1,
    col_masks: list | None = None,
    effective_rank: bool = False,
) -> dict:
    """
    CD-HSA Steps A1-A5: estimación del subespacio Hankel común poblacional.

    **A1**: Construir Hankel por bloques y normalizar geométricamente.

    **A2**: Estimar el rank de señal confiable de cada grabación.
      - 'fixed': usar ``fixed_rank`` directamente.
      - 'reproducibilidad': reproducibilidad entre bloques temporales.
      - 'variance': menor r que explica ``var_explained`` de la energía
        de la Hankel embebida (criterio X%), por grabación.

    **A3**: Estimar el subespacio común poblacional vía SVD de la
      concatenación de bases locales B = [U₁ ... U_N] / √N.
      Los valores λ_j = σ_j(B) (VALORES SINGULARES, no su cuadrado;
      homogéneo con Step D y Def. 3.13 del paper) miden la commonalidad
      de cada dirección.

    **A4**: Calcular la matriz de alineación a_{sc,j} = ||U_sc^T w_j||²,
      que cuantifica cuánto de cada dirección común está presente en
      cada grabación individual.

    **A5**: Resumir commonalidad (media) y prevalencia (percentil inferior)
      de cada dirección poblacional.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] es un array (p, T) con los datos EEG del sujeto s,
        condición c. Canales en filas, tiempo en columnas.
    L : int
        Número de retardos Hankel (embedding depth).
    rank_method : {'fixed', 'reproducibility', 'variance'}
        'fixed': usar ``fixed_rank`` directamente (r constante -> el test
        de rango del Step B es degenerado, Remark 3.6 del paper).
        'reproducibility': reproducibilidad entre bloques temporales.
        'variance': menor r que explica ``var_explained`` de la energía
        de la Hankel embebida (criterio X%), por grabación — ver
        ``select_rank_variance``.
    fixed_rank : int
        Rank local fijo cuando ``rank_method='fixed'``.
    var_explained : float
        Fracción de varianza a explicar cuando ``rank_method='variance'``
        (p.ej. 0.99 para X=99%). También se usa como criterio del
        estimador de rango efectivo cuando ``effective_rank=True``.
    var_max_rank : int
        Tope de candidatos cuando ``rank_method='variance'`` (y del
        espectro del estimador efectivo).
    var_min_rank : int
        Piso de rango cuando ``rank_method='variance'`` (y mínimo del
        estimador efectivo).
    rmax : int
        Máximo rank candidato para reproducibilidad.
    n_blocks : int
        Bloques temporales para reproducibilidad.
    repro_threshold : float
        Umbral de reproducibilidad.
    repro_strategy : {'consecutive', 'gap'}
        Estrategia de selección de rank (ver ``select_rank_reproducibility``).
    max_common : int
        Máximo número de direcciones comunes a estimar.
    prevalence_quantile : float
        Percentil inferior para la medida de prevalencia.
    col_masks : list of list of arrays or None
        col_masks[s][c] es una máscara booleana de longitud
        ``T_sc - L + 1`` (columnas de nivel 2) o None. Las columnas
        False se excluyen de TODO el cálculo de esa grabación (SVD,
        normas, energía) — ver ``level2_boundary_mask``. None = sin
        enmascaramiento (comportamiento v3).
    effective_rank : bool
        Si True, calcula ADEMÁS del rango primario un rango efectivo
        por grabación con el criterio X% sobre el espectro de la misma
        Hankel normalizada (un único ``svds`` extra por grabación
        cuando rank_method != 'variance'; gratis cuando = 'variance').
        Se almacena en R['rank_effective'] y permite que el test de
        rango del Step B diga algo incluso con rank_method='fixed'
        (Remark 3.6): es el "estimador de dimensionalidad efectiva".

    Returns
    -------
    result : dict with keys:
        U : list of list of arrays
            U[s][c] shape (p*L, r_sc) — base local Hankel ortonormal.
        rank : ndarray, shape (S, C)
            Rank local seleccionado para cada grabación.
        rank_effective : ndarray, shape (S, C)
            Rango efectivo X% — SOLO presente cuando
            ``effective_rank=True`` (junto con 'rank_effective_censored'
            y 'svals'); si no, la clave no existe.
        hankel_norm : ndarray, shape (S, C)
            ||H_sc||_F original (antes de normalización; con
            enmascaramiento, de las columnas válidas).
        repro_curve : list of list of arrays or None
            Curvas de reproducibilidad (solo si rank_method='reproducibility').
        W : ndarray, shape (p*L, q)
            Direcciones comunes poblacionales (columnas ortonormales).
        lambda_ : ndarray, shape (q,)
            λ_j = σ_j(B) ∈ [0, 1] (valores singulares de B).
        alignment : ndarray, shape (S*C, q)
            a_{sc,j} = ||U_sc^T w_j||² para cada grabación y dirección.
        sc_index : ndarray, shape (S*C, 2)
            (s, c) correspondiente a cada fila de alignment.
        mean_alignment : ndarray, shape (q,)
        median_alignment : ndarray, shape (q,)
        prevalence : ndarray, shape (q,)
        min_alignment : ndarray, shape (q,)
        col_masks : list of list of arrays or None
            La máscara usada por grabación (None si no se enmascaró).
        boundary_dropped : ndarray, shape (S, C)
            Número de columnas de nivel 2 excluidas por la máscara
            (0 si no hay máscara).
        S : int, C : int, p : int, d : int
            Dimensiones del problema.
        L_used : int
            L tal cual se pasó.

    Notes
    -----
    El enmascaramiento de fronteras (``col_masks``) es equivalente a
    eliminar las columnas de nivel 2 que cruzan fronteras de
    concatenación antes de cualquier cálculo: las columnas enmascaradas
    son tratadas como cero y no contribuyen ni a la SVD ni a las normas.
    Con ``rank_method='reproducibility'`` el enmascaramiento no está
    soportado (los bloques temporales internos del método no respetan
    las fronteras) y se lanza NotImplementedError.

    CRÍTICA GLOBAL vs MATLAB
    ------------------------
    - La lógica es equivalente al MATLAB cdhsa_A1_A5.m.
    - Corregimos el orden ascendente/descendente de scipy svds.
    - Añadimos ``repro_strategy='gap'`` como alternativa al criterio
      consecutivo del MATLAB.
    - Añadimos warnings cuando se reducen bloques automáticamente.
    - Normalización por ||H||_F: conservada por compatibilidad con B/C,
      aunque no afecta a U_sc.
    - No construimos los proyectores pL×pL explícitamente (igual que MATLAB).

    NOTA (v2): Esta versión usa ``scipy.sparse.linalg.LinearOperator``
    en lugar de materializar la matriz block-Hankel H de shape (p*L, K).
    Esto reduce el pico de memoria de ~12 GB a ~O(p*K) por bloque temporal.
    """
    # ---- Dimensiones del problema ----
    S = len(X)
    C = len(X[0])
    N = S * C

    # Validar dimensionalidad
    p = np.asarray(X[0][0]).shape[0]
    for s in range(S):
        for c in range(C):
            Xsc = np.asarray(X[s][c])
            if Xsc.ndim != 2 or Xsc.shape[0] != p:
                raise ValueError(
                    f"X[{s}][{c}] tiene shape {Xsc.shape}, "
                    f"se espera (p={p}, T) con T >= L+2={L + 2}"
                )
            if Xsc.shape[1] < L + 2:
                raise ValueError(
                    f"X[{s}][{c}] tiene T={Xsc.shape[1]}, "
                    f"demasiado corto para L={L} (necesita >= {L + 2})"
                )

    # Validar col_masks y conflicto con 'reproducibility'
    if col_masks is not None:
        if len(col_masks) != S or any(len(row) != C for row in col_masks):
            raise ValueError(
                f"col_masks debe tener la forma (S={S}, C={C}) "
                f"(lista de listas de máscaras o None)."
            )
        if rank_method == "reproducibility":
            raise NotImplementedError(
                "El enmascaramiento de fronteras no está soportado con "
                "rank_method='reproducibility' (sus bloques temporales "
                "internos cruzan las fronteras de concatenación). Use "
                "rank_method='fixed' o 'variance' con super-sujetos."
            )

    # ---- A1-A2: Hankel + normalización + rank local ----
    U_all: list[list[NDArray]] = [[None] * C for _ in range(S)]
    ranks = np.zeros((S, C), dtype=int)
    hankel_norms = np.zeros((S, C), dtype=np.float64)
    repro_curves: list[list[NDArray | None]] = [[None] * C for _ in range(S)]
    var_curves: list[list[NDArray | None]] = [[None] * C for _ in range(S)]
    var_achieved = np.full((S, C), np.nan, dtype=np.float64)
    var_censored = np.zeros((S, C), dtype=bool)
    boundary_dropped = np.zeros((S, C), dtype=int)
    rank_effective = np.zeros((S, C), dtype=int) if effective_rank else None
    eff_censored = np.zeros((S, C), dtype=bool) if effective_rank else None
    svals_cell: list[list[NDArray | None]] = (
        [[None] * C for _ in range(S)] if effective_rank else None
    )

    for s in range(S):
        for c in range(C):
            Xi = np.asarray(X[s][c], dtype=np.float64)
            mask_sc = None
            if col_masks is not None and col_masks[s][c] is not None:
                mask_sc = np.asarray(col_masks[s][c])
                if mask_sc.shape[0] != Xi.shape[1] - L + 1:
                    raise ValueError(
                        f"col_masks[{s}][{c}] tiene longitud {mask_sc.shape[0]} "
                        f"pero la matriz de nivel 2 de X[{s}][{c}] tiene "
                        f"{Xi.shape[1] - L + 1} columnas."
                    )
                boundary_dropped[s, c] = int(
                    np.sum(~mask_sc.astype(bool))
                )

            # v2: compute Frobenius norm without materializing H
            # (v4: de las columnas válidas si hay máscara de fronteras)
            hnorm = compute_block_hankel_fro(Xi, L, col_mask=mask_sc)
            if hnorm <= np.finfo(np.float64).eps:
                raise ValueError(
                    f"Hankel casi cero en sujeto {s}, condición {c}"
                )
            hankel_norms[s, c] = hnorm

            # A1: normalización geométrica via LinearOperator (no cambia U_sc, pero la
            # guardamos para B/C que necesita la escala original)
            linop = make_block_hankel_linop(Xi, L, col_mask=mask_sc)
            linop_norm = _make_scaled_linop(linop, hnorm)

            # A2: rank local
            if rank_method == "fixed":
                K_hankel = Xi.shape[1] - L + 1
                r = min(fixed_rank, p * L, K_hankel)
                repro = None
            elif rank_method == "reproducibility":
                r, repro = select_rank_reproducibility(
                    Xi,
                    L,
                    rmax=rmax,
                    n_blocks=n_blocks,
                    threshold=repro_threshold,
                    strategy=repro_strategy,
                )
            elif rank_method == "variance":
                Ui_var, r, s_vals_sc, cum_var_sc, censored_sc = select_rank_variance(
                    linop_norm,
                    var_explained=var_explained,
                    var_max_rank=var_max_rank,
                    var_min_rank=var_min_rank,
                )
                var_curves[s][c] = cum_var_sc
                var_achieved[s, c] = float(cum_var_sc[r - 1]) if r >= 1 else np.nan
                var_censored[s, c] = bool(censored_sc)
                if censored_sc:
                    import warnings
                    warnings.warn(
                        f"rank_method='variance': var_explained={var_explained} "
                        f"no alcanzada con var_max_rank={var_max_rank} candidatos "
                        f"en (s={s}, c={c}); r censurado en {r} "
                        f"(var lograda={cum_var_sc[-1]:.4f})",
                        stacklevel=2,
                    )
                repro = None
                # Ui ya computada por select_rank_variance
                Ui = Ui_var
                U_all[s][c] = Ui
                ranks[s, c] = r
                if effective_rank:
                    # Con 'variance' el rango seleccionado YA es el efectivo X%.
                    rank_effective[s, c] = r
                    eff_censored[s, c] = bool(censored_sc)
                    svals_cell[s][c] = s_vals_sc
                continue
            else:
                raise ValueError(
                    f"rank_method debe ser 'fixed', 'reproducibility' o "
                    f"'variance', got '{rank_method}'"
                )

            Ui = truncated_left_svd(linop_norm, r)
            U_all[s][c] = Ui
            ranks[s, c] = r
            repro_curves[s][c] = repro

            # Rango efectivo (estimador de dimensionalidad, Remark 3.6):
            # un único svds extra sobre la MISMA Hankel normalizada.
            if effective_rank:
                m_op, n_op = linop_norm.shape
                k_probe = int(min(max(var_max_rank, r), m_op - 1, n_op - 1))
                if k_probe >= 1:
                    _, s_probe = truncated_left_svd_with_values(
                        linop_norm, k_probe
                    )
                    svals_cell[s][c] = s_probe
                    r_eff, cen_eff = _effective_rank_from_spectrum(
                        s_probe, var_explained, r_min=var_min_rank
                    )
                    rank_effective[s, c] = int(r_eff)
                    eff_censored[s, c] = bool(cen_eff)
                else:
                    rank_effective[s, c] = int(r)
                    eff_censored[s, c] = False
                    svals_cell[s][c] = np.array([])

    # ---- A3: subespacio común poblacional ----
    # M₀ = (1/N) Σ U_sc U_sc^T
    # Equivale a SVD de B = [U₁...U_N]/√N
    Ucat = np.concatenate(
        [U_all[s][c] for s in range(S) for c in range(C)],
        axis=1,
    )
    B = Ucat / np.sqrt(N)

    qmax = min(max_common, B.shape[0], B.shape[1])
    W, sigma_B = truncated_left_svd_with_values(B, qmax)
    # v4: λ_j = σ_j(B) — VALOR SINGULAR (homogéneo con Step D / Def. 3.13
    # del paper; antes era σ², lo que hacía λ inconsistente entre A y D).
    # σ_j(B) ≤ 1 siempre (B tiene columnas ortonormales / √N), y como el
    # cambio es monótono el criterio (i) de A6 no se altera.
    lambda_ = np.clip(sigma_B, 0.0, 1.0)  # clipping numérico

    # ---- A4: alineación por grabación ----
    alignment = np.zeros((N, qmax), dtype=np.float64)
    sc_index = np.zeros((N, 2), dtype=int)
    row = 0
    for s in range(S):
        for c in range(C):
            Ui = U_all[s][c]
            G = W.T @ Ui  # shape (qmax, r_sc)
            alignment[row, :] = np.sum(G ** 2, axis=1)
            sc_index[row, :] = [s, c]
            row += 1

    # ---- A5: commonalidad y prevalencia ----
    mean_alignment = np.mean(alignment, axis=0)
    median_alignment = np.median(alignment, axis=0)
    prevalence = np.quantile(alignment, prevalence_quantile, axis=0)
    min_alignment = np.min(alignment, axis=0)

    result = {
        "U": U_all,
        "rank": ranks,
        "hankel_norm": hankel_norms,
        "repro_curve": repro_curves,
        "rank_method": rank_method,
        "var_explained": var_explained if rank_method == "variance" else None,
        "var_curve": var_curves,
        "var_explained_achieved": var_achieved,
        "var_censored": var_censored,
        "col_masks": col_masks,
        "boundary_dropped": boundary_dropped,
        "W": W,
        "lambda_": lambda_,
        "alignment": alignment,
        "sc_index": sc_index,
        "mean_alignment": mean_alignment,
        "median_alignment": median_alignment,
        "prevalence": prevalence,
        "min_alignment": min_alignment,
        "S": S,
        "C": C,
        "p": p,
        "d": p * L,
        "L_used": L,
        "N": N,
    }
    if effective_rank:
        result["effective_rank"] = True
        result["rank_effective"] = rank_effective
        result["rank_effective_censored"] = eff_censored
        result["svals"] = svals_cell
    return result
