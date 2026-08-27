"""
src.pipelines.run_cdhsa_v2 - Memory-Optimized CD-HSA Orchestrator
================================================================

Same pipeline as run_cdhsa.py but uses the _v2 modules that avoid
materializing the block-Hankel matrix.  The changes are:

  1. cdhsa_A1_A5  →  cdhsa_A1_A5 from a_common_subspace_v2.py
     (LinearOperator + svds, H never materialized)
  2. cdhsa_BC_condition_tests  →  cdhsa_BC_condition_tests from b_energy_v2.py
     (energy computed block-by-block, H never materialized)
  3. cdhsa_D_condition_specific_modes → from d_condition_specific_v2.py
     (removed dead import of build_block_hankel; algorithm unchanged
      since D operates only on small U/W0 matrices)

Expected memory reduction: from ~97 GB peak to ~2-5 GB peak
(with the same parameters that produced 97 GB before).

Usage (CLI — drop-in replacement for run_cdhsa.py)::

    python -m src.pipelines.run_cdhsa_v2 \\
        --session session1 \\
        --tasks eyesclosed music \\
        --n-super-subjects 5 \\
        --total-subjects 60 \\
        --t-start 100 --t-end 200 \\
        --L 10 --hankel-depth 10 \\
        --fixed-rank 15 --a6-n-null 500 --bc-n-perm 5000

Usage (programmatic)::

    from src.pipelines.run_cdhsa_v2 import run_cdhsa_v2, CDHSAConfig
    cfg = CDHSAConfig(fixed_rank=10, a6_n_null=100, bc_n_perm=5000)
    result = run_cdhsa_v2(X, L, cfg)
    print(result.summary())

The CLI is a 100% drop-in replacement for run_cdhsa.py.  The batch
runner (run_batch_cdhsa.py) can point to this module via
--pipeline-script or by changing pipeline_module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# =====================================================================
# Reuse CDHSAConfig and CDHSAResult from the original module
# =====================================================================
try:
    from src.pipelines.run_cdhsa import CDHSAConfig, CDHSAResult
except ImportError:
    from run_cdhsa import CDHSAConfig, CDHSAResult


# =====================================================================
# run_cdhsa_v2 (memory-optimized)
# =====================================================================

def run_cdhsa_v2(
    X: list[list[NDArray[np.floating]]],
    L: int,
    config: CDHSAConfig | None = None,
) -> CDHSAResult:
    """Run the full CD-HSA pipeline (memory-optimized version).

    Parameters
    ----------
    X : list of list of arrays, shape (p, T)
        X[s][c] is the data matrix (e.g. first-level Hankel) for subject s,
        condition c, with shape (p, T).
    L : int
        Block-Hankel embedding depth.
    config : CDHSAConfig, optional

    Returns
    -------
    result : CDHSAResult
        Identical structure to run_cdhsa().

    Memory optimization
    -------------------
    - Steps A1-A5: uses LinearOperator + svds instead of materializing
      the block-Hankel H of shape (p*L, K).  Peak memory per (s,c):
      O(p*K) instead of O(p*L*K).
    - Steps B/C: computes H^T W_k and ||H||_F^2 block-by-block from X
      instead of materializing H.  Peak memory per (s,c):
      O(K * dim_k) instead of O(p*L*K).
    - Step D: uses d_condition_specific_v2.py which removes the dead
      import of build_block_hankel. The algorithm is identical since D
      only operates on small U_sc and W0 matrices.
    - Steps A6, C (tangent) are unchanged (they operate on small
      matrices already).
    """
    # Import v2 modules
    try:
        from src.cdhsa.a_common_subspace_v2 import cdhsa_A1_A5 as cdhsa_A1_A5_v2
    except ImportError:
        from src.cdhsa.a_common_subspace_v2 import cdhsa_A1_A5 as cdhsa_A1_A5_v2

    try:
        from src.cdhsa.b_energy_v2 import cdhsa_BC_condition_tests as cdhsa_BC_v2
    except ImportError:
        from src.cdhsa.b_energy_v2 import cdhsa_BC_condition_tests as cdhsa_BC_v2

    # Steps A6, C are unchanged (operate on small matrices)
    try:
        from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank
    except ImportError:
        from src.cdhsa.a6_common_rank import cdhsa_A6_common_rank

    try:
        from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test
    except ImportError:
        from src.cdhsa.c_geometry import cdhsa_tangent_geometry_test

    # Step D: v2 (removed dead build_block_hankel import, algorithm identical)
    try:
        from src.cdhsa.d_condition_specific_v2 import cdhsa_D_condition_specific_modes as cdhsa_D_v2
    except ImportError:
        from src.cdhsa.d_condition_specific_v2 import cdhsa_D_condition_specific_modes as cdhsa_D_v2

    if config is None:
        config = CDHSAConfig()

    # ---- A1-A5 (v2: LinearOperator) ----
    print("[v2] Running A1-A5 (memory-optimized: LinearOperator + svds)...")
    R = cdhsa_A1_A5_v2(
        X, L,
        rank_method=config.rank_method,
        fixed_rank=config.fixed_rank,
        rmax=config.rmax,
        n_blocks=config.n_blocks,
        repro_threshold=config.repro_threshold,
        repro_strategy=config.repro_strategy,
        max_common=config.max_common,
        prevalence_quantile=config.prevalence_quantile,
    )
    print(f"[v2] A1-A5 done. {R['S']}x{R['C']} recordings, d={R['d']}, "
          f"p={R['p']}, qmax={len(R['lambda_'])}")

    # ---- A6 (unchanged) ----
    a6_max = (config.a6_max_common if config.a6_max_common > 0
             else min(20, len(R['lambda_'])))
    A6 = cdhsa_A6_common_rank(R, opts={
        'max_common': a6_max,
        'n_folds': config.a6_n_folds,
        'n_null': config.a6_n_null,
        'alpha': config.a6_alpha,
        'seed': config.a6_seed,
    })
    print(f"[v2] A6 done. r0 = {A6['r0']}")

    # ---- B/C (v2: block-wise energy) ----
    BC = None
    if not config.skip_bc and A6['r0'] >= 1:
        bc_opts = {
            'blocks': config.bc_blocks,
            'energy_metric': config.bc_energy_metric,
            'geometry_metric': config.bc_geometry_metric,
            'n_perm': config.bc_n_perm,
            'seed': config.bc_seed,
            'alpha': config.bc_alpha,
        }
        if config.bc_condition_names:
            bc_opts['condition_names'] = config.bc_condition_names
        print(f"[v2] Running B/C (memory-optimized: block-wise energy, "
              f"{config.bc_n_perm} perms)...")
        BC = cdhsa_BC_v2(X, L, R, A6, opts=bc_opts)
        n_sig_e = int(np.sum(BC['sig_energy_maxF']))
        n_sig_g = int(np.sum(BC['sig_geometry_maxF']))
        print(f"[v2] B/C done. sig_energy={n_sig_e}, sig_geom={n_sig_g}")

    # ---- C: tangent geometry (unchanged) ----
    G = None
    if not config.skip_tangent and A6['r0'] >= 1:
        t_blocks = config.tangent_blocks
        if t_blocks is None:
            t_blocks = [np.arange(1, A6['r0'] + 1, dtype=int)]
        G = cdhsa_tangent_geometry_test(R, A6, blocks=t_blocks, opts={
            'n_perm': config.tangent_n_perm,
            'seed': config.tangent_seed,
            'alpha': config.tangent_alpha,
        })

    # ---- D: condition-specific (v2) ----
    D = None
    if not config.skip_d and A6['r0'] >= 1:
        D = cdhsa_D_v2(X, L, R, A6, opts={
            'max_specific': config.d_max_specific,
            'residual_rank_method': config.d_residual_rank_method,
            'residual_rank_threshold': config.d_residual_rank_threshold,
            'fixed_residual_rank': config.d_fixed_residual_rank,
            'prevalence_quantile': config.prevalence_quantile,
        })

    return CDHSAResult(config=config, R=R, A6=A6, BC=BC, G=G, D=D)


# =====================================================================
# CLI — drop-in replacement for run_cdhsa.py
# =====================================================================
# Import CLI helpers from the original module.  The only difference is
# that we call run_cdhsa_v2() instead of run_cdhsa() for step 3.
# =====================================================================

try:
    from src.pipelines.run_cdhsa import (
        _parse_args as _orig_parse_args,
        _resolve_out_dir as _orig_resolve_out_dir,
        save_results as _orig_save_results,
        build_hankel_from_eeg,
        build_hankel_single_ss,
        characterize_hankel_matrices,
    )
    _HAS_CLI = True
except ImportError:
    _HAS_CLI = False


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — identical to run_cdhsa.main() but uses v2 pipeline.

    This function is a drop-in replacement.  It reuses all the
    data-loading, argument-parsing, and result-saving logic from the
    original ``run_cdhsa`` module.  The ONLY change is that step 3
    calls ``run_cdhsa_v2()`` instead of ``run_cdhsa()``.
    """
    if not _HAS_CLI:
        print("ERROR: Cannot import CLI helpers from run_cdhsa.", file=sys.stderr)
        print("Make sure run_cdhsa.py is importable.", file=sys.stderr)
        return 1

    args = _orig_parse_args(argv)
    verbose = "INFO" if args.verbose else None

    # 0. Resolve output directory
    if not args.no_save:
        out_dir = _orig_resolve_out_dir(args)
    else:
        out_dir = None

    # 1. Build Hankel matrices (same as original)
    single_mode = args.super_subject_id is not None

    if single_mode:
        print("=" * 70)
        print("  CONSTRUYENDO MATRICES DE HANKEL (SINGLE SUPER-SUBJECT)")
        print("=" * 70)

        X, hankel_info = build_hankel_single_ss(
            super_subject_id=args.super_subject_id,
            session=args.session,
            tasks=args.tasks,
            subjects_per_super_subject=args.subjects_per_super_subject,
            subject_start_offset=args.subject_start_offset,
            db_path=args.db_path,
            t_start=args.t_start,
            t_stop=args.t_end,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            hankel_depth=args.hankel_depth,
            verbose=verbose,
        )
    else:
        print("=" * 70)
        print("  CONSTRUYENDO MATRICES DE HANKEL DESDE SUPER-SUJETOS")
        print("=" * 70)

        X, hankel_info = build_hankel_from_eeg(
            session=args.session,
            tasks=args.tasks,
            n_super_subjects=args.n_super_subjects,
            total_subjects=args.total_subjects,
            subject_start_offset=args.subject_start_offset,
            db_path=args.db_path,
            t_start=args.t_start,
            t_stop=args.t_end,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            hankel_depth=args.hankel_depth,
            verbose=verbose,
        )

    # 2. Characterize (same as original)
    characterization = characterize_hankel_matrices(X, hankel_info)
    print(characterization)

    n_valid = sum(
        1 for s in range(len(X)) for c in range(len(X[s]))
        if X[s][c].size > 0
    )
    if n_valid == 0:
        print("\n[ERROR] No se construyeron matrices validas.")
        return 1

    # 3. Configure and run CD-HSA (v2!)
    cfg = CDHSAConfig(
        fixed_rank=args.fixed_rank,
        rank_method=args.rank_method,
        a6_n_null=args.a6_n_null,
        bc_n_perm=args.bc_n_perm,
        bc_condition_names=list(args.tasks),
        d_max_specific=args.d_max_specific,
        skip_bc=args.skip_bc,
        skip_tangent=args.skip_tangent,
        skip_d=args.skip_d,
    )

    S = len(X)
    print("\n" + "=" * 70)
    print("  EJECUTANDO CD-HSA (v2: memory-optimized)")
    print("=" * 70)
    print(f"  Modo                  : {'single-SS' if single_mode else 'multi-SS'}")
    if single_mode:
        print(f"  Super-sujeto ID        : {args.super_subject_id}")
        print(f"  Subjects por SS       : {args.subjects_per_super_subject}")
    else:
        print(f"  Super-sujetos (S)     : {args.n_super_subjects}")
        print(f"  Subjects por SS       : {args.total_subjects // args.n_super_subjects}")
    print(f"  S (matrices)          : {S}")
    print(f"  Condiciones (C)       : {len(args.tasks)}")
    print(f"  L (subespacio)        : {args.L}")
    print(f"  fixed_rank            : {cfg.fixed_rank}")
    print(f"  a6_n_null             : {cfg.a6_n_null}")
    print(f"  bc_n_perm             : {cfg.bc_n_perm}")
    if out_dir is not None:
        print(f"  Out dir               : {out_dir}")
    print("")
    sys.stdout.flush()

    # >>> THIS IS THE ONLY LINE THAT DIFFERS FROM run_cdhsa.py <<<
    result = run_cdhsa_v2(X, args.L, cfg)

    # 4. Results
    print("")
    print(result.summary())

    # 5. Save (same as original)
    if out_dir is not None:
        _orig_save_results(
            out_dir=out_dir,
            X=X,
            hankel_info=hankel_info,
            characterization=characterization,
            result=result,
            cfg=cfg,
            L=args.L,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
