#!/usr/bin/env python3
"""
test_ss_shuffle_batch.py
========================
Suite de verificacion del bloque ``ss_shuffle`` integrado en
``src/batch_runs/run_batch_cdhsa.py`` (TODO-EN-UNO por JSON, sin
scripts aparte).

Que verifica (74 checks):

  A. Validacion del bloque ``ss_shuffle`` en ``_load_params``
  B. Aislamiento total de rutas (results/cache/params + env BASE_*)
  C. Nombres de mode_map (token taskshuf) + contrato de rutas del run
     v4 real del usuario
  D. Fase de reuso/migracion (mueve/copia el run previo + mode_map +
     checkpoint; guardia _taskshuf{seed})
  E. Red de seguridad de ``_filter_todo`` (arrays en disco => skip)
  F. Fase de figuras de influencia (skip por marcador, --no-topo,
     fallos)
  G. E2E de ``_run_inner``: corrida completa de cero (train + extract
     + figuras), corrida con reuso del v4 (train NO invocado) y
     idempotencia de re-runs
  H. Sin bloque ``ss_shuffle`` => comportamiento historico intacto

Uso::

    python scripts/test_ss_shuffle_batch.py

(No requiere pytest; tampoco requiere la BD ni las dependencias
cientificas: los subprocesos se reemplazan por fakes.)
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Mini-framework
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        _FAILURES.append(name + (f"  ({extra})" if extra else ""))
        print(f"  [FAIL] {name}{('  (' + extra + ')') if extra else ''}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Entorno + import fresco del runner (src.utils.config lee el .env al
# importar: presetear las variables ANTES de cada reload)
# ---------------------------------------------------------------------------

_ENV_KEYS = [
    "BASE_RESULTS_PATH", "BASE_CACHE_PATH", "BASE_PARAMS_FILE",
    "LOGGING_BASE_PATH", "DB_TEST_RETEST_GEDAI_PATH",
    "DATA_SIMULATIONS_CACHE_PATH", "NTFY_CHANNEL",
]


class Tree:
    """Arbol temporal con results/cache/params/logs 'reales'."""

    def __init__(self, base: Path):
        self.base = base
        self.results = base / "results"
        self.cache = base / "cache"
        self.params = base / "params"
        self.logs = base / "logs"
        for d in (self.results, self.cache, self.params, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def apply_env(self) -> None:
        os.environ["BASE_RESULTS_PATH"] = str(self.results)
        os.environ["BASE_CACHE_PATH"] = str(self.cache)
        os.environ["BASE_PARAMS_FILE"] = str(self.params)
        os.environ["LOGGING_BASE_PATH"] = str(self.logs)
        os.environ["DB_TEST_RETEST_GEDAI_PATH"] = str(self.base / "db")
        os.environ["DATA_SIMULATIONS_CACHE_PATH"] = str(self.cache / "sim")
        os.environ.pop("NTFY_CHANNEL", None)


def _fresh_modules():
    """Recarga src.utils.config y el runner con el entorno actual."""
    for mod in list(sys.modules):
        if mod == "src" or mod.startswith("src."):
            del sys.modules[mod]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import src.utils.config as cfg
    cfg = importlib.reload(cfg)
    import src.batch_runs.run_batch_cdhsa as rb
    rb = importlib.reload(rb)
    return rb


def _make_params(**over) -> dict:
    """Params del control (v4, como el JSON entregado) mutables."""
    p = {
        "experiment_label": "multiss_cdhsa_5ss_taskshuffle_control_seed20261022",
        "super_subjects": {
            "n_super_subjects": 5,
            "subjects_per_super_subject": 12,
            "subject_start_offset": 1,
            "total_subjects": 60,
            "task_shuffle": {"enabled": True, "seed": 20261022,
                             "extract_modes": True},
        },
        "sessions": ["session1"],
        "tasks": ["eyesclosed", "mathematic"],
        "time_window": {"t_start": 100.0, "t_end": 200.0},
        "cdhsa_params": {
            "L": 10, "hankel_depth": 10, "l_freq": 1.0, "h_freq": 40.0,
            "fixed_rank": 20, "rank_method": "variance",
            "var_explained": 0.99, "var_max_rank": 50, "var_min_rank": 1,
            "a6_max_common": 0, "a6_null_type": "haar", "a6_n_null": 200,
            "bc_n_perm": 5000, "tangent_blocks_mode": "cumulative",
            "d_max_specific": 16, "d_rank_adaptive": True,
            "d_n_null_specific": 200, "d_alpha_specific": 0.05,
            "d_loso": True, "d_loso_n_perm": 1000,
            "run_replica_consistency": True, "no_boundary_mask": False,
            "skip_bc": False, "skip_tangent": False, "skip_d": False,
        },
        "latent_plots": {"enabled": False},
        "execution": {
            "max_workers": 1, "delay": 0.0, "run_comparison": False,
            "mode_extract_top_n": 4,
        },
    }
    p.update(over)
    return p


def _add_ss_shuffle(p: dict, **over) -> dict:
    blk = {
        "enabled": True, "root_name": "ss_shufle",
        "reuse_existing_run": True, "reuse_copy": False,
        "influence_figures": True,
    }
    blk.update(over)
    p["ss_shuffle"] = blk
    return p


# Ruta relativa del job dentro del arbol results (contrato con el run
# v4 REAL del usuario; ver _get_output_dir / _rank_tag).
_EXPECTED_REL = (
    "cdhsa/session1/nSS5_L10_var99c50_a6n200_bcn5000"
    "/1.0-40.0Hz_depth10"
    "/from100.0s_to200.0s_eyesclosed_mathematic_taskshuf20261022"
)
_EXPECTED_KEY = (
    "multiss_5|session1|eyesclosed+mathematic|100.0|200.0|L10|var99c50"
    "|mc30|cumu|bm1|ts20261022"
)


def _seed_old_v4_run(tree: Tree, with_mode_map: bool = False) -> Path:
    """Crea el run v4 'real' (arbol viejo) como lo dejo el usuario."""
    out = tree.results / _EXPECTED_REL
    out.mkdir(parents=True, exist_ok=True)
    (out / "cdhsa_arrays.npz").write_bytes(b"fake-arrays")
    (out / "hankel_matrices.npz").write_bytes(b"fake-hankel")
    (out / "hankel_info.json").write_text("{}")
    if with_mode_map:
        mm = tree.params / (
            "mode_map_4_modes_session1_eyesclosed_mathematic_"
            "100.0s-200.0s_taskshuf20261022.json"
        )
        mm.write_text("{}")
    return out


# ---------------------------------------------------------------------------
# Fake Popen: sustituye TODOS los subprocesos del batch
# ---------------------------------------------------------------------------

class FakePopen:
    calls: list[list[str]] = []

    def __init__(self, cmd, stdout=None, stderr=None, text=False, env=None):
        self.cmd = [str(c) for c in cmd]
        FakePopen.calls.append(self.cmd)
        self.pid = 424242
        self.returncode = 0
        self._lines = ["[FAKE] ok"]
        self._apply_side_effects()
        self.stdout = iter(self._lines)

    # -- side effects ------------------------------------------------------
    def _apply_side_effects(self) -> None:
        c = self.cmd
        if "src.cdhsa.extract_mode_indices" in c:
            i = c.index("-o")
            Path(c[i + 1]).write_text(json.dumps({"fake": "mode_map"}))
        elif any("analyze_channel_influence_taskshuffle" in x for x in c):
            i = c.index("--results-dir")
            rd = Path(c[i + 1])
            m = re.search(r"_taskshuf(\d+)$", rd.name)
            seed = m.group(1) if m else "0"
            fig = rd / f"channel_influence_taskshuf{seed}"
            fig.mkdir(parents=True, exist_ok=True)
            (fig / "summary_channel_influence.json").write_text("{}")
        elif "src.pipelines.run_cdhsa" in c:
            # El pipeline real resuelve su out_dir desde --out-dir +
            # config; el test le inyecta la ruta exacta via
            # --__fake-outdir (ver _patch_build_command).
            i = c.index("--__fake-outdir")
            out = Path(c[i + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "cdhsa_arrays.npz").write_bytes(b"fake-arrays")
            (out / "hankel_matrices.npz").write_bytes(b"fake-hankel")
            (out / "hankel_info.json").write_text("{}")

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


def _patch_popen(rb, fake=FakePopen):
    rb.subprocess.Popen = fake
    rb._HAS_PSUTIL = False
    rb._HAS_MEM_TRACKER = False


def _patch_build_command(runner):
    """Anade al cmd del pipeline un arg solo-test con el out_dir exacto
    del job (el pipeline real lo resuelve solo; el fake lo necesita)."""
    orig = runner._build_command

    def wrapped(job):
        cmd = orig(job)
        cmd += ["--__fake-outdir", str(runner._get_output_dir(job))]
        return cmd

    runner._build_command = wrapped


def _invoked(module_token: str) -> int:
    n = 0
    for c in FakePopen.calls:
        if any(module_token in str(x) for x in c):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def suite_a_validation(tmp: Path) -> None:
    section("A. Validacion del bloque ss_shuffle (_load_params)")

    tree = Tree(tmp / "a")
    tree.apply_env()
    rb = _fresh_modules()

    def _load_exit(params: dict):
        f = tmp / "a.json"
        f.write_text(json.dumps(params))
        try:
            rb._load_params(f)
            return None
        except SystemExit as e:
            return e.code

    base = _make_params()

    # A1: params validos (con ss_shuffle) pasan la validacion
    r = _load_exit(_add_ss_shuffle(_make_params()))
    check("A1: JSON completo con ss_shuffle valida", r is None)

    # A2: ss_shuffle no-dict -> exit 1
    r = _load_exit(_make_params(ss_shuffle="nope"))
    check("A2: ss_shuffle no-objeto -> exit(1)", r == 1)

    # A3: enabled no-bool -> exit 1
    bad = _add_ss_shuffle(_make_params())
    bad["ss_shuffle"]["enabled"] = "yes"
    r = _load_exit(bad)
    check("A3: ss_shuffle.enabled no-bool -> exit(1)", r == 1)

    # A4: root_name con ruta -> exit 1
    bad = _add_ss_shuffle(_make_params())
    bad["ss_shuffle"]["root_name"] = "../escapes"
    r = _load_exit(bad)
    check("A4: root_name con ruta -> exit(1)", r == 1)

    # A5: root_name ".." -> exit 1
    bad = _add_ss_shuffle(_make_params())
    bad["ss_shuffle"]["root_name"] = ".."
    r = _load_exit(bad)
    check("A5: root_name '..' -> exit(1)", r == 1)

    # A6: influence_figures sin task_shuffle -> exit 1
    p = _add_ss_shuffle(_make_params())
    p["ss_shuffle"]["influence_figures"] = True
    p["super_subjects"]["task_shuffle"] = {"enabled": False, "seed": 1}
    r = _load_exit(p)
    check("A6: influence_figures sin task_shuffle -> exit(1)", r == 1)

    # A7: influence_figures sin extract_modes -> exit 1
    p = _add_ss_shuffle(_make_params())
    p["super_subjects"]["task_shuffle"]["extract_modes"] = False
    r = _load_exit(p)
    check("A7: influence_figures sin extract_modes -> exit(1)", r == 1)

    # A8: influence_figures con top_n=0 -> exit 1
    p = _add_ss_shuffle(_make_params())
    p["execution"]["mode_extract_top_n"] = 0
    r = _load_exit(p)
    check("A8: influence_figures con top_n=0 -> exit(1)", r == 1)

    # A9: reuse_existing_run sin task_shuffle -> exit 1
    p = _add_ss_shuffle(_make_params())
    p["ss_shuffle"]["reuse_existing_run"] = True
    p["super_subjects"]["task_shuffle"] = {"enabled": False, "seed": 1}
    r = _load_exit(p)
    check("A9: reuse_existing_run sin task_shuffle -> exit(1)", r == 1)

    # A10: extract_modes no-bool -> exit 1
    p = _make_params()
    p["super_subjects"]["task_shuffle"]["extract_modes"] = "si"
    r = _load_exit(p)
    check("A10: task_shuffle.extract_modes no-bool -> exit(1)", r == 1)

    # A11: sin bloque ss_shuffle -> valida normal (historico)
    r = _load_exit(_make_params())
    check("A11: sin bloque ss_shuffle valida (historico)", r is None)


def suite_b_isolation(tmp: Path) -> None:
    section("B. Aislamiento total de rutas (results/cache/params + env)")

    tree = Tree(tmp / "b")
    tree.apply_env()
    rb = _fresh_modules()

    runner = rb.CDHSABatchRunner(_add_ss_shuffle(_make_params()))

    check("B1: output_dir aislado",
          runner.output_dir == tree.results / "ss_shufle")
    check("B2: cache_dir aislado",
          runner.cache_dir == tree.cache / "ss_shufle")
    check("B3: params_dir aislado",
          runner.params_dir == tree.params / "ss_shufle")
    check("B4: rutas viejas memorizadas",
          runner._old_output_dir == tree.results
          and runner._old_cache_dir == tree.cache
          and runner._old_params_dir == tree.params)
    check("B5: env BASE_RESULTS_PATH exportado",
          os.environ.get("BASE_RESULTS_PATH")
          == str(tree.results / "ss_shufle"))
    check("B6: env BASE_CACHE_PATH exportado",
          os.environ.get("BASE_CACHE_PATH") == str(tree.cache / "ss_shufle"))
    check("B7: env BASE_PARAMS_FILE exportado",
          os.environ.get("BASE_PARAMS_FILE") == str(tree.params / "ss_shufle"))
    check("B8: env LOGGING_BASE_PATH exportado",
          os.environ.get("LOGGING_BASE_PATH")
          == str(tree.logs / "ss_shufle"))
    check("B9: carpetas aisladas creadas",
          (tree.results / "ss_shufle").is_dir()
          and (tree.cache / "ss_shufle").is_dir()
          and (tree.params / "ss_shufle").is_dir())
    check("B10: checkpoint aislado (sufix del usuario intacto)",
          runner._checkpoint_path()
          == tree.cache / "ss_shufle"
          / "batch_checkpoint_cdhsa_ec_ma_taskshuffle.json")
    check("B11: CSV log del batch dentro del arbol aislado",
          str(runner.log_file).startswith(str(tree.results / "ss_shufle")))

    # B12: sin ss_shuffle => rutas de siempre
    runner2 = rb.CDHSABatchRunner(_make_params())
    check("B12: sin ss_shuffle => rutas de siempre",
          runner2.output_dir == tree.results
          and runner2.cache_dir == tree.cache
          and runner2.params_dir == tree.params)


def suite_c_naming(tmp: Path) -> None:
    section("C. Nombres mode_map + contrato de rutas del run v4")

    tree = Tree(tmp / "c")
    tree.apply_env()
    rb = _fresh_modules()
    runner = rb.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    job = runner.all_jobs[0]

    # C1: out_dir del control = ruta EXACTA del run v4 (bajo el arbol nuevo)
    out_dir = runner._get_output_dir(job)
    check("C1: out_dir == ruta v4 real bajo ss_shufle",
          out_dir == tree.results / "ss_shufle" / _EXPECTED_REL,
          str(out_dir))

    # C2: checkpoint key con token ts{seed}
    check("C2: checkpoint key == contrato",
          runner._checkpoint_key(job) == _EXPECTED_KEY,
          runner._checkpoint_key(job))

    # C3: mode_map con token taskshuf
    name = runner._mode_map_json_name(job, 4)
    check("C3: mode_map con token taskshuf",
          name == ("mode_map_4_modes_session1_eyesclosed_mathematic_"
                   "100.0s-200.0s_taskshuf20261022.json"), name)

    # C4: job SIN shuffle => nombre historico sin token
    job_nosh = dict(job)
    ss_nosh = dict(job_nosh["ss_cfg"])
    ss_nosh["task_shuffle"] = {"enabled": False}
    job_nosh["ss_cfg"] = ss_nosh
    name2 = runner._mode_map_json_name(job_nosh, 4)
    check("C4: sin shuffle => nombre historico (sin token)",
          name2 == ("mode_map_4_modes_session1_eyesclosed_mathematic_"
                    "100.0s-200.0s.json"), name2)

    # C5: el comando del pipeline lleva la seed de permutacion
    cmd = runner._build_command(job)
    check("C5: --ss-task-shuffle-seed en el comando",
          "--ss-task-shuffle-seed" in cmd and "20261022" in cmd)


def suite_d_reuse(tmp: Path) -> None:
    section("D. Fase de reuso/migracion (run previo -> arbol aislado)")

    tree = Tree(tmp / "d")
    tree.apply_env()
    rb = _fresh_modules()
    old_out = _seed_old_v4_run(tree, with_mode_map=True)

    runner = rb.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    new_out = runner.output_dir / _EXPECTED_REL
    runner._ss_shuffle_reuse_phase()

    check("D1: run migrado al arbol aislado",
          (new_out / "cdhsa_arrays.npz").is_file())
    check("D2: run movido (ya no esta en el arbol viejo)",
          not (old_out / "cdhsa_arrays.npz").exists())
    key = runner._checkpoint_key(runner.all_jobs[0])
    check("D3: clave del job en el checkpoint aislado",
          key in runner.checkpoint)
    cp_file = runner.cache_dir / \
        "batch_checkpoint_cdhsa_ec_ma_taskshuffle.json"
    check("D4: checkpoint aislado persistido con la clave",
          cp_file.is_file() and key in json.loads(
              cp_file.read_text())["completed"])
    mm = tree.params / "ss_shufle" / (
        "mode_map_4_modes_session1_eyesclosed_mathematic_"
        "100.0s-200.0s_taskshuf20261022.json")
    check("D5: mode_map previo migrado a params aislado", mm.is_file())
    check("D6: mode_map original CONSERVADO en params/ viejo",
          (tree.params / mm.name).is_file())

    # D7: re-run de la fase => nada que migrar (idempotente)
    runner._ss_shuffle_reuse_phase()
    check("D7: fase de reuso idempotente", (new_out / "cdhsa_arrays.npz")
          .is_file())

    # D8: guardia paranoica: dir sin token _taskshuf{seed} NO se migra
    tree2 = Tree(tmp / "d2")
    tree2.apply_env()
    rb2 = _fresh_modules()
    bad_rel = _EXPECTED_REL.replace("_taskshuf20261022", "")
    bad_out = tree2.results / bad_rel
    bad_out.mkdir(parents=True, exist_ok=True)
    (bad_out / "cdhsa_arrays.npz").write_bytes(b"x")
    runner2 = rb2.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    runner2._ss_shuffle_reuse_phase()
    check("D8: dir SIN token taskshuf NO se migra (guardia)",
          (bad_out / "cdhsa_arrays.npz").is_file()
          and not (runner2.output_dir / bad_rel / "cdhsa_arrays.npz")
          .exists())

    # D9: reuse_copy=true conserva el original
    tree3 = Tree(tmp / "d3")
    tree3.apply_env()
    rb3 = _fresh_modules()
    old3 = _seed_old_v4_run(tree3)
    runner3 = rb3.CDHSABatchRunner(
        _add_ss_shuffle(_make_params(), reuse_copy=True))
    runner3._ss_shuffle_reuse_phase()
    new3 = runner3.output_dir / _EXPECTED_REL
    check("D9: reuse_copy=true copia (original conservado)",
          (old3 / "cdhsa_arrays.npz").is_file()
          and (new3 / "cdhsa_arrays.npz").is_file())

    # D10: sin run previo => no migra y no explota
    tree4 = Tree(tmp / "d4")
    tree4.apply_env()
    rb4 = _fresh_modules()
    runner4 = rb4.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    runner4._ss_shuffle_reuse_phase()
    check("D10: sin run previo => nada migrado, sin crash",
          not (runner4.output_dir / _EXPECTED_REL).exists())

    # D11: reuse_existing_run=false => no migra aunque exista
    tree5 = Tree(tmp / "d5")
    tree5.apply_env()
    rb5 = _fresh_modules()
    old5 = _seed_old_v4_run(tree5)
    runner5 = rb5.CDHSABatchRunner(
        _add_ss_shuffle(_make_params(), reuse_existing_run=False))
    runner5._ss_shuffle_reuse_phase()
    check("D11: reuse_existing_run=false no migra",
          (old5 / "cdhsa_arrays.npz").is_file()
          and not (runner5.output_dir / _EXPECTED_REL
                   / "cdhsa_arrays.npz").exists())


def suite_e_filter_todo(tmp: Path) -> None:
    section("E. _filter_todo: red de seguridad (arrays => skip)")

    tree = Tree(tmp / "e")
    tree.apply_env()
    rb = _fresh_modules()

    # E1: control job con arrays en el out_dir (sin checkpoint) => skip
    runner = rb.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    out = runner.output_dir / _EXPECTED_REL
    out.mkdir(parents=True, exist_ok=True)
    (out / "cdhsa_arrays.npz").write_bytes(b"x")
    todo = runner._filter_todo(runner.all_jobs)
    check("E1: arrays en disco => job saltado", len(todo) == 0)
    check("E2: clave auto-sanada en el checkpoint en memoria",
          runner._checkpoint_key(runner.all_jobs[0]) in runner.checkpoint)

    # E3: sin arrays => job pendiente
    tree2 = Tree(tmp / "e2")
    tree2.apply_env()
    rb2 = _fresh_modules()
    runner2 = rb2.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    todo2 = runner2._filter_todo(runner2.all_jobs)
    check("E3: sin arrays => job pendiente", len(todo2) == 1)

    # E4: job NORMAL (sin shuffle) con arrays pero sin checkpoint =>
    #     NO se salta (comportamiento historico intacto)
    tree3 = Tree(tmp / "e3")
    tree3.apply_env()
    rb3 = _fresh_modules()
    p = _make_params()
    p["super_subjects"].pop("task_shuffle")
    runner3 = rb3.CDHSABatchRunner(p)
    out3 = runner3._get_output_dir(runner3.all_jobs[0])
    out3.mkdir(parents=True, exist_ok=True)
    (out3 / "cdhsa_arrays.npz").write_bytes(b"x")
    todo3 = runner3._filter_todo(runner3.all_jobs)
    check("E4: job real con arrays y sin checkpoint NO se salta",
          len(todo3) == 1)


def suite_f_figures(tmp: Path) -> None:
    section("F. Fase de figuras de influencia")

    fake_script = _fake_influence_script(tmp)

    tree = Tree(tmp / "f")
    tree.apply_env()
    rb = _fresh_modules()
    _patch_popen(rb)

    # F1: desactivada => None
    p = _make_params()
    p["super_subjects"].pop("task_shuffle")
    runner = rb.CDHSABatchRunner(p)
    check("F1: fase desactivada => None",
          runner._run_influence_figures() is None)

    # F2: activada + script inexistente => (0,0,1)
    tree2 = Tree(tmp / "f2")
    tree2.apply_env()
    rb2 = _fresh_modules()
    _patch_popen(rb2)
    runner2 = rb2.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    res = runner2._run_influence_figures()
    check("F2: script inexistente => fallo reportado", res == (0, 0, 1),
          str(res))

    # F3: marcador existente => skip sin invocar el script
    tree3 = Tree(tmp / "f3")
    tree3.apply_env()
    rb3 = _fresh_modules()
    _patch_popen(rb3)
    runner3 = rb3.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    runner3.ss_shuffle_cfg["influence_figures_script"] = str(fake_script)
    out3 = runner3.output_dir / _EXPECTED_REL
    out3.mkdir(parents=True, exist_ok=True)
    (out3 / "cdhsa_arrays.npz").write_bytes(b"x")
    (out3 / "channel_influence_taskshuf20261022"
     / "summary_channel_influence.json").parent.mkdir(
        parents=True, exist_ok=True)
    (out3 / "channel_influence_taskshuf20261022"
     / "summary_channel_influence.json").write_text("{}")
    FakePopen.calls = []
    res = runner3._run_influence_figures()
    check("F3: marcador existente => skip (script no invocado)",
          res[1] == 1 and _invoked("analyze_channel_influence") == 0,
          str(res))

    # F4: todo listo => invoca con --results-dir/--mode-map/--top-n
    tree4 = Tree(tmp / "f4")
    tree4.apply_env()
    rb4 = _fresh_modules()
    _patch_popen(rb4)
    runner4 = rb4.CDHSABatchRunner(
        _add_ss_shuffle(_make_params()))
    runner4.ss_shuffle_cfg["influence_figures_script"] = str(fake_script)
    out4 = runner4.output_dir / _EXPECTED_REL
    out4.mkdir(parents=True, exist_ok=True)
    (out4 / "cdhsa_arrays.npz").write_bytes(b"x")
    mm4 = runner4.params_dir / runner4._mode_map_json_name(
        runner4.all_jobs[0], 4)
    mm4.parent.mkdir(parents=True, exist_ok=True)
    mm4.write_text("{}")
    FakePopen.calls = []
    res = runner4._run_influence_figures()
    called = _invoked("analyze_channel_influence")
    cmd = FakePopen.calls[0] if called else []
    check("F4: invoca el script una vez (rc 0 + marcador)",
          res[0] == 1 and called == 1, str(res))
    check("F5: cmd lleva --results-dir / --top-n / --mode-map",
          "--results-dir" in cmd and "--top-n" in cmd
          and "--mode-map" in cmd and "4" in cmd, " ".join(cmd))

    # F6: figures_no_topo => --no-topo
    tree5 = Tree(tmp / "f5")
    tree5.apply_env()
    rb5 = _fresh_modules()
    _patch_popen(rb5)
    runner5 = rb5.CDHSABatchRunner(
        _add_ss_shuffle(_make_params(), figures_no_topo=True))
    runner5.ss_shuffle_cfg["influence_figures_script"] = str(fake_script)
    out5 = runner5.output_dir / _EXPECTED_REL
    out5.mkdir(parents=True, exist_ok=True)
    (out5 / "cdhsa_arrays.npz").write_bytes(b"x")
    mm5 = runner5.params_dir / runner5._mode_map_json_name(
        runner5.all_jobs[0], 4)
    mm5.parent.mkdir(parents=True, exist_ok=True)
    mm5.write_text("{}")
    FakePopen.calls = []
    runner5._run_influence_figures()
    check("F6: figures_no_topo => --no-topo en el cmd",
          FakePopen.calls and "--no-topo" in FakePopen.calls[0])

    # F7: falta mode_map => fallo contado
    tree6 = Tree(tmp / "f6")
    tree6.apply_env()
    rb6 = _fresh_modules()
    _patch_popen(rb6)
    runner6 = rb6.CDHSABatchRunner(
        _add_ss_shuffle(_make_params()))
    runner6.ss_shuffle_cfg["influence_figures_script"] = str(fake_script)
    out6 = runner6.output_dir / _EXPECTED_REL
    out6.mkdir(parents=True, exist_ok=True)
    (out6 / "cdhsa_arrays.npz").write_bytes(b"x")
    res = runner6._run_influence_figures()
    check("F7: falta mode_map => n_fail=1", res == (0, 0, 1), str(res))


def _e2e_common(rb, runner, fake_script):
    runner.ss_shuffle_cfg["influence_figures_script"] = str(fake_script)
    FakePopen.calls = []
    _patch_popen(rb)
    _patch_build_command(runner)
    return runner._run_inner()


def _fake_influence_script(tmp: Path) -> Path:
    """Script fake con el NOMBRE del real (lo reconoce FakePopen)."""
    d = tmp / "fake_scripts"
    d.mkdir(parents=True, exist_ok=True)
    s = d / "analyze_channel_influence_taskshuffle.py"
    s.write_text("# fake\n")
    return s


def suite_g_e2e(tmp: Path) -> None:
    section("G. E2E _run_inner: TODO-EN-UNO (una sola corrida)")

    fake_script = _fake_influence_script(tmp)

    # ---- G1-G4: corrida de CERO (sin run previo) ----
    tree = Tree(tmp / "g1")
    tree.apply_env()
    rb = _fresh_modules()
    runner = rb.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    rc = _e2e_common(rb, runner, fake_script)
    new_out = runner.output_dir / _EXPECTED_REL

    check("G1: rc == 0", rc == 0, str(rc))
    check("G2: train invocado UNA vez (pipeline cdhsa)",
          _invoked("src.pipelines.run_cdhsa") == 1)
    check("G3: extraccion invocada UNA vez",
          _invoked("src.cdhsa.extract_mode_indices") == 1)
    check("G4: figuras invocadas UNA vez",
          _invoked("analyze_channel_influence") == 1)
    check("G5: todo dentro del arbol aislado",
          (new_out / "cdhsa_arrays.npz").is_file()
          and (new_out / "channel_influence_taskshuf20261022"
               / "summary_channel_influence.json").is_file())
    mm = runner.params_dir / runner._mode_map_json_name(
        runner.all_jobs[0], 4)
    check("G6: mode_map en params aislado con token", mm.is_file())
    check("G7: NADA fuera del arbol aislado (results viejo limpio)",
          not (tree.results / "cdhsa").exists())
    cp = runner.cache_dir / "batch_checkpoint_cdhsa_ec_ma_taskshuffle.json"
    check("G8: checkpoint aislado con la clave",
          cp.is_file() and _EXPECTED_KEY in json.loads(
              cp.read_text())["completed"])

    # ---- G9-G12: reuso del run v4 REAL ----
    tree2 = Tree(tmp / "g2")
    tree2.apply_env()
    rb2 = _fresh_modules()
    old_out = _seed_old_v4_run(tree2, with_mode_map=True)
    runner2 = rb2.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    rc2 = _e2e_common(rb2, runner2, fake_script)

    check("G9: rc == 0 (reuso)", rc2 == 0, str(rc2))
    check("G10: train NO invocado (reuso del v4, sin recalcular)",
          _invoked("src.pipelines.run_cdhsa") == 0)
    check("G11: mode_map reusado (migrado) y figuras generadas",
          (runner2.params_dir / "mode_map_4_modes_session1_"
           "eyesclosed_mathematic_100.0s-200.0s_taskshuf20261022.json")
          .is_file()
          and (runner2.output_dir / _EXPECTED_REL
               / "channel_influence_taskshuf20261022"
               / "summary_channel_influence.json").is_file())
    check("G12: el run real v4 YA NO esta en el arbol viejo (movido)",
          not (old_out / "cdhsa_arrays.npz").exists())

    # ---- G13-G14: re-run (idempotencia) ----
    tree3 = Tree(tmp / "g3")
    tree3.apply_env()
    rb3 = _fresh_modules()
    runner3 = rb3.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    rc3 = _e2e_common(rb3, runner3, fake_script)      # corrida 1 (de cero)
    FakePopen.calls = []
    rb3b = _fresh_modules()
    tree3.apply_env()  # re-aplicar: el runner 3 exporto BASE_* aisladas
    runner3b = rb3b.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    rc3b = _e2e_common(rb3b, runner3b, fake_script)    # corrida 2

    check("G13: re-run rc == 0 y SIN train (todo ya hecho)",
          rc3b == 0 and _invoked("src.pipelines.run_cdhsa") == 0)
    check("G14: re-run NO regenera figuras (skip por marcador)",
          _invoked("analyze_channel_influence") == 0)

    # G15: figures_force=true las regenera
    tree4 = Tree(tmp / "g4")
    tree4.apply_env()
    rb4 = _fresh_modules()
    runner4 = rb4.CDHSABatchRunner(_add_ss_shuffle(_make_params()))
    rc4 = _e2e_common(rb4, runner4, fake_script)      # corrida 1
    FakePopen.calls = []
    tree4.apply_env()
    rb4b = _fresh_modules()
    _patch_popen(rb4b)
    p4 = _add_ss_shuffle(_make_params(), figures_force=True)
    runner4b = rb4b.CDHSABatchRunner(p4)
    runner4b.ss_shuffle_cfg["influence_figures_script"] = str(fake_script)
    rc4b = runner4b._run_inner()
    check("G15: figures_force=true regenera las figuras",
          rc4b == 0 and _invoked("analyze_channel_influence") == 1)


def suite_h_historic(tmp: Path) -> None:
    section("H. Sin ss_shuffle: comportamiento historico intacto")

    tree = Tree(tmp / "h")
    tree.apply_env()
    rb = _fresh_modules()
    _patch_popen(rb)

    # JSON del run REAL (sin task_shuffle, sin ss_shuffle)
    p = _make_params()
    p["super_subjects"].pop("task_shuffle")
    p["execution"]["mode_extract_top_n"] = 4
    runner = rb.CDHSABatchRunner(p)
    FakePopen.calls = []
    _patch_build_command(runner)
    rc = runner._run_inner()

    check("H1: rc == 0", rc == 0, str(rc))
    check("H2: train + extract invocados, figuras NO (fase solo control)",
          _invoked("src.pipelines.run_cdhsa") == 1
          and _invoked("src.cdhsa.extract_mode_indices") == 1
          and _invoked("analyze_channel_influence") == 0)
    job = runner.all_jobs[0]
    out = runner._get_output_dir(job)
    check("H3: out_dir del run real SIN token taskshuf y en results/",
          out == tree.results / _EXPECTED_REL.replace("_taskshuf20261022",
                                                       ""),
          str(out))
    check("H4: mode_map real SIN token en params/ viejo",
          (tree.params / "mode_map_4_modes_session1_eyesclosed_mathematic_"
           "100.0s-200.0s.json").is_file())


# ---------------------------------------------------------------------------

def main() -> int:
    global _PASS, _FAIL
    tmp_root = Path(tempfile.mkdtemp(prefix="ss_shuffle_batch_"))
    saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        suite_a_validation(tmp_root)
        suite_b_isolation(tmp_root)
        suite_c_naming(tmp_root)
        suite_d_reuse(tmp_root)
        suite_e_filter_todo(tmp_root)
        suite_f_figures(tmp_root)
        suite_g_e2e(tmp_root)
        suite_h_historic(tmp_root)
    except Exception:
        print("\n[EXCEPTION] durante la suite:")
        traceback.print_exc()
        _FAIL += 1
        _FAILURES.append("exception-durante-suite")
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"RESULTADO: {_PASS} PASS / {_FAIL} FAIL")
    if _FAILURES:
        print("Fallos:")
        for f in _FAILURES:
            print(f"  - {f}")
    print("=" * 70)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
