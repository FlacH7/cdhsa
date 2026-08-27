"""
Memory Tracker for EEG Pipeline Profiling
===========================================

Modo de uso rapido para todo el pipeline:

    # === En tu script principal (el que orquesta todo) ===
    from memory_tracker import MemoryMonitor, profile_memory

    # 1. Arrancar el monitor en background al inicio del pipeline
    monitor = MemoryMonitor(
        interval_sec=0.1,      # samplear cada 100ms
        logger=logger,          # tu logger existente
        log_spikes=True,        # loguear cuando hay saltos bruscos
        spike_threshold_mb=100, # loguear si sube >100MB entre samples
    )
    monitor.start()

    # ... tu pipeline completo corre aqui ...
    run_subject(subject_1, task_a)
    run_subject(subject_1, task_b)
    run_subject(subject_2, task_a)
    # ...

    # 2. Al final del pipeline: reporte completo
    monitor.stop()
    report = monitor.report()
    print(report)
    monitor.save_timeline_csv('/path/to/memory_timeline.csv')

=== Usar como decorador en funciones especificas ===

    @profile_memory            # usa el nombre de la funcion como label
    def _build_multivariate_hankel(X, T):
        ...

    @profile_memory('SVD truncado', show_top_allocs=5)
    def run_svd(H, k):
        ...

=== Mezclar: background + checkpoints manuales ===

    monitor = MemoryMonitor(logger=logger, interval_sec=0.05)
    monitor.start()

    monitor.checkpoint('pipeline start')
    ...
    monitor.checkpoint('after filtering')
    ...
    monitor.checkpoint('after hankel')
    ...

    monitor.stop()
    monitor.summary()   # tabla de checkpoints
    monitor.report()    # timeline del background + picos

El reporte final incluye:
- Timeline de memoria (RSS + Python alloc) muestreada continuamente
- Momento y valor del pico maximo
- Deteccion de saltos bruscos (spikes)
- Si se usaron checkpoints, se correlacionan con la timeline
- Tabla de top allocations por archivo/linea (tracemalloc)

Dependencies
------------
- ``psutil``  (RSS a nivel proceso)
- ``tracemalloc`` (stdlib, allocations a nivel Python)
"""

from __future__ import annotations

import atexit
import csv
import functools
import gc
import logging
import os
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class MemorySample:
    t: float
    wall: float
    rss_mb: float
    alloc_mb: float


@dataclass
class Checkpoint:
    label: str
    t: float
    wall: float
    rss_mb: float
    alloc_mb: float


@dataclass
class MemorySpike:
    t: float
    delta_rss_mb: float
    rss_before_mb: float
    rss_after_mb: float


# ===================================================================
# MemoryMonitor
# ===================================================================

class MemoryMonitor:
    """
    Monitor de memoria en background que muestrea continuamente
    mientras el pipeline ejecuta, con soporte para checkpoints
    manuales y generacion de reporte al final.
    """

    def __init__(
        self,
        interval_sec: float = 0.1,
        logger=None,
        log_level: int = logging.INFO,
        gc_before_sample: bool = False,
        log_spikes: bool = True,
        spike_threshold_mb: float = 50.0,
        tracemalloc_start: bool = True,
    ):
        if logger is None:
            self._logger = logging.getLogger('mem_monitor')
        elif isinstance(logger, str):
            self._logger = logging.getLogger(logger)
        else:
            self._logger = logger

        self._log_level = log_level
        self._interval = interval_sec
        self._gc_before = gc_before_sample
        self._log_spikes = log_spikes
        self._spike_threshold = spike_threshold_mb

        self._t0 = 0.0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._timeline = []
        self._checkpoints = []
        self._spikes = []
        self._tracemalloc_started_here = False
        self._tracemalloc_start_snap = None
        self._registered_atexit = False

    # ---- internal measurements ----

    def _read_rss_mb(self) -> float:
        if HAS_PSUTIL:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        try:
            with open('/proc/%d/status' % os.getpid()) as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return float(line.split()[1]) / 1024
        except (FileNotFoundError, ValueError):
            pass
        return -1.0

    def _read_alloc_mb(self) -> float:
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        return -1.0

    def _elapsed(self) -> float:
        if self._t0 == 0:
            return 0.0
        return time.time() - self._t0

    # ---- background thread loop ----

    def _background_loop(self):
        last_rss = self._read_rss_mb()
        while self._running:
            if self._gc_before:
                gc.collect()

            rss = self._read_rss_mb()
            alloc = self._read_alloc_mb()
            t = self._elapsed()
            wall = time.time()

            sample = MemorySample(t=t, wall=wall, rss_mb=rss, alloc_mb=alloc)

            with self._lock:
                self._timeline.append(sample)

                if (self._log_spikes and last_rss > 0 and rss > 0
                        and rss - last_rss >= self._spike_threshold):
                    spike = MemorySpike(
                        t=t,
                        delta_rss_mb=rss - last_rss,
                        rss_before_mb=last_rss,
                        rss_after_mb=rss,
                    )
                    self._spikes.append(spike)
                    self._logger.log(
                        self._log_level,
                        f'[MEM] SPIKE at t={t:.2f}s  RSS {last_rss:,.0f} -> {rss:,.0f} MB  (delta={rss - last_rss:+,.0f} MB)',
                    )

                last_rss = rss

            time.sleep(self._interval)

    # ---- public API: start / stop ----

    def start(self):
        if self._running:
            self._logger.warning('[MEM] Monitor ya esta corriendo.')
            return

        if not tracemalloc.is_tracing():
            tracemalloc.start(25)
            self._tracemalloc_started_here = True
        self._tracemalloc_start_snap = tracemalloc.take_snapshot()

        self._timeline.clear()
        self._checkpoints.clear()
        self._spikes.clear()

        self._t0 = time.time()
        self._running = True

        self._thread = threading.Thread(
            target=self._background_loop,
            name='MemoryMonitor',
            daemon=True,
        )
        self._thread.start()

        if not self._registered_atexit:
            atexit.register(self.stop)
            self._registered_atexit = True

        self._logger.log(
            self._log_level,
            '[MEM] Monitor arrancado (interval=%.3fs, psutil=%s, tracemalloc=%s)',
            self._interval, HAS_PSUTIL, tracemalloc.is_tracing(),
        )

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._logger.log(
            self._log_level,
            '[MEM] Monitor detenido. %d muestras, %d checkpoints, %d spikes.',
            len(self._timeline), len(self._checkpoints), len(self._spikes),
        )

    # ---- public API: checkpoints ----

    def checkpoint(self, label: str):
        gc.collect()
        rss = self._read_rss_mb()
        alloc = self._read_alloc_mb()
        t = self._elapsed()
        wall = time.time()

        cp = Checkpoint(label=label, t=t, wall=wall, rss_mb=rss, alloc_mb=alloc)

        with self._lock:
            self._checkpoints.append(cp)

            if len(self._checkpoints) >= 2:
                prev = self._checkpoints[-2]
                dr = rss - prev.rss_mb
                da = alloc - prev.alloc_mb
                self._logger.log(
                    self._log_level,
                    f'[MEM] CHECKPOINT "{label}" t={t:.2f}s  '
                    f'RSS={rss:,.1f} MB  Alloc={alloc:,.1f} MB  '
                    f'delta_RSS={dr:+,.1f} MB  delta_alloc={da:+,.1f} MB',
                )
            else:
                self._logger.log(
                    self._log_level,
                    f'[MEM] CHECKPOINT (baseline) "{label}" t={t:.2f}s  RSS={rss:,.1f} MB  Alloc={alloc:,.1f} MB',
                )

    # ---- public API: reports ----

    def summary(self) -> str:
        cps = self._checkpoints
        if len(cps) < 2:
            return '[MEM] Menos de 2 checkpoints registrados.'

        lines = []
        lines.append('=== Memory Checkpoints Summary ===')
        hdr = '  %-40s %7s %10s %11s %11s %12s' % (
            'Step', 't (s)', 'RSS (MB)', 'Alloc (MB)', 'Delta_RSS', 'Delta_Alloc')
        lines.append(hdr)
        lines.append('  ' + '-' * 97)

        deltas = []
        for i, cp in enumerate(cps):
            if i == 0:
                dr, da = 0.0, 0.0
            else:
                dr = cp.rss_mb - cps[i - 1].rss_mb
                da = cp.alloc_mb - cps[i - 1].alloc_mb
            deltas.append((dr, da))

        sorted_d = sorted(deltas, key=lambda x: x[0], reverse=True)
        if len(sorted_d) >= 3:
            top3_thresh = sorted_d[2][0]
        else:
            top3_thresh = float('inf')

        for i, cp in enumerate(cps):
            dr, da = deltas[i]
            hot = dr >= top3_thresh and dr > 0
            marker = ' <<<' if hot else ''
            row = ('  %-40s %7.2f %10s %11s %+11s %+12s%s'
                   % (cp.label, cp.t,
                      f'{cp.rss_mb:,.1f}', f'{cp.alloc_mb:,.1f}',
                      f'{dr:,.1f}', f'{da:,.1f}', marker))
            lines.append(row)

        lines.append('  ' + '-' * 97)

        for line in lines:
            self._logger.log(self._log_level, '[MEM] %s', line)

        return '\n'.join(lines)

    def report(self) -> str:
        with self._lock:
            timeline = list(self._timeline)
            checkpoints = list(self._checkpoints)
            spikes = list(self._spikes)

        if not timeline:
            return '[MEM] No hay datos de timeline. start() fue llamado?'

        rss_vals = [s.rss_mb for s in timeline if s.rss_mb > 0]
        alloc_vals = [s.alloc_mb for s in timeline if s.alloc_mb > 0]

        total_time = timeline[-1].t if timeline else 0.0
        peak_rss = max(rss_vals) if rss_vals else 0.0
        peak_alloc = max(alloc_vals) if alloc_vals else 0.0
        avg_rss = sum(rss_vals) / len(rss_vals) if rss_vals else 0.0

        peak_sample = max(timeline, key=lambda s: s.rss_mb if s.rss_mb > 0 else -1)

        peak_cp_label = ''
        if checkpoints:
            nearest = min(checkpoints, key=lambda c: abs(c.t - peak_sample.t))
            if abs(nearest.t - peak_sample.t) < 2.0:
                peak_cp_label = f' (cerca de checkpoint: {nearest.label})'

        baseline_rss = rss_vals[0] if rss_vals else 0.0
        final_rss = rss_vals[-1] if rss_vals else 0.0
        net_growth = final_rss - baseline_rss

        lines = []
        lines.append('')
        lines.append('=' * 70)
        lines.append('  MEMORY MONITOR REPORT')
        lines.append('=' * 70)
        lines.append(f'  Tiempo total monitoreado : {total_time:.1f} s')
        lines.append(f'  Muestras tomadas        : {len(timeline)}')
        lines.append(f'  Checkpoints registrados  : {len(checkpoints)}')
        lines.append(f'  Spikes detectados        : {len(spikes)}')
        lines.append('')
        lines.append('--- RSS (Resident Set Size) ---')
        lines.append(f'  Baseline  : {baseline_rss:>10,.1f} MB')
        lines.append(f'  Peak      : {peak_rss:>10,.1f} MB  at t={peak_sample.t:.2f}s{peak_cp_label}')
        lines.append(f'  Final     : {final_rss:>10,.1f} MB')
        lines.append(f'  Average   : {avg_rss:>10,.1f} MB')
        lines.append(f'  Net growth: {net_growth:>+10,.1f} MB')
        lines.append('')

        if alloc_vals:
            lines.append('--- Python Allocator (tracemalloc) ---')
            lines.append(f'  Peak  : {peak_alloc:>10,.1f} MB')
            lines.append(f'  Final : {alloc_vals[-1]:>10,.1f} MB')
            lines.append('')

        if spikes:
            lines.append('--- Top Memory Spikes ---')
            for sp in sorted(spikes, key=lambda s: s.delta_rss_mb, reverse=True)[:10]:
                cp_label = ''
                if checkpoints:
                    nearest = min(checkpoints, key=lambda c: abs(c.t - sp.t))
                    if abs(nearest.t - sp.t) < 1.0:
                        cp_label = f'  <-- cerca de: {nearest.label}'
                lines.append(
                    f'  t={sp.t:>7.2f}s  RSS {sp.rss_before_mb:,.0f} -> {sp.rss_after_mb:,.0f} MB'
                    f'  delta={sp.delta_rss_mb:+,.0f} MB{cp_label}')
            lines.append('')

        if checkpoints and timeline:
            lines.append('--- Checkpoints vs Timeline ---')
            for cp in checkpoints:
                lines.append(
                    f'  t={cp.t:>7.2f}s  "{cp.label}"'
                    f'  RSS={cp.rss_mb:,.1f} MB  Alloc={cp.alloc_mb:,.1f} MB')
            lines.append('')

        lines.append('=' * 70)
        report_str = '\n'.join(lines)
        for line in lines:
            self._logger.log(self._log_level, '[MEM] %s', line)

        return report_str

    def top_allocations(self, n: int = 15) -> str:
        if self._tracemalloc_start_snap is None:
            return '[MEM] No hay snapshot inicial de tracemalloc.'
        if not tracemalloc.is_tracing():
            return '[MEM] tracemalloc no esta activo.'

        snap_now = tracemalloc.take_snapshot()
        stats = snap_now.compare_to(self._tracemalloc_start_snap, 'lineno')

        lines = [f'=== Top {n} Allocations (desde inicio del monitor) ===']
        for stat in stats[:n]:
            lines.append(
                f'  {stat.traceback}: {stat.size / (1024 * 1024):,.1f} MB ({stat.count} blocks)')

        result = '\n'.join(lines)
        for line in lines:
            self._logger.log(self._log_level, '[MEM] %s', line)
        return result

    def save_timeline_csv(self, path: str):
        with self._lock:
            timeline = list(self._timeline)
            checkpoints = list(self._checkpoints)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t_sec', 'wall_time', 'rss_mb', 'alloc_mb', 'checkpoint_label'])
            for s in timeline:
                cp_label = ''
                for cp in checkpoints:
                    if abs(cp.t - s.t) < self._interval:
                        cp_label = cp.label
                        break
                writer.writerow([
                    '%.4f' % s.t, '%.4f' % s.wall, '%.2f' % s.rss_mb,
                    '%.2f' % s.alloc_mb, cp_label,
                ])

        self._logger.log(
            self._log_level,
            '[MEM] Timeline guardada en: %s (%d filas)',
            path, len(timeline),
        )

    # ---- properties ----

    @property
    def current_rss_mb(self) -> float:
        return self._read_rss_mb()

    @property
    def current_alloc_mb(self) -> float:
        return self._read_alloc_mb()

    @property
    def peak_rss_mb(self) -> float:
        with self._lock:
            valid = [s.rss_mb for s in self._timeline if s.rss_mb > 0]
            return max(valid) if valid else 0.0

    @property
    def elapsed_sec(self) -> float:
        return self._elapsed()

    @property
    def is_running(self) -> bool:
        return self._running

    # ---- context manager ----

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
        if self._tracemalloc_started_here and tracemalloc.is_tracing():
            tracemalloc.stop()


# ===================================================================
# Global singleton
# ===================================================================

_global_monitor = None


def get_global_monitor(interval_sec=0.1, logger=None, **kwargs):
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(
            interval_sec=interval_sec, logger=logger, **kwargs)
    return _global_monitor


# ===================================================================
# Decorador @profile_memory
# ===================================================================

def profile_memory(func_or_label=None, *, logger=None, log_level=logging.INFO,
                   show_top_allocs=0, use_monitor=False):
    """
    Decorador para perfilar memoria de una funcion.

    Tres formas de uso::

        @profile_memory
        def build_hankel(X, T):
            ...

        @profile_memory('hankel construction')
        def build_hankel(X, T):
            ...

        @profile_memory(show_top_allocs=5)
        def build_hankel(X, T):
            ...
    """
    def _decorate(func):
        label = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if use_monitor:
                mon = get_global_monitor(logger=logger)
                mon.checkpoint('>>> %s START' % label)
                result = func(*args, **kwargs)
                mon.checkpoint('<<< %s END' % label)
                return result

            _lg = _resolve_logger(logger)
            gc.collect()

            tm_started = False
            if not tracemalloc.is_tracing():
                tracemalloc.start(25)
                tm_started = True
            snap_before = tracemalloc.take_snapshot()

            rss_before = -1.0
            if HAS_PSUTIL:
                rss_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            t0 = time.time()
            _lg.log(log_level, f'[MEM] >>> {label}  RSS={rss_before:,.1f} MB')

            try:
                result = func(*args, **kwargs)
            finally:
                gc.collect()
                elapsed = time.time() - t0
                snap_after = tracemalloc.take_snapshot()

                rss_after = -1.0
                if HAS_PSUTIL:
                    rss_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                delta = rss_after - rss_before if rss_before > 0 else 0.0

                _lg.log(log_level,
                         f'[MEM] <<< {label}  RSS={rss_after:,.1f} MB  delta={delta:+,.1f} MB  elapsed={elapsed:.2f}s')

                if show_top_allocs > 0:
                    stats = snap_after.compare_to(snap_before, 'lineno')
                    for stat in stats[:show_top_allocs]:
                        _lg.log(log_level,
                                 f'[MEM]     {stat.traceback}: {stat.size / (1024 * 1024):,.1f} MB ({stat.count} blocks)')

                if tm_started:
                    tracemalloc.stop()

            return result

        return wrapper

    # --- resolver los 3 modos de llamada ---
    if callable(func_or_label):
        return _decorate(func_or_label)
    elif isinstance(func_or_label, str):
        def _label_decorator(func):
            label = func_or_label

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if use_monitor:
                    mon = get_global_monitor(logger=logger)
                    mon.checkpoint('>>> %s START' % label)
                    result = func(*args, **kwargs)
                    mon.checkpoint('<<< %s END' % label)
                    return result

                _lg = _resolve_logger(logger)
                gc.collect()
                tm_started = False
                if not tracemalloc.is_tracing():
                    tracemalloc.start(25)
                    tm_started = True
                snap_before = tracemalloc.take_snapshot()
                rss_before = -1.0
                if HAS_PSUTIL:
                    rss_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                t0 = time.time()
                _lg.log(log_level, f'[MEM] >>> {label}  RSS={rss_before:,.1f} MB')
                try:
                    result = func(*args, **kwargs)
                finally:
                    gc.collect()
                    elapsed = time.time() - t0
                    snap_after = tracemalloc.take_snapshot()
                    rss_after = -1.0
                    if HAS_PSUTIL:
                        rss_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    delta = rss_after - rss_before if rss_before > 0 else 0.0
                    _lg.log(log_level,
                             f'[MEM] <<< {label}  RSS={rss_after:,.1f} MB  delta={delta:+,.1f} MB  elapsed={elapsed:.2f}s')
                    if show_top_allocs > 0:
                        stats = snap_after.compare_to(snap_before, 'lineno')
                        for stat in stats[:show_top_allocs]:
                            _lg.log(log_level,
                                     f'[MEM]     {stat.traceback}: {stat.size / (1024 * 1024):,.1f} MB ({stat.count} blocks)')
                    if tm_started:
                        tracemalloc.stop()
                return result
            return wrapper
        return _label_decorator
    else:
        return _decorate


def _resolve_logger(logger):
    if logger is None:
        return logging.getLogger('mem_monitor')
    elif isinstance(logger, str):
        return logging.getLogger(logger)
    return logger


# ===================================================================
# Legacy API aliases
# ===================================================================

MemoryTracker = MemoryMonitor
track_memory_decorator = profile_memory


def track_memory(label, logger=None, log_level=logging.INFO,
                 gc_before=True, show_top_allocs=0):
    return profile_memory(label, logger=logger, log_level=log_level,
                          show_top_allocs=show_top_allocs)


def log_memory_now(label, logger=None, log_level=logging.INFO):
    gc.collect()
    _lg = _resolve_logger(logger)
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) if HAS_PSUTIL else -1.0
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)
    alloc = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
    _lg.log(log_level, f'[MEM] "{label}"  RSS={rss:,.1f} MB  Python_alloc={alloc:,.1f} MB')
    global _global_monitor
    if _global_monitor is not None and _global_monitor.is_running:
        _global_monitor.checkpoint(label)
    return {'rss_mb': rss, 'alloc_mb': alloc, 'label': label}
