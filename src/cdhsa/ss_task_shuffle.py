# -*- coding: utf-8 -*-
"""Control negativo: permutacion de tareas al armar los super-sujetos.

Experimento de control (task-shuffle / negative control) para el
pipeline CD-HSA multi-SS. La idea es perturbar EXCLUSIVAMENTE el vinculo
sujeto-condicion en el momento de armar cada super-sujeto, manteniendo
todo lo demas intacto (mismos sujetos, mismas grabaciones, mismas
ventanas, mismo preprocesado, mismos parametros):

* **Permutacion dentro de sujeto (``within_subject``)**: para cada
  super-sujeto ``s`` y cada sujeto miembro ``i``, las ``C`` grabaciones
  (una por tarea) de ese sujeto se reparten AL AZAR entre las ``C``
  celdas ``(s, c)`` del super-sujeto. Con C=2 es un coin-flip por
  sujeto: o mantiene el orden natural o lo intercambia.

  Consecuencias estructurales del diseño:

  - Cada celda sigue teniendo EXACTAMENTE una grabacion por sujeto
    miembro (los tamanos de celda no cambian; las duraciones pueden
    variar levemente si alguna grabacion se salta por duracion).
  - Cada grabacion se usa EXACTAMENTE una vez (no hay duplicados ni
    datos sin usar): el consumo total de datos es identico al run real.
  - Cada celda termina siendo una MEZCLA ALEATORIA de tareas
    (~Binomial(n_miembros, 1/C) por tarea): el "contraste entre
    condiciones" del pipeline queda reducido a ruido de mezcla.
  - Los MISMOS sujetos aparecen en las C celdas de cada SS (el
    contraste sigue emparejado por sujeto): lo unico destruido es la
    informacion de que tarea es cada grabacion.

Que deberia morir y que deberia sobrevivir (interpretacion previa al
experimento; ver README_TASK_SHUFFLE.md):

* **Step D (modos especificos)**: debe dar vacio o ruido puro. La
  regla Haar adaptativa es liberal y puede retener algunos modos, pero
  el contraste de prevalencia debe caer a ~0 y el LOSO debe quedar
  <= 0 (sin replicacion). Si el LOSO sale POSITIVO o los modos
  replican, el metodo esta forzando estructura donde no la hay: falla
  del control.
* **Step B (tests F de energia por modo comun)**: con celdas
  mezcladas, el contraste ec-vs-ma por modo se promedia hacia cero:
  F ~ 1. Si algunos modos siguen con F significativa, la "senal" de
  redistribucion de energia del run real era en parte artefacto de
  composicion (mala noticia para la clasificacion por energias).
* **A1-A5 / A6 (backbone)**: los modos comunes NO dependen de las
  etiquetas de tarea (el backbone es compartido entre tareas), asi que
  r0 y el espectro lambda deberian quedar casi identicos al run real.
  Cuantificar: angulos principales entre W0 real y W0 shuffeado. Si
  se distorsiona, el "comun" era parcialmente un efecto de alinear
  mismas tareas.

Determinismo: la asignacion es funcion de (seed, S, member_ids, tasks)
y se registra completa en ``hankel_info.json`` (claves
``task_shuffle_seed`` y ``task_shuffle``) para trazabilidad y para que
el auto-descubrimiento de la clasificacion session2 pueda EXCLUIR los
trains permutados (control negativo, no modelo).

"""
from __future__ import annotations

import numpy as np

__all__ = [
    "build_task_shuffle_assignment",
    "member_tasks_for_cell",
    "summarize_task_mix",
    "validate_task_shuffle_seed",
]


def build_task_shuffle_assignment(
    ss_member_ids: dict[int, list[int]],
    tasks: list[str],
    seed: int,
) -> dict[int, dict[int, list[str]]]:
    """Construir la permutacion de tareas por (SS, sujeto).

    Parameters
    ----------
    ss_member_ids : dict[int, list[int]]
        ``{ss_id (1-indexed): [subject_ids]}`` de cada super-sujeto.
        Se recorre en orden de ss_id y, dentro, en el orden de la lista
        de miembros: la secuencia de llamadas al RNG es determinista
        dado (seed, esta estructura).
    tasks : list[str]
        Las C tareas del experimento (el orden FIJA los indices de
        celda: la celda c pide nominalmente tasks[c]).
    seed : int
        Semilla del generador (``np.random.default_rng``).

    Returns
    -------
    assignment : dict[int, dict[int, list[str]]]
        ``assignment[ss_id][subject_id]`` = lista de C tareas donde la
        posicion ``c`` es la tarea REAL cuya grabacion de ese sujeto
        entra en la celda ``(s, c)``. Cada lista es una permutacion de
        ``tasks`` (sin repeticiones): todas las grabaciones del sujeto
        se usan, cada una en una celda distinta.

    Raises
    ------
    ValueError
        Si hay menos de 2 tareas (la permutacion seria identidad), si
        ``tasks`` tiene duplicados, o si un sujeto aparece en mas de
        un SS (cada grabacion debe usarse exactamente una vez).
    """
    tasks = list(tasks)
    if len(tasks) < 2:
        raise ValueError(
            "task_shuffle requiere >= 2 tareas (con 1 tarea la "
            "permutacion es la identidad y no hay control que hacer)."
        )
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"tasks con duplicados: {tasks}")

    # Un sujeto no puede estar en dos SS: cada grabacion debe usarse
    # exactamente una vez en todo el experimento.
    seen: set[int] = set()
    for sid, members in ss_member_ids.items():
        for subj in members:
            if subj in seen:
                raise ValueError(
                    f"El sujeto {subj} aparece en mas de un "
                    f"super-sujeto (p.ej. SS {sid}): el task_shuffle "
                    f"requiere particiones disjuntas."
                )
            seen.add(subj)

    rng = np.random.default_rng(seed)
    assignment: dict[int, dict[int, list[str]]] = {}
    for ss_id in sorted(ss_member_ids):
        assignment[ss_id] = {}
        for subj in ss_member_ids[ss_id]:
            perm = rng.permutation(len(tasks))
            assignment[ss_id][subj] = [tasks[int(i)] for i in perm]
    return assignment


def member_tasks_for_cell(
    assignment: dict[int, dict[int, list[str]]],
    ss_id: int,
    member_ids: list[int],
    c_idx: int,
) -> list[str]:
    """Tarea REAL de cada miembro para la celda ``(ss_id, c_idx)``.

    Orden alineado con ``member_ids`` (la lista que el loader recibe
    como ``subject_ids``): ``member_tasks[i]`` es la tarea que se le
    pide al sujeto ``member_ids[i]``.
    """
    per_ss = assignment[ss_id]
    out = []
    for subj in member_ids:
        if subj not in per_ss:
            raise KeyError(
                f"Sujeto {subj} no esta en la asignacion del SS {ss_id}."
            )
        out.append(per_ss[subj][c_idx])
    return out


def summarize_task_mix(member_tasks: list[str]) -> str:
    """Resumen compacto de la mezcla de tareas de una celda.

    Ejemplo: ``"ec x5, ma x7"`` (para nombres largos usa iniciales
    simples: cuenta por tarea separada por comas).
    """
    if not member_tasks:
        return "(vacia)"
    counts: dict[str, int] = {}
    for t in member_tasks:
        counts[t] = counts.get(t, 0) + 1
    return ", ".join(f"{t} x{n}" for t, n in sorted(counts.items()))


def validate_task_shuffle_seed(seed: int | None) -> int | None:
    """Normalizar/validar la semilla leida de la config o el CLI."""
    if seed is None:
        return None
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"La semilla de task_shuffle debe ser >= 0 (got {seed}).")
    return seed
