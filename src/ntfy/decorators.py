# -*- coding: utf-8 -*-
"""Decoradores y context manager para notificar el estado de funciones largas.

Casos de uso tipicos (corridas por SSH + ``nohup``)::

    from src.ntfy import notify_on_critical_error, notify_success

    # 1) Avisar si una excepcion critica mata la ejecucion (y relanzarla
    #    para que el proceso muera como corresponde):
    @notify_on_critical_error(channel=NTFY_CHANNEL, title="[proyecto] ERROR")
    def pipeline(**cfg):
        ...

    # 2) Avisar cuando una funcion larga termina bien:
    @notify_on_success(channel=NTFY_CHANNEL)
    def entrenar_modelo(**cfg):
        ...

    # 3) Monitor completo (inicio + exito + error):
    @notify_calls(channel=NTFY_CHANNEL, notify_start=True)
    def experimento(**cfg):
        ...

    # 4) Bloques sueltos:
    with watch("A6 common-rank (SS1)"):
        correr_a6()

Filosofia: una notificacion JAMAS debe alterar el comportamiento del codigo
que monitoriza.  Los decoradores solo observan; si el propio envio falla, se
registra un warning y se sigue adelante (salvo que la funcion decorada falle,
en cuyo caso la excepcion original se relanza intacta).
"""

from __future__ import annotations

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar

from .client import NtfyClient, get_default_client

__all__ = [
    "notify_on_success",
    "notify_on_critical_error",
    "notify_on_error",
    "notify_calls",
    "watch",
    "format_duration",
    "format_exception",
]

LOGGER = logging.getLogger("ntfy.decorators")

#: Lineas del traceback incluidas en la notificacion de error.
TRACEBACK_TAIL_LINES = 12

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Formatea segundos como ``'1h 04m 30s'`` / ``'4m 12s'`` / ``'37s'``."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_exception(exc: BaseException, tail: int = TRACEBACK_TAIL_LINES) -> str:
    """Traza el traceback de ``exc`` recortada a las ultimas ``tail`` lineas.

    Si se recorta, se conserva siempre la primera linea (``Traceback ...``)
    para que el formato siga reconociendose en el telefono.
    """
    chunks = traceback.format_exception(type(exc), exc, exc.__traceback__)
    lines = [ln for chunk in chunks for ln in chunk.splitlines()]
    if len(lines) > tail:
        header = [lines[0]] if lines[0].startswith("Traceback") else []
        lines = header + ["    [...] traceback recortado"] + lines[-tail:]
    return "\n".join(lines)


def _label(func: Callable) -> str:
    """Etiqueta corta de una funcion: ``modulo.Nombre`` (o ``clase.metodo``)."""
    module = getattr(func, "__module__", "") or ""
    module = module.rsplit(".", 1)[-1]
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", "funcion")
    return f"{module}.{name}" if module else name


def _client_for(client: Optional[NtfyClient]) -> NtfyClient:
    return client if client is not None else get_default_client()


def _safe(fn: Callable[[], Any]) -> None:
    """Ejecuta ``fn`` tragandose cualquier error (la notificacion no rompe nada)."""
    try:
        fn()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("[ntfy] fallo enviando la notificacion: %s", exc)


def _decorator_apply(decorator: Callable[[F], F], obj: Any) -> Any:
    """Permite usar el decorador con o sin parentesis."""
    return decorator(obj) if obj is not None else decorator


def _merge_kwargs(base: Dict[str, Any], **defaults: Any) -> Dict[str, Any]:
    """``dict(base)`` donde ``defaults`` solo aplica si la clave no existe."""
    merged = dict(defaults)
    merged.update(base)
    return merged


# ---------------------------------------------------------------------------
# notify_on_critical_error
# ---------------------------------------------------------------------------

def notify_on_critical_error(
    _func: Optional[F] = None,
    *,
    channel: Optional[str] = None,
    client: Optional[NtfyClient] = None,
    title: str = "ERROR critico",
    include_traceback: bool = True,
    re_raise: bool = True,
    notify_start: bool = False,
    catch_system_exit: bool = False,
    priority: str = "urgent",
    tags: Any = ("rotating_light",),
    **send_kwargs: Any,
):
    """Decora una funcion para avisar (urgent) si una excepcion critica escapa.

    La notificacion incluye el tipo y mensaje de la excepcion, el tiempo que
    llevaba la ejecucion y (por defecto) las ultimas lineas del traceback.
    Despues de notificar, la excepcion se RELANZA para que la ejecucion se
    detenga exactamente igual que sin el decorador: notificar no cura nada.

    Se puede usar con o sin parentesis::

        @notify_on_critical_error
        def f(): ...

        @notify_on_critical_error(channel="mi-canal", title="[X] error")
        def f(): ...

    Parametros
    ----------
    channel:
        Canal donde publicar (default: el del cliente por defecto, que lee
        ``$NTFY_CHANNEL``).
    client:
        :class:`~src.ntfy.client.NtfyClient` explicito (default: el del
        proceso).
    re_raise:
        Relanzar la excepcion despues de notificar (default ``True``).
        ``False` solo si sabes muy bien por que: tragar la excepcion puede
        dejar la corrida en estado inconsistente.
    catch_system_exit:
        Notificar tambien los ``SystemExit`` con codigo distinto de 0 (p. ej.
        ``sys.exit(1)`` por parametros invalidos).  Los ``SystemExit`` con
        codigo 0 (``--help``) nunca generan notificacion.  Default ``False``.
    notify_start:
        Enviar ademas un aviso informativo al entrar en la funcion.
    """

    def decorator(func: F) -> F:
        err_kw = _merge_kwargs(
            send_kwargs, channel=channel, title=title,
            priority=priority, tags=tags,
        )
        start_kw = _merge_kwargs(send_kwargs, channel=channel, title="Iniciando")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ntfy_client = _client_for(client)
            if notify_start:
                _safe(lambda: ntfy_client.info(
                    f"{_label(func)} iniciando...", **start_kw,
                ))
            t0 = time.monotonic()
            try:
                return func(*args, **kwargs)
            except SystemExit as exc:
                if catch_system_exit and exc.code not in (None, 0):
                    _safe(lambda: ntfy_client.error(
                        f"{_label(func)} termino con sys.exit({exc.code!r}) tras "
                        f"{format_duration(time.monotonic() - t0)}.\n"
                        "La ejecucion se detuvo antes de completarse "
                        "(revisa el log: parametros/JSON/config).",
                        **err_kw,
                    ))
                raise
            except Exception as exc:
                elapsed = format_duration(time.monotonic() - t0)
                parts = [
                    f"{_label(func)} fallo tras {elapsed}.",
                    f"Excepcion: {type(exc).__name__}: {exc}",
                ]
                if include_traceback:
                    parts += ["", format_exception(exc)]
                _safe(lambda: ntfy_client.error("\n".join(parts), **err_kw))
                if re_raise:
                    raise
                return None

        return wrapper  # type: ignore[return-value]

    return _decorator_apply(decorator, _func)


#: Alias corto.
notify_on_error = notify_on_critical_error


# ---------------------------------------------------------------------------
# notify_on_success
# ---------------------------------------------------------------------------

def notify_on_success(
    _func: Optional[F] = None,
    *,
    channel: Optional[str] = None,
    client: Optional[NtfyClient] = None,
    title: str = "Completado",
    include_duration: bool = True,
    send_result: bool = False,
    notify_start: bool = False,
    priority: str = "high",
    tags: Any = ("white_check_mark", "tada"),
    **send_kwargs: Any,
):
    """Decora una funcion para avisar cuando termina sin errores.

    La notificacion incluye la duracion (por defecto) y opcionalmente el
    ``repr()`` del resultado (``send_result=True``, recortado a 300 chars).
    Cualquier excepcion se propaga intacta: este decorador NO notifica
    errores (para eso esta :func:`notify_on_critical_error` o
    :func:`notify_calls`).
    """

    def decorator(func: F) -> F:
        ok_kw = _merge_kwargs(
            send_kwargs, channel=channel, title=title,
            priority=priority, tags=tags,
        )
        start_kw = _merge_kwargs(send_kwargs, channel=channel, title="Iniciando")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ntfy_client = _client_for(client)
            if notify_start:
                _safe(lambda: ntfy_client.info(
                    f"{_label(func)} iniciando...", **start_kw,
                ))
            t0 = time.monotonic()
            result = func(*args, **kwargs)
            message = f"{_label(func)} termino OK"
            if include_duration:
                message += f" en {format_duration(time.monotonic() - t0)}"
            if send_result:
                message += f"\nResultado: {repr(result)[:300]}"
            _safe(lambda: ntfy_client.success(message, **ok_kw))
            return result

        return wrapper  # type: ignore[return-value]

    return _decorator_apply(decorator, _func)


# ---------------------------------------------------------------------------
# notify_calls: inicio + exito + error en un solo decorador
# ---------------------------------------------------------------------------

def notify_calls(
    _func: Optional[F] = None,
    *,
    channel: Optional[str] = None,
    client: Optional[NtfyClient] = None,
    title: str = "Monitor",
    notify_start: bool = False,
    on_success: bool = True,
    on_error: bool = True,
    include_duration: bool = True,
    include_traceback: bool = True,
    re_raise: bool = True,
    success_priority: str = "high",
    error_priority: str = "urgent",
    **send_kwargs: Any,
):
    """Monitoriza una funcion: aviso de inicio opcional + exito + error critico.

    Union de :func:`notify_on_success` y :func:`notify_on_critical_error`::

        @notify_calls(channel="mi-canal", notify_start=True)
        def experimento_largo():
            ...
    """

    def decorator(func: F) -> F:
        ok_kw = _merge_kwargs(
            send_kwargs, channel=channel, title=f"{title} -- OK",
            priority=success_priority, tags=("white_check_mark", "tada"),
        )
        err_kw = _merge_kwargs(
            send_kwargs, channel=channel, title=f"{title} -- ERROR",
            priority=error_priority, tags=("rotating_light",),
        )
        start_kw = _merge_kwargs(send_kwargs, channel=channel, title="Iniciando")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ntfy_client = _client_for(client)
            label = _label(func)
            if notify_start:
                _safe(lambda: ntfy_client.info(f"{label} iniciando...", **start_kw))
            t0 = time.monotonic()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if on_error:
                    elapsed = format_duration(time.monotonic() - t0)
                    message = (
                        f"{label} fallo tras {elapsed}.\n"
                        f"Excepcion: {type(exc).__name__}: {exc}"
                    )
                    if include_traceback:
                        message += "\n\n" + format_exception(exc)
                    _safe(lambda: ntfy_client.error(message, **err_kw))
                if re_raise:
                    raise
                return None
            if on_success:
                message = f"{label} termino OK"
                if include_duration:
                    message += f" en {format_duration(time.monotonic() - t0)}"
                _safe(lambda: ntfy_client.success(message, **ok_kw))
            return result

        return wrapper  # type: ignore[return-value]

    return _decorator_apply(decorator, _func)


# ---------------------------------------------------------------------------
# watch: context manager
# ---------------------------------------------------------------------------

class watch:
    """Context manager que notifica el destino de un bloque de codigo.

    Ejemplo::

        from src.ntfy import watch

        with watch("A6 common-rank (SS1)", channel="mi-canal"):
            correr_a6()

    Al salir sin errores -> notificacion de exito con la duracion.
    Al salir con excepcion -> notificacion urgente con el traceback; la
    excepcion se propaga intacta (``__exit__`` devuelve ``False``).
    """

    def __init__(
        self,
        label: str,
        *,
        channel: Optional[str] = None,
        client: Optional[NtfyClient] = None,
        title: Optional[str] = None,
        notify_start: bool = False,
        success_priority: str = "high",
        error_priority: str = "urgent",
        include_traceback: bool = True,
        **send_kwargs: Any,
    ) -> None:
        self.label = str(label)
        self._client = client
        self.notify_start = notify_start
        self.include_traceback = include_traceback
        self.ok_kw = _merge_kwargs(
            send_kwargs, channel=channel,
            title=title or f"{label} -- OK",
            priority=success_priority, tags=("white_check_mark",),
        )
        self.err_kw = _merge_kwargs(
            send_kwargs, channel=channel,
            title=title or f"{label} -- ERROR",
            priority=error_priority, tags=("rotating_light",),
        )
        self.start_kw = _merge_kwargs(send_kwargs, channel=channel, title="Iniciando")
        self._t0: Optional[float] = None

    def __enter__(self) -> "watch":
        self._t0 = time.monotonic()
        if self.notify_start:
            _safe(lambda: _client_for(self._client).info(
                f"{self.label} iniciando...", **self.start_kw,
            ))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        ntfy_client = _client_for(self._client)
        elapsed = format_duration(time.monotonic() - (self._t0 or time.monotonic()))
        if exc is None:
            _safe(lambda: ntfy_client.success(
                f"{self.label} termino OK en {elapsed}", **self.ok_kw,
            ))
        else:
            message = (
                f"{self.label} fallo tras {elapsed}.\n"
                f"Excepcion: {type(exc).__name__}: {exc}"
            )
            if self.include_traceback and exc.__traceback__ is not None:
                message += "\n\n" + format_exception(exc)
            _safe(lambda: ntfy_client.error(message, **self.err_kw))
        return False  # nunca tragar la excepcion
