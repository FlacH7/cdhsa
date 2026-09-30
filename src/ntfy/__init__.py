# -*- coding: utf-8 -*-
"""ntfy -- notificaciones push para corridas largas de computo.

Paquete autocontenido (solo biblioteca estandar de Python, cero dependencias)
para enviar notificaciones al telefono a traves de https://ntfy.sh (o un
servidor self-hosted).  Pensado para monitorizar experimentos que corren
horas en un server remoto por SSH + ``nohup``.

Setup rapido
-----------
1. Elige un nombre de canal largo y aleatorio (el canal ES la contrasena)::

       NTFY_CHANNEL=mi-canal-secreto-8f3k2q

   y anade la variable a tu ``.env`` (o exportala en el shell).
2. Instala la app `ntfy` en el telefono (Android/iOS) y suscribete a ese
   mismo topic.
3. Prueba::

       python -m src.ntfy test

Uso tipico
----------
>>> from src.ntfy import notify_success, notify_on_critical_error
>>> notify_success("Entrenamiento terminado", channel="mi-canal")
>>>
>>> @notify_on_critical_error(channel="mi-canal", title="[proy] ERROR")
... def pipeline_largo(**cfg):
...     ...

Documentacion completa en ``README.md`` (dentro de esta misma carpeta).
"""

from .client import (
    DEFAULT_SERVER,
    VALID_PRIORITIES,
    NtfyClient,
    NtfyError,
    configure,
    get_default_client,
    reset_default_client,
    notify,
    notify_info,
    notify_success,
    notify_warning,
    notify_error,
    ping,
)
from .decorators import (
    format_duration,
    format_exception,
    notify_calls,
    notify_on_critical_error,
    notify_on_error,
    notify_on_success,
    watch,
)

__version__ = "1.0.0"

__all__ = [
    # cliente
    "NtfyClient",
    "NtfyError",
    "configure",
    "get_default_client",
    "reset_default_client",
    "notify",
    "notify_info",
    "notify_success",
    "notify_warning",
    "notify_error",
    "ping",
    "DEFAULT_SERVER",
    "VALID_PRIORITIES",
    # decoradores
    "notify_on_success",
    "notify_on_critical_error",
    "notify_on_error",
    "notify_calls",
    "watch",
    "format_duration",
    "format_exception",
]
