# -*- coding: utf-8 -*-
"""Cliente HTTP ligero para el servicio de notificaciones `ntfy`.

Disenado para monitorizar corridas largas de computo (SSH + ``nohup``) sin
acoplar el codigo cientifico a la infraestructura de notificaciones.

Caracteristicas clave
---------------------
* **Cero dependencias**: usa unicamente la biblioteca estandar
  (``urllib.request``), por lo que la carpeta se puede copiar tal cual a
  cualquier otro repo.
* **A prueba de fallos**: por defecto un error de red/HTTP *nunca* interrumpe
  la ejecucion del programa anfitrion; se registra como ``logging.warning`` y
  :meth:`NtfyClient.send` devuelve ``None``.
* **El canal (topic) actua como contrasena**: en ntfy cualquiera que sepa el
  nombre del canal puede leer y escribir.  El canal se resuelve en este orden:

  1. argumento ``channel`` explicito en la llamada,
  2. canal del cliente (constructor / :func:`configure`),
  3. variable de entorno ``NTFY_CHANNEL``.

* **Servidor configurable** (``NTFY_SERVER``): ntfy.sh oficial o uno
  self-hosted.
* **Reintentos** configurables con backoff corto (por defecto 1 reintento).

Variables de entorno reconocidas (todas opcionales):

``NTFY_CHANNEL``
    Canal por defecto.  Vacio o ausente -> notificaciones deshabilitadas.

``NTFY_SERVER``
    Servidor por defecto (default ``https://ntfy.sh``).

``NTFY_ENABLED``
    ``0`` / ``false`` / ``no`` / ``off`` -> silencia el modulo por completo
    sin tocar el codigo.

Referencia de la API HTTP: https://docs.ntfy.sh/publish/
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Union

__all__ = [
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
]

LOGGER = logging.getLogger("ntfy")

#: Instancia publica oficial del servicio.
DEFAULT_SERVER = "https://ntfy.sh"

#: Prioridades validas del protocolo (de menor a mayor urgencia).
VALID_PRIORITIES = ("min", "low", "default", "high", "urgent")

#: Alias numerico aceptado por ntfy (1-5).
_PRIORITY_ALIASES = {1: "min", 2: "low", 3: "default", 4: "high", 5: "urgent"}

#: Timeout por defecto (s): una notificacion jamas debe frenar la corrida.
DEFAULT_TIMEOUT = 10.0

#: Limite practico del cuerpo del mensaje (ntfy trunca en ~4 KB).
MAX_MESSAGE_CHARS = 3900

#: Marcador que se anade cuando el mensaje se recorta.
_TRUNCATION_MARKER = "\n[... mensaje truncado]"

#: Un topic valido: alfanumerico, guiones y guiones bajos.
_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class NtfyError(RuntimeError):
    """Fallo al enviar una notificacion (red, HTTP != 2xx, canal invalido...)."""


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _normalize_priority(priority: Any) -> Optional[str]:
    """Devuelve la prioridad canonica ('min'...'urgent') o None si es invalida."""
    if priority is None:
        return None
    if isinstance(priority, bool):  # bool es subclase de int: evitar True==1
        return None
    if isinstance(priority, int):
        return _PRIORITY_ALIASES.get(priority)
    p = str(priority).strip().lower()
    return p if p in VALID_PRIORITIES else None


def _normalize_tags(tags: Union[str, Iterable[str], None]) -> Optional[List[str]]:
    """Normaliza ``'a,b'`` | iterable | None -> ``['a', 'b']`` | None."""
    if tags is None:
        return None
    if isinstance(tags, str):
        parts = [p.strip() for p in tags.split(",")]
    else:
        parts = [str(p).strip() for p in tags]
    out = [p for p in parts if p]
    return out or None


def _sanitize_header(value: Any) -> str:
    """Los headers HTTP no admiten saltos de linea: se sustituyen por ' / '."""
    return str(value).replace("\r", " ").replace("\n", " / ").strip()


def _env_flag(env_name: str) -> bool:
    """Lee un flag booleano de una variable de entorno (default True)."""
    raw = os.getenv(env_name, "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------

class NtfyClient:
    """Cliente HTTP para publicar notificaciones en un canal de ntfy.

    Parametros
    ----------
    channel:
        Canal (topic) al que publicar.  Si es ``None`` se lee de la variable
        de entorno ``NTFY_CHANNEL``.
    server:
        URL base del servidor (default ``https://ntfy.sh`` o ``$NTFY_SERVER``).
    timeout:
        Timeout en segundos para cada intento HTTP.
    enabled:
        Interruptor maestro (default: valor de ``$NTFY_ENABLED``).  Con
        ``enabled=False`` todas las llamadas son no-ops silenciosos.
    raise_on_error:
        Si es ``True``, los fallos de envio lanzan :class:`NtfyError` en vez
        de limitarse a registrar un warning y devolver ``None``.  En una
        corrida cientifica esto casi nunca es deseable; se usa sobre todo en
        scripts de prueba y en el CLI.
    retries:
        Reintentos extra ante errores de red o HTTP 429 (default 1, con
        backoff corto).
    auth:
        Credenciales si el topic esta protegido: ``"usuario:password"`` o
        un token de acceso ``"tk_..."`` (requiere cuenta en ntfy.sh o un
        servidor self-hosted con autenticacion).
    """

    def __init__(
        self,
        channel: Optional[str] = None,
        server: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        enabled: Optional[bool] = None,
        raise_on_error: bool = False,
        retries: int = 1,
        auth: Optional[str] = None,
    ) -> None:
        if enabled is None:
            enabled = _env_flag("NTFY_ENABLED")
        self._enabled = bool(enabled)
        self.channel = (channel or os.getenv("NTFY_CHANNEL") or "").strip() or None
        self.server = (
            server or os.getenv("NTFY_SERVER") or DEFAULT_SERVER
        ).strip().rstrip("/")
        self.timeout = float(timeout)
        self.raise_on_error = bool(raise_on_error)
        self.retries = max(0, int(retries))
        self._auth_header = self._build_auth_header(auth)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_auth_header(auth: Optional[str]) -> Optional[str]:
        if not auth:
            return None
        if ":" in auth:  # usuario:password -> Basic
            cred = base64.b64encode(auth.encode("utf-8")).decode("ascii")
            return f"Basic {cred}"
        return f"Bearer {auth}"  # token de acceso

    @property
    def enabled(self) -> bool:
        """True si el cliente puede enviar (interruptor activo y canal)."""
        return bool(self._enabled and self.channel)

    def __repr__(self) -> str:  # pragma: no cover
        state = "on" if self.enabled else "off"
        return f"<NtfyClient {state} server={self.server!r} channel={self.channel!r}>"

    # ------------------------------------------------------------------
    # Envio
    # ------------------------------------------------------------------

    def send(
        self,
        message: str,
        *,
        title: Optional[str] = None,
        priority: Optional[str] = None,
        tags: Union[str, Iterable[str], None] = None,
        click: Optional[str] = None,
        actions: Optional[Iterable[Dict[str, Any]]] = None,
        delay: Optional[str] = None,
        markdown: bool = False,
        email: Optional[str] = None,
        icon: Optional[str] = None,
        filename: Optional[str] = None,
        cache: bool = True,
        firebase: bool = True,
        channel: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raise_on_error: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Publica ``message`` en el canal y devuelve la respuesta JSON.

        Devuelve ``None`` (sin lanzar) si el cliente esta deshabilitado, si
        no hay canal o si el envio fallo y ``raise_on_error`` es falso.

        Parametros clave
        ----------------
        title:
            Titulo de la notificacion.
        priority:
            ``'min' | 'low' | 'default' | 'high' | 'urgent'`` (o 1-5).
            Una prioridad invalida se ignora con un warning.
        tags:
            Emojis por *shortcode*, p. ej. ``('white_check_mark', 'tada')``
            o ``'fire,chart'``.
        click:
            URL que se abre al pulsar la notificacion.
        actions:
            Botones de accion, p. ej.::

                [{"action": "view", "label": "Ver repo",
                  "url": "https://github.com/usuario/repo"}]

        delay:
            Entrega diferida, p. ej. ``'30min'``, ``'11h'``, ``'9am'``
            o ``'tomorrow, 9:00'`` (max 3 dias en ntfy.sh).
        markdown:
            Renderizar el cuerpo como Markdown.
        email:
            Enviar ademas una copia por correo.
        icon / filename / icon URL:
            Ver la referencia de la API en https://docs.ntfy.sh/publish/.
        cache:
            ``False`` -> cabecera ``Cache: no`` (el mensaje no queda en el
            historial del topic para suscriptores tardios).
        firebase:
            ``False`` -> cabecera ``Firebase: no`` (utiles para self-hosted).
        channel:
            Canal explicito para esta llamada (pisa el del cliente).
        extra_headers:
            Headers adicionales (escape hatch para features nuevas de ntfy).
        raise_on_error:
            Pisa, solo para esta llamada, la opcion del constructor.
        """
        raise_opt = self.raise_on_error if raise_on_error is None else bool(raise_on_error)

        if not self._enabled:
            LOGGER.debug(
                "[ntfy] deshabilitado (enabled=False); omitido: %r",
                title or str(message)[:40],
            )
            return None

        topic = (channel or self.channel or "").strip()
        if not topic:
            LOGGER.debug(
                "[ntfy] sin canal configurado (NTFY_CHANNEL); omitido: %r",
                title or str(message)[:40],
            )
            return None

        if not _TOPIC_RE.match(topic):
            return self._fail(
                NtfyError(
                    f"Canal ntfy invalido: {topic!r} "
                    "(se espera [A-Za-z0-9_-], max 64 chars)"
                ),
                raise_opt,
            )

        # --- Cuerpo -------------------------------------------------
        message = "" if message is None else str(message)
        if len(message) > MAX_MESSAGE_CHARS:
            message = message[:MAX_MESSAGE_CHARS] + _TRUNCATION_MARKER

        # --- Cabeceras ----------------------------------------------
        headers: Dict[str, str] = {}
        if title:
            headers["Title"] = _sanitize_header(title)
        prio = _normalize_priority(priority)
        if prio:
            headers["Priority"] = prio
        elif priority is not None:
            LOGGER.warning("[ntfy] prioridad invalida (%r); se ignora.", priority)
        tags_n = _normalize_tags(tags)
        if tags_n:
            headers["Tags"] = ",".join(tags_n)
        if click:
            headers["Click"] = _sanitize_header(click)
        if delay:
            headers["Delay"] = _sanitize_header(delay)
        if markdown:
            headers["Markdown"] = "true"
        if email:
            headers["Email"] = _sanitize_header(email)
        if icon:
            headers["Icon"] = _sanitize_header(icon)
        if filename:
            headers["Filename"] = _sanitize_header(filename)
        if actions:
            headers["Actions"] = json.dumps(list(actions))
        if not cache:
            headers["Cache"] = "no"
        if not firebase:
            headers["Firebase"] = "no"
        if self._auth_header:
            headers["Authorization"] = self._auth_header
        if extra_headers:
            for key, value in extra_headers.items():
                headers[str(key)] = _sanitize_header(value)

        url = f"{self.server}/{topic}"
        data = message.encode("utf-8")

        last_error: Optional[NtfyError] = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                try:
                    return json.loads(body)
                except ValueError:
                    return {"raw": body}
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:200]
                except Exception:  # pragma: no cover
                    pass
                last_error = NtfyError(
                    f"HTTP {exc.code} publicando en {url}: {detail or exc.reason}"
                )
                # 429 (rate limit) y los 5xx son transitorios -> reintento;
                # el resto de los 4xx no suele recuperarse.
                transient = exc.code == 429 or exc.code >= 500
                if not transient or attempt >= self.retries:
                    return self._fail(last_error, raise_opt)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = NtfyError(f"Error de red contactando {url}: {exc}")
                if attempt >= self.retries:
                    return self._fail(last_error, raise_opt)

            if attempt < self.retries:
                time.sleep(1.0 + attempt)  # backoff corto: 1 s, 2 s, ...

        return self._fail(last_error, raise_opt)  # pragma: no cover

    def _fail(self, exc: NtfyError, raise_it: bool):
        if raise_it:
            raise exc
        LOGGER.warning("%s", exc)
        return None

    # ------------------------------------------------------------------
    # Atajos semanticos
    # ------------------------------------------------------------------

    def info(
        self, message: str, *, title: str = "Informacion",
        tags: Union[str, Iterable[str], None] = ("information_source",),
        priority: Optional[str] = "low", **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Aviso informativo (prioridad baja)."""
        return self.send(message, title=title, tags=tags, priority=priority, **kwargs)

    def success(
        self, message: str, *, title: str = "Completado",
        tags: Union[str, Iterable[str], None] = ("white_check_mark", "tada"),
        priority: Optional[str] = "high", **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Aviso de exito (prioridad alta)."""
        return self.send(message, title=title, tags=tags, priority=priority, **kwargs)

    def warning(
        self, message: str, *, title: str = "Atencion",
        tags: Union[str, Iterable[str], None] = ("warning",),
        priority: Optional[str] = "high", **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Aviso de alerta (prioridad alta)."""
        return self.send(message, title=title, tags=tags, priority=priority, **kwargs)

    def error(
        self, message: str, *, title: str = "ERROR",
        tags: Union[str, Iterable[str], None] = ("rotating_light",),
        priority: Optional[str] = "urgent", **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Aviso de error critico (prioridad urgente: suena y vibra)."""
        return self.send(message, title=title, tags=tags, priority=priority, **kwargs)

    def ping(
        self, message: str = "ping", *, title: str = "Ping ntfy",
        tags: Union[str, Iterable[str], None] = ("bell",),
        priority: Optional[str] = "min", **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Notificacion minima para verificar la conectividad del canal."""
        return self.send(message, title=title, tags=tags, priority=priority, **kwargs)


# ---------------------------------------------------------------------------
# Cliente por defecto del proceso (singleton perezoso)
# ---------------------------------------------------------------------------

_DEFAULT_CLIENT: Optional[NtfyClient] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_client() -> NtfyClient:
    """Devuelve el cliente por defecto, creandolo si es la primera vez.

    El cliente por defecto resuelve su canal/servidor de las variables de
    entorno ``NTFY_CHANNEL`` / ``NTFY_SERVER`` en el momento de la CREACION
    (se cachea).  Si cambias el entorno en caliente, llama
    :func:`reset_default_client` o :func:`configure`.
    """
    global _DEFAULT_CLIENT
    with _DEFAULT_LOCK:
        if _DEFAULT_CLIENT is None:
            _DEFAULT_CLIENT = NtfyClient()
        return _DEFAULT_CLIENT


def reset_default_client() -> None:
    """Olvida el cliente por defecto (la proxima llamada lo re-crea)."""
    global _DEFAULT_CLIENT
    with _DEFAULT_LOCK:
        _DEFAULT_CLIENT = None


def configure(
    channel: Optional[str] = None,
    server: Optional[str] = None,
    **kwargs: Any,
) -> NtfyClient:
    """Configura el cliente por defecto del proceso.

    Ejemplo::

        import ntfy
        ntfy.configure(channel="mi-canal")
        ntfy.notify_success("todo bien")
    """
    global _DEFAULT_CLIENT
    with _DEFAULT_LOCK:
        _DEFAULT_CLIENT = NtfyClient(channel=channel, server=server, **kwargs)
        return _DEFAULT_CLIENT


# ---------------------------------------------------------------------------
# Funciones de modulo (atajos sobre el cliente por defecto)
# ---------------------------------------------------------------------------

def notify(message: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Envia una notificacion cruda (ver :meth:`NtfyClient.send`)."""
    return get_default_client().send(message, **kwargs)


def notify_info(message: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Aviso informativo (prioridad baja)."""
    return get_default_client().info(message, **kwargs)


def notify_success(message: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Aviso de exito (prioridad alta)."""
    return get_default_client().success(message, **kwargs)


def notify_warning(message: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Aviso de alerta (prioridad alta)."""
    return get_default_client().warning(message, **kwargs)


def notify_error(message: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Aviso de error critico (prioridad urgente)."""
    return get_default_client().error(message, **kwargs)


def ping(message: str = "ping", **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Ping de conectividad contra el canal configurado."""
    return get_default_client().ping(message, **kwargs)
