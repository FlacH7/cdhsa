# -*- coding: utf-8 -*-
"""CLI del modulo ntfy: prueba rapida y envio desde la terminal.

Uso::

    python -m src.ntfy test
        Notificacion de prueba al canal de $NTFY_CHANNEL (todos los campos).

    python -m src.ntfy ping
        Ping minimo (prioridad 'min', no hace ruido).

    python -m src.ntfy send "mensaje" --title "Titulo" --priority high
        Envio arbitrario.

Opciones comunes a todos los subcomandos::

    --channel MI-CANAL   Canal destino (default: $NTFY_CHANNEL)
    --server URL         Servidor (default: $NTFY_SERVER o https://ntfy.sh)

Resolucion del canal, en orden de precedencia:

    1. --channel (argumento del comando);
    2. variable NTFY_CHANNEL exportada en la shell;
    3. NTFY_CHANNEL leida de un fichero .env en el directorio actual
       o en cualquier directorio padre (ver ``_load_env_file``). Esto hace
       que ``python -m src.ntfy test`` funcione desde la raiz del repo tal
       cual lo hace el batch (que la lee via src.utils.config).

A diferencia de la libreria (silenciosa ante fallos para no romper nunca una
corrida), el CLI SI devuelve codigo de salida != 0 si la notificacion no se
pudo enviar: es un herramienta de diagnostico.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .client import NtfyClient, NtfyError


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_env_file() -> Path | None:
    """Carga un ``.env`` (si existe) del CWD o de cualquier directorio padre.

    Mini-parser solo-stdlib pensado para el CLI: hace que
    ``python -m src.ntfy test`` funcione desde la raiz del repo sin exportar
    nada en la shell, igual que el batch (que la lee via ``src.utils.config``
    y python-dotenv). La LIBRERIA no usa esto a proposito: ``client.py`` solo
    ve variables del entorno del proceso, para seguir siendo 100% portable.

    Reglas (compatibles con python-dotenv en lo esencial):

    * busca ``.env`` en el directorio actual y hacia ARRIBA hasta la raiz;
      usa el PRIMERO que encuentra (el mas cercano al CWD gana);
    * ignora lineas vacias y comentarios que empiezan por ``#``;
    * admite el prefijo ``export `` y espacios alrededor del ``=``:
      ``NTFY_CHANNEL = valor`` y ``NTFY_CHANNEL=valor`` son equivalentes
      (la primera es la sintaxis que suele colarse en los .env);
    * quita comillas envolventes del valor (``"..."`` o ``'...'``);
    * NO sobreescribe variables que ya existan en el entorno: lo que este
      exportado en la shell (o fijado por el propio proceso) manda, igual
      que ``load_dotenv()`` por defecto.

    Comentarios al final de linea (``KEY=val # nota``) NO se admiten: ponlos
    en linea propia. Ante cualquier error devuelve None en silencio — esto
    es una conveniencia diagnostica, no una dependencia critica.

    Returns
    -------
    Path | None
        Ruta del .env cargado (None si no se encontro ninguno o fallo).
    """
    try:
        here = Path.cwd()
        env_path: Path | None = None
        for cand in (here, *here.parents):
            if (cand / ".env").is_file():
                env_path = cand / ".env"
                break
        if env_path is None:
            return None
        for raw in env_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export ") or line.startswith("export\t"):
                line = line[7:].lstrip()
            key, _, val = line.partition("=")
            key = key.strip()
            if not _ENV_KEY_RE.match(key):
                continue  # clave rara: ignorar linea, no romper el CLI
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1].strip()
            if key not in os.environ:
                os.environ[key] = val
        return env_path
    except Exception:
        return None


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--channel", "-c", default=None,
        help="Canal (topic) destino. Default: la variable NTFY_CHANNEL.",
    )
    parser.add_argument(
        "--server", default=None,
        help="Servidor ntfy. Default: NTFY_SERVER o https://ntfy.sh.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntfy",
        description=(
            "Notificaciones push para corridas largas. "
            "Configura NTFY_CHANNEL en el .env o usa --channel."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser(
        "test", help="Notificacion de prueba con todos los campos.",
    )
    _add_common_args(p_test)
    p_test.add_argument("--priority", default="high")

    p_ping = sub.add_parser("ping", help="Ping minimo (prioridad 'min').")
    _add_common_args(p_ping)

    p_send = sub.add_parser("send", help="Enviar un mensaje.")
    _add_common_args(p_send)
    p_send.add_argument("message", help="Cuerpo del mensaje.")
    p_send.add_argument("--title", default=None)
    p_send.add_argument(
        "--priority", default=None,
        help="min | low | default | high | urgent",
    )
    p_send.add_argument(
        "--tags", default=None,
        help="Shortcodes de emoji separados por coma, p.ej. 'fire,chart'.",
    )
    p_send.add_argument("--click", default=None, help="URL al pulsar.")
    p_send.add_argument(
        "--delay", default=None,
        help="Entrega diferida: 30min, 11h, 9am, ...",
    )
    p_send.add_argument(
        "--markdown", action="store_true", help="Renderizar como Markdown.",
    )

    return parser


def _test_message(channel: str, server: str) -> str:
    return (
        f"Si ves esto, el canal **{channel}** esta funcionando.\n\n"
        f"- Servidor : `{server}`\n"
        f"- Python   : `{platform.python_version()}`\n"
        f"- Host     : `{platform.node() or 'desconocido'}`\n"
        f"- Fecha    : `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- ID       : `{uuid.uuid4().hex[:8]}`\n\n"
        "Puedes borrar este mensaje en la app deslizando a un lado."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # El CLI (a diferencia de la libreria) SI lee el .env del repo:
    # es una herramienta de diagnostico pensada para lanzarse desde la
    # raiz del repo, igual que el batch. Nunca sobreescribe lo que ya
    # este exportado en la shell.
    _load_env_file()

    channel = (args.channel or os.getenv("NTFY_CHANNEL") or "").strip()
    if not channel:
        print(
            "ERROR: no hay canal. Opciones:\n"
            "  1) NTFY_CHANNEL=mi-canal en el .env del repo, lanzando el "
            "comando\n     desde la raiz del repo (o cualquier subdirectorio);\n"
            "  2) export NTFY_CHANNEL=mi-canal en la shell;\n"
            "  3) --channel mi-canal en el propio comando.",
            file=sys.stderr,
        )
        return 2

    # El CLI si quiere excepciones: es una herramienta de diagnostico.
    client = NtfyClient(
        channel=channel, server=args.server, raise_on_error=True,
    )

    try:
        if args.command == "test":
            server = args.server or os.getenv("NTFY_SERVER") or "https://ntfy.sh"
            response = client.send(
                _test_message(channel, server),
                title="Prueba de ntfy",
                priority=args.priority,
                tags=("white_check_mark", "gear"),
                markdown=True,
            )
        elif args.command == "ping":
            response = client.ping()
        else:  # send
            kwargs = {}
            if args.title:
                kwargs["title"] = args.title
            if args.priority:
                kwargs["priority"] = args.priority
            if args.tags:
                kwargs["tags"] = args.tags
            if args.click:
                kwargs["click"] = args.click
            if args.delay:
                kwargs["delay"] = args.delay
            if args.markdown:
                kwargs["markdown"] = True
            response = client.send(args.message, **kwargs)
    except NtfyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    msg_id = (response or {}).get("id", "?")
    print(f"[OK] Notificacion enviada al canal '{channel}' (id={msg_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
