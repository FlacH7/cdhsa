# ntfy — Notificaciones push para corridas largas

Paquete **autocontenido y reutilizable** (solo biblioteca estándar de Python,
**cero dependencias**) para enviar notificaciones al teléfono a través de
[ntfy](https://ntfy.sh) — servicio gratuito y sin cuenta — o de un servidor
self-hosted.

**Motivación típica**: lanzas un experimento de varias horas en un server por
SSH con `nohup` y te vas. Con este módulo el propio proceso te avisa al móvil
cuando termina, cuándo falla un job, o si una excepción crítica lo mata.
Nada de revisar el log al día siguiente para descubrir que murió a los 20
minutos.

---

## 1. Instalación

Copia la carpeta `ntfy/` a cualquier proyecto:

```
tu-repo/
└── src/
    └── ntfy/          <- esta carpeta completa
        ├── __init__.py
        ├── client.py
        ├── decorators.py
        ├── __main__.py
        └── README.md
```

No hay nada que instalar con pip: dentro solo se usan `urllib`, `json`,
`logging`, `threading` y `time`. Funciona igual como `src.ntfy` (dentro de un
paquete `src`) o suelto como `ntfy` (renombra la ruta como te convenga).

---

## 2. Configuración

### 2.1 El canal (topic)

En ntfy no hay usuarios ni contraseñas para topics públicos: **el nombre del
canal es la credencial**. Cualquiera que sepa el nombre puede leer y publicar
ahí, así que:

* Elige un nombre **largo y aleatorio**, p. ej. `cdhsa-batch-8f3k2qmx` (no
  uses `test`, `alertas`, ni nada adivinable).
* No envíes por el canal datos sensibles (rutas internas críticas, IPs,
  resultados confidenciales del paper antes de publicar).
* Si quieres más control, puedes crear una cuenta gratuita en ntfy.sh y
  **reservar** el topic (impide que otros lo usen), o self-hostear ntfy y
  apuntar `NTFY_SERVER` a tu servidor.

### 2.2 Variables de entorno

| Variable        | Default             | Significado                                             |
|-----------------|---------------------|---------------------------------------------------------|
| `NTFY_CHANNEL`  | *(vacío = off)*     | Canal por defecto. Sin ella el módulo es un no-op.      |
| `NTFY_SERVER`   | `https://ntfy.sh`   | Servidor (para self-hosted).                            |
| `NTFY_ENABLED`  | `1`                 | `0`/`false`/`no`/`off` silencia todo sin tocar código.  |

### 2.3 El teléfono

1. Instala la app **ntfy** (Android: Play Store / F-Droid; iOS: App Store).
2. Pulsa `+` y suscríbete al topic **exacto** que pusiste en `NTFY_CHANNEL`.
3. (Android) Para que las prioridades `high`/`urgent` suenen incluso con el
   teléfono en silencio, concede el permiso de notificaciones y considera
   activar la entrega instantánea por WebSocket en los ajustes del topic.

### 2.4 Prueba rápida

```bash
NTFY_CHANNEL=mi-canal-secreto python -m src.ntfy test
# o apuntando a un canal concreto sin tocar el .env:
python -m src.ntfy test --channel mi-canal-secreto
```

Deberías recibir una notificación con ✅ en menos de un par de segundos. Si
no llega, revisa que el topic del teléfono sea *exactamente* el mismo
(distingue mayúsculas… por convención usa solo minúsculas y guiones).

---

## 3. Uso básico (API)

### 3.1 Funciones de módulo (lo más simple)

```python
from src.ntfy import notify, notify_info, notify_success, notify_warning, notify_error, ping

notify("mensaje crudo", title="Titulo", priority="high", tags=("fire",))
notify_info("arrancando el preprocesado")           # prioridad baja
notify_success("Entrenamiento terminado en 3h 12m") # ✅ prioridad alta
notify_warning("3 jobs fallaron, el resto bien")     # ⚠️ prioridad alta
notify_error("OOM: el job de SS3 murió")             # 🚨 prioridad urgente
ping()                                               # campanita mínima
```

Todas aceptan los mismos extras que `send()` (ver tabla en §5), incluido
`channel="otro-canal"` para un envío puntual a otro topic.

### 3.2 Cliente explícito (cuando quieres otra configuración)

```python
from src.ntfy import NtfyClient

client = NtfyClient(
    channel="mi-canal",
    server="https://ntfy.midominio.org",  # opcional: self-hosted
    timeout=5.0,                          # segundos por intento
    retries=2,                            # reintentos ante errores de red
    # auth="usuario:password",            # o "tk_..." si el topic es privado
    # raise_on_error=True,               # NO recomendado en producción
)
client.success("listo")
```

### 3.3 Configurar el cliente por defecto del proceso

```python
import ntfy  # si lo usas como paquete suelto
ntfy.configure(channel="mi-canal", server="https://ntfy.sh")
ntfy.notify_success("todo bien")
```

---

## 4. Decoradores y context manager

### 4.1 `@notify_on_critical_error` — el avisador de desastres

Notifica con prioridad **urgent** si una excepción escapa de la función y
**la relanza** (la ejecución se detiene exactamente igual que sin decorador;
notificar no cura nada):

```python
from src.ntfy import notify_on_critical_error

@notify_on_critical_error(title="[CD-HSA] ERROR crítico", catch_system_exit=True)
def main():
    ...
```

Opciones útiles:

| Opción             | Default | Significado                                              |
|--------------------|---------|----------------------------------------------------------|
| `include_traceback`| `True`  | Adjunta las últimas 12 líneas del traceback.             |
| `catch_system_exit`| `False` | Avisa también de `sys.exit(código≠0)` (p. ej. JSON de parámetros inválido). `--help` (exit 0) nunca avisa. |
| `notify_start`     | `False` | Avisa también al *entrar* en la función.                 |
| `re_raise`         | `True`  | Relanza la excepción tras notificar. Déjalo en `True`.  |
| `channel` / `client` | —     | Destino (default: `$NTFY_CHANNEL` vía cliente por defecto). |

### 4.2 `@notify_on_success` — el avisador de final feliz

```python
from src.ntfy import notify_on_success

@notify_on_success(title="[proyecto] Fin de entrenamiento", send_result=True)
def entrenar():
    ...  # notifica "modulo.entrenar terminó OK en 4h 02m 11s"
```

### 4.3 `@notify_calls` — monitor completo

Une los dos anteriores (+ inicio opcional) en un solo decorador:

```python
from src.ntfy import notify_calls

@notify_calls(title="[proyecto] Experimento", notify_start=True)
def experimento():
    ...
```

### 4.4 `watch` — bloques sueltos

```python
from src.ntfy import watch

with watch("A6 common-rank (SS1)"):
    correr_a6()   # avisa del destino del bloque, la excepción sigue propagándose
```

### 4.5 `format_duration` / `format_exception`

Helpers públicos que usan los decoradores, por si quieres construir tus
propios mensajes: `format_duration(15126.3)` → `"4h 12m 06s"`.

---

## 5. Referencia de `send()`

`NtfyClient.send(message, **opciones)` y su reflejo `notify(message, **opciones)`:

| Opción           | Tipo                | Significado                                        |
|------------------|---------------------|----------------------------------------------------|
| `title`          | `str`               | Título de la notificación.                         |
| `priority`       | `str` o `int` 1–5   | `min, low, default, high, urgent` (ver §6).        |
| `tags`           | lista o `'a,b'`     | Emojis por shortcode (ver §7).                    |
| `click`          | `str`               | URL que se abre al pulsar la notificación.         |
| `actions`        | lista de dicts      | Botones de acción (ver §8.2).                      |
| `delay`          | `str`               | Entrega diferida: `'30min'`, `'11h'`, `'9am'`, `'tomorrow, 9:00'` (máx. 3 días). |
| `markdown`       | `bool`              | Renderiza el cuerpo como Markdown.                 |
| `email`          | `str`               | Copia por correo.                                  |
| `icon`           | `str` (URL)         | Icono de la notificación.                           |
| `filename`       | `str`               | Nombre mostrado si se usa como adjunto.             |
| `cache`          | `bool` (`True`)     | `False` → `Cache: no` (no queda en el historial).   |
| `firebase`       | `bool` (`True`)     | `False` → `Firebase: no` (útil self-hosted).        |
| `channel`        | `str`               | Canal puntual para esta llamada.                    |
| `extra_headers`  | `dict`              | Cabeceras crudas extra (escape hatch).              |
| `raise_on_error` | `bool`              | Pisa la opción del constructor.                    |

**Semántica de retorno**: dict con la respuesta JSON del servidor
(`{"id": "...", "event": "message", ...}`) si todo fue bien; `None` si el
cliente estaba deshabilitado, no había canal, o el envío falló (en modo
por-defecto silencioso). El cuerpo se trunca automáticamente a ~3.9 KB con
marcador `[... mensaje truncado]`.

**Robustez**: si no hay canal configurado, todo el módulo es un no-op
silencioso (solo `logger.debug`), por lo que puedes dejar las llamadas
escritas aunque el servidor no tenga `.env` configurado.

---

## 6. Prioridades

| Valor      | Comportamiento aproximado en el teléfono                     |
|------------|---------------------------------------------------------------|
| `min`      | Ni siquiera aparece como pop-up; solo queda en la app.        |
| `low`      | Sin sonido ni vibración.                                       |
| `default`  | Sonido/vibración según los ajustes del sistema.               |
| `high`     | Suena aunque el teléfono esté en modo no molestar (por topic).|
| `urgent`   | Máximo volumen/vibración; en Android insiste hasta que la veas.|

Defaults de los atajos: `info→low`, `success→high`, `warning→high`,
`error→urgent`, `ping→min`.

---

## 7. Emojis (`tags`) que más se usan

| Shortcode            | Emoji | Shortcode             | Emoji |
|----------------------|-------|-----------------------|-------|
| `white_check_mark`   | ✅    | `rotating_light`      | 🚨    |
| `tada`               | 🎉    | `warning`             | ⚠️    |
| `fire`               | 🔥    | `information_source`  | ℹ️    |
| `chart_with_upwards_trend` | 📈 | `chart_with_downwards_trend` | 📉 |
| `brain`              | 🧠    | `computer`            | 💻    |
| `hourglass`          | ⏳    | `bell`               | 🔔    |
| `rocket`             | 🚀    | `bug`                 | 🐛    |
| `satellite_antenna`  | 📡    | `zzz`                 | 💤    |

Puedes encadenar varios: `tags=("brain", "chart_with_upwards_trend")`.

---

## 8. Recetas

### 8.1 Patrón canary (saber en el minuto 0 que el canal va bien)

Envía una notificación al **arrancar** la corrida. Si en el móvil no suena en
los primeros minutos, mata el proceso y revisa el `.env`: te acabas de
ahorrar horas de corrida inútil.

```python
notify_info("Batch iniciado: 25 jobs (ventana 100–200 s). Host: gpu-server.")
```

### 8.2 Botones de acción

```python
notify_success(
    "Corrida v5 terminada",
    title="[CD-HSA] Batch OK",
    tags=("white_check_mark", "brain"),
    click="https://github.com/FlacH7/cdhsa",          # al pulsar la notificación
    actions=[{
        "action": "view", "label": "Ver repo",
        "url": "https://github.com/FlacH7/cdhsa",
    }],
)
```

### 8.3 Watchdog diferido (silencio = muerte)

Un proceso matado con `kill -9` o por el OOM killer no puede avisar. Puedes
programar un mensaje diferido al arrancar: si a las 11 h no ha llegado el
"[OK]" del final, es que murió por el camino.

```python
notify(
    "Si no llegó un [OK] antes de este aviso, la corrida murió "
    "silenciosamente (kill -9 / OOM). Revisa el log.",
    title="Watchdog del batch",
    delay="11h",
    priority="default",
)
```

### 8.4 Silenciar temporalmente sin tocar código

```bash
NTFY_ENABLED=0 nohup python -m src.batch_runs.run_batch_cdhsa ...
```

### 8.5 Integración típica con `.env` + config del repo

```python
# src/utils/config.py  (igual que el resto de variables)
NTFY_CHANNEL = os.getenv("NTFY_CHANNEL")

# donde se use:
from src.utils.config import NTFY_CHANNEL
notify_success("terminó", channel=NTFY_CHANNEL or None)
```

---

## 9. Límites del servicio (ntfy.sh gratuito)

* **Rate limit**: ~60 mensajes/hora por IP con ráfaga inicial de 60. Para un
  batch normal (arranque + 1 por fallo + final) sobra; no lo uses como log.
* **Tamaño del mensaje**: ~4 KB (aquí se trunca a 3.9 KB).
* **`delay`**: máximo 3 días.
* **Fiabilidad**: ntfy.sh es "best effort" sin SLA; si necesitas garantías,
  self-hostea el servidor y apunta `NTFY_SERVER` a él.
* **Timeouts**: el cliente usa 10 s por intento y 1 reintento por defecto —
  una notificación nunca debe frenar tu corrida.

---

## 10. Solución de problemas

| Síntema | Causa probable |
|---------|----------------|
| No llega nada | Topic en la app distinto del `NTFY_CHANNEL` (revisa carácter a carácter). |
| No llega nada y hay warning en el log | El server no tiene salida a Internet (puerto 443) o hay proxy: define `https_proxy`. |
| `[ERROR] HTTP 403` | El topic está reservado por otra cuenta: cambia de canal. |
| `[ERROR] HTTP 429` | Rate limit: demasiados mensajes seguidos. |
| Solo llega al abrir la app | (Android) Conexión instantánea desactivada; habilita WebSocket en los ajustes del topic. |
| `Canal ntfy invalido` | El topic tiene espacios, `:` o `#`. Solo `[A-Za-z0-9_-]`, máx. 64. |

Log interno: el módulo escribe en el logger `"ntfy"` — con `logging.basicConfig(level=logging.DEBUG)` verás cada omisión/envío.

---

## 11. Integración actual en `run_batch_cdhsa.py` (CD-HSA)

El batch usa exactamente lo descrito arriba:

1. `NTFY_CHANNEL` se importa desde `src.utils.config` (como el resto de variables).
2. **Inicio**: info con nº de jobs, ventana, host y directorio de salidas (canary).
3. **Fallo de job**: error urgente por cada job que muere (el batch continúa).
4. **Final**: success si `Fallos == 0`; warning con el resumen si hubo fallos.
5. `@notify_on_critical_error(catch_system_exit=True)` sobre `main()`: si una
   excepción no capturada (o un `sys.exit(1)` por JSON inválido) mata el
   batch, llega un urgent con el traceback antes de que el proceso muera.

Sin `NTFY_CHANNEL` definido, todo esto es no-op y el batch se comporta
exactamente igual que antes.
