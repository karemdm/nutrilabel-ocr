"""Local web interface: open one image or a whole folder and review the results.

    nutrilabel ui                       # http://127.0.0.1:8765
    nutrilabel ui --engine paddleocr --port 9000

Pure standard library (``http.server``): no web framework to install. Images are
processed on this machine and never leave it. The server binds to 127.0.0.1 by
default; use ``--host 0.0.0.0`` only inside a container or a trusted network.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

MAX_UPLOAD_BYTES = 30 * 1024 * 1024

_LOCK = threading.Lock()  # OCR models are not guaranteed to be thread-safe
_READERS: dict[str, Any] = {}


def page_html(preloaded: dict | None = None) -> str:
    """The single-page UI. ``preloaded`` renders a static snapshot (used for docs)."""
    html = resources.files("nutrilabel.ui").joinpath("index.html").read_text(encoding="utf-8")
    if preloaded is not None:
        payload = json.dumps(preloaded, ensure_ascii=False).replace("</", "<\\/")
        html = html.replace("<!--PRELOADED-->", f"<script>window.__PRELOADED__ = {payload};</script>")
    return html


def analyse_image(data: bytes, engine: str, name: str = "image") -> dict[str, Any]:
    """Run the full pipeline on raw image bytes and build the JSON the page expects."""
    from nutrilabel.imaging.preprocess import load_image
    from nutrilabel.imaging.viz import draw_result
    from nutrilabel.pipeline import NutritionReader

    try:
        image = load_image(data)
    except Exception as e:  # noqa: BLE001 - unreadable file
        return {"name": name, "error": f"cannot open image: {e}"}
    try:
        with _LOCK:
            if engine not in _READERS:
                _READERS[engine] = NutritionReader(engine=engine)
            reader = _READERS[engine]
            t0 = time.perf_counter()
            result = reader.read(image)
            total = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - show the photo and the error on the page
        _READERS.pop(engine, None)
        return {"name": name, "engine": engine, "error": f"{type(e).__name__}: {e}",
                "original": _data_url(image), "width": image.width, "height": image.height}
    annotated = draw_result(image, result, panel=False)
    payload = result.to_dict()
    payload.update(
        name=name,
        width=image.width,
        height=image.height,
        total_s=round(total, 3),
        original=_data_url(image),
        annotated=_data_url(annotated),
    )
    return payload


def _has_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch not installed
        return False


def _data_url(img, max_side: int = 1400) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class _Handler(BaseHTTPRequestHandler):
    default_engine = "auto"
    server_version = "nutrilabel-ui"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _same_origin(self) -> bool:
        """Refuse uploads posted by other web sites the user happens to visit."""
        origin = self.headers.get("Origin")
        if not origin:
            return True  # same-origin fetches from older browsers, curl, tests
        host = self.headers.get("Host", "")
        return urlparse(origin).netloc == host

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, page_html().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/engines":
            from nutrilabel.engines import available_engines

            engines = available_engines(include_vlm=True)
            classic = available_engines()
            default = self.default_engine if self.default_engine != "auto" else (
                classic[0] if classic else (engines[0] if engines else ""))
            self._json({"engines": engines, "classic": classic,
                        "vlm": [e for e in engines if e not in classic],
                        "gpu": _has_gpu(), "default": default})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/read":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._json({"error": "empty or too large upload (max 30 MB)"},
                       HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        if not self._same_origin():
            self._json({"error": "cross-origin request refused"}, HTTPStatus.FORBIDDEN)
            return
        from nutrilabel.engines import available_engines

        params = parse_qs(url.query)
        engine = params.get("engine", [self.default_engine])[0] or self.default_engine
        if engine != "auto" and engine not in available_engines(include_vlm=True):
            self._json({"error": f"engine {engine!r} is not installed"}, HTTPStatus.BAD_REQUEST)
            return
        name = params.get("name", ["image"])[0][:200]
        data = self.rfile.read(length)
        try:
            self._json(analyse_image(data, engine, name))
        except Exception as e:  # noqa: BLE001 - report the error to the page
            self._json({"name": name, "error": f"{type(e).__name__}: {e}"}, 500)


def serve(host: str = "127.0.0.1", port: int = 8765, engine: str = "auto",
          open_browser: bool = True) -> None:
    handler = type("Handler", (_Handler,), {"default_engine": engine})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"nutrilabel UI running on {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
