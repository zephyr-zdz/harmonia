"""Local UI server (Phase 4): one HTML page + a tiny JSON API. stdlib only; binds 127.0.0.1.

    harmonia ui song.flac            # transcribe (cached) + analyse, open the timeline
    harmonia ui progression.txt      # symbolic input, no audio

API
  GET  /                -> index.html
  GET  /api/state       -> {name, has_audio, recognition, analysis}
  GET  /audio           -> the audio file (HTTP Range supported, for seeking)
  POST /api/analyze     {edits: [{start, end, chord}], key: null | "C major"} -> {analysis}
                        edits replace the chord candidates of every frame whose midpoint
                        lies in [start, end) with the user's chord at probability 1.0
  POST /api/export      {edits, key} -> writes outputs/annotations/<name>.{lab,analysis.json}
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sys
import threading
import webbrowser
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from ..analysis import analyze
from ..config import load_config
from ..io import lab_to_recognition, parse_progression
from ..schema import ChordCandidate, RecognitionResult
from ..theory.chord import ChordParseError, parse_chord

ROOT = Path(__file__).resolve().parents[2]
AUDIO_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".opus"}
_MIME = {".flac": "audio/flac", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
         ".ogg": "audio/ogg", ".opus": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff"}


class UIState:
    def __init__(self, src: Path, cfg: dict, key: str | None = None):
        self.src = src
        self.cfg = cfg
        self.key = key
        self.audio: Path | None = src if src.suffix.lower() in AUDIO_SUFFIXES else None
        self.recognition = self._load_recognition()
        if self.audio is None:  # a saved RecognitionResult remembers its audio file
            p = Path(self.recognition.source.get("path") or "")
            if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file():
                self.audio = p
        self.lock = threading.Lock()

    def _cache_path(self) -> Path:
        st = self.src.stat()
        fe = json.dumps(self.cfg.get("frontend", {}), sort_keys=True)
        h = hashlib.sha256(f"{self.src.resolve()}|{st.st_size}|{st.st_mtime}|{fe}".encode()).hexdigest()[:16]
        return ROOT / "outputs" / "cache" / f"{self.src.stem}.{h}.recognition.json"

    def _load_recognition(self) -> RecognitionResult:
        if self.audio is not None:
            cache = self._cache_path()
            if cache.is_file():
                return RecognitionResult.from_dict(json.loads(cache.read_text(encoding="utf-8")))
            from ..frontend.pipeline import transcribe  # optional 'audio' extra
            print(f"transcribing {self.src.name} (cached afterwards) ...", file=sys.stderr)
            rec = transcribe(self.src, self.cfg)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")
            return rec
        if self.src.suffix == ".lab":
            return lab_to_recognition(self.src)
        if self.src.suffix == ".json":
            return RecognitionResult.from_dict(json.loads(self.src.read_text(encoding="utf-8")))
        return parse_progression(self.src.read_text(encoding="utf-8"), on_error="flag")

    def apply_edits(self, edits: list[dict]) -> RecognitionResult:
        rec = deepcopy(self.recognition)
        for e in edits:
            label = str(e["chord"]).strip()
            parse_chord(label)  # raises ChordParseError -> 400
            s, t = float(e["start"]), float(e["end"])
            for f in rec.frames:
                mid = f.time + f.duration / 2
                if s <= mid < t:
                    f.candidates = [ChordCandidate(label, 1.0)]
        return rec

    def analyse(self, edits: list[dict], key: str | None) -> dict[str, Any]:
        rec = self.apply_edits(edits)
        return analyze(rec, config=self.cfg, key=key or self.key).to_dict()


def _make_handler(state: UIState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
            if "/api/" in (args[0] if args else ""):
                sys.stderr.write("ui: " + fmt % args + "\n")

        def _json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = resources.files("harmonia.ui").joinpath("index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                with state.lock:
                    analysis = state.analyse([], None)
                self._json({"name": state.src.name, "has_audio": state.audio is not None,
                            "recognition": state.recognition.to_dict(), "analysis": analysis,
                            "key": state.key})
            elif self.path == "/audio" and state.audio is not None:
                self._send_audio()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_audio(self) -> None:
            path = state.audio
            size = path.stat().st_size
            ctype = _MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(size - int(b), 0)
                end = int(b) if (a and b) else size - 1
                end = min(end, size - 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def do_POST(self) -> None:
            try:
                req = self._body()
                edits, key = req.get("edits", []), req.get("key")
                with state.lock:
                    if self.path == "/api/analyze":
                        self._json({"analysis": state.analyse(edits, key)})
                    elif self.path == "/api/export":
                        self._json(_export(state, edits, key))
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
            except ChordParseError as e:
                self._json({"error": f"chord: {e}"}, 400)
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def _export(state: UIState, edits: list[dict], key: str | None) -> dict:
    rec = state.apply_edits(edits)
    res = analyze(rec, config=state.cfg, key=key or state.key)
    out = ROOT / "outputs" / "annotations"
    out.mkdir(parents=True, exist_ok=True)
    lab = out / f"{state.src.stem}.lab"
    lines = [f"{s.start:.3f}\t{s.end:.3f}\t{s.chord_harte}" for s in res.segments]
    lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
    aj = out / f"{state.src.stem}.analysis.json"
    aj.write_text(res.to_json(), encoding="utf-8")
    return {"lab": str(lab.relative_to(ROOT)), "analysis": str(aj.relative_to(ROOT)), "n_edits": len(edits)}


def serve(src: str | Path, port: int = 8765, key: str | None = None, config: str | None = None,
          open_browser: bool = True) -> None:
    state = UIState(Path(src), load_config(config), key)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(state))
    url = f"http://127.0.0.1:{port}/"
    print(f"Harmonia UI: {url}  (Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
