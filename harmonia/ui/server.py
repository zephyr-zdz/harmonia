"""Local UI server (Phase 4): one HTML page + a tiny JSON API over a song LIBRARY.
stdlib only; binds 127.0.0.1.

    harmonia ui                          # library = [ui].library in config (data/external/user_music)
    harmonia ui DIR [DIR|FILE ...]       # audio, progression .txt, .lab, *.recognition.json

Songs whose recognition already exists (UI cache in outputs/cache, or a `harmonia batch` run in
outputs/runs/*) open instantly; others are transcribed in a background worker on first open
(one at a time) and cached.

API (every song-specific call takes the song id)
  GET  /                  -> index.html
  GET  /api/library       -> {items: [{id, name, group, kind, status}]}
                             status: ready | cached | new | queued | transcribing | error: ...
  POST /api/open {id}     -> {status}   (starts transcription if needed)
  GET  /api/state?id=     -> {id, name, has_audio, recognition, analysis}   (409 if not ready)
  GET  /audio?id=         -> audio bytes (HTTP Range supported)
  POST /api/analyze       {id, edits: [{start, end, chord}], key} -> {analysis}
  POST /api/export        {id, edits, key} -> writes outputs/annotations/<name>.{lab,analysis.json}
Edits replace the candidates of every frame whose midpoint lies in [start, end) with the
user's chord at probability 1.0.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sys
import threading
import traceback
import webbrowser
from copy import deepcopy
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..analysis import analyze
from ..config import load_config
from ..io import lab_to_recognition, parse_progression
from ..schema import ChordCandidate, RecognitionResult
from ..theory.chord import ChordParseError, parse_chord

ROOT = Path(__file__).resolve().parents[2]
AUDIO_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".opus"}
SYMBOLIC_SUFFIXES = {".txt", ".lab"}
_MIME = {".flac": "audio/flac", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
         ".ogg": "audio/ogg", ".opus": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff"}


@dataclass
class Item:
    id: str
    path: Path
    name: str
    group: str
    kind: str  # audio | symbolic | recognition


class Song:
    def __init__(self, item: Item, recognition: RecognitionResult, cfg: dict, key: str | None):
        self.item = item
        self.recognition = recognition
        self.cfg = cfg
        self.key = key
        self.audio: Path | None = item.path if item.kind == "audio" else None
        if self.audio is None:  # a saved RecognitionResult remembers its audio file
            p = Path(recognition.source.get("path") or "")
            if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file():
                self.audio = p
        self.lock = threading.Lock()

    def apply_edits(self, edits: list[dict]) -> RecognitionResult:
        rec = deepcopy(self.recognition)
        for e in edits:
            label = str(e["chord"]).strip()
            parse_chord(label)  # raises ChordParseError -> 400
            s, t = float(e["start"]), float(e["end"])
            for f in rec.frames:
                if s <= f.time + f.duration / 2 < t:
                    f.candidates = [ChordCandidate(label, 1.0)]
        return rec

    def analyse(self, edits: list[dict], key: str | None) -> dict[str, Any]:
        with self.lock:
            return analyze(self.apply_edits(edits), config=self.cfg, key=key or self.key).to_dict()


def _item_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]


class Library:
    def __init__(self, inputs: list[Path], cfg: dict, key: str | None = None):
        self.cfg = cfg
        self.key = key
        self.lock = threading.Lock()
        self.worker = threading.Lock()  # one transcription at a time
        self.songs: dict[str, Song] = {}
        self.status: dict[str, str] = {}
        self.runs = self._index_runs()
        self.items = self._scan(inputs)
        for it in self.items:
            self.status[it.id] = ("cached" if self._existing(it.path) else "new") if it.kind == "audio" else "cached"

    # ---- discovery -------------------------------------------------------------------
    def _scan(self, inputs: list[Path]) -> list[Item]:
        found: list[tuple[Path, str]] = []
        for inp in inputs:
            if inp.is_dir():
                for p in sorted(inp.rglob("*")):
                    if p.is_file() and (p.suffix.lower() in AUDIO_SUFFIXES or p.suffix.lower() in SYMBOLIC_SUFFIXES
                                        or p.name.endswith(".recognition.json")):
                        found.append((p, str(p.parent.relative_to(inp)) if p.parent != inp else inp.name))
            elif inp.is_file():
                found.append((inp, inp.parent.name))
            else:
                print(f"ui: skipping missing input {inp}", file=sys.stderr)
        audio = {p.resolve() for p, _ in found if p.suffix.lower() in AUDIO_SUFFIXES}
        items, seen = [], set()
        for p, group in found:
            if p.name.endswith(".recognition.json"):
                src = self._source_of(p)
                if src is not None and src in audio:
                    continue  # the audio itself is listed; its saved recognition is used as cache
                kind, name = "recognition", p.name[: -len(".recognition.json")]
            elif p.suffix.lower() in AUDIO_SUFFIXES:
                kind, name = "audio", p.stem
            else:
                kind, name = "symbolic", p.stem
            iid = _item_id(p)
            if iid not in seen:
                seen.add(iid)
                items.append(Item(iid, p.resolve(), name, group, kind))
        return items

    @staticmethod
    def _source_of(rec_json: Path) -> Path | None:
        try:
            src = json.loads(rec_json.read_text(encoding="utf-8")).get("source", {}).get("path")
            return Path(src).resolve() if src else None
        except (OSError, json.JSONDecodeError):
            return None

    def _index_runs(self) -> dict[Path, Path]:
        """audio path -> newest recognition.json written by `harmonia batch` under outputs/runs."""
        out: dict[Path, Path] = {}
        for f in sorted((ROOT / "outputs" / "runs").glob("*/*.recognition.json"), key=lambda p: p.stat().st_mtime):
            src = self._source_of(f)
            if src is not None:
                out[src] = f
        return out

    def _cache_path(self, path: Path) -> Path:
        st = path.stat()
        fe = json.dumps(self.cfg.get("frontend", {}), sort_keys=True)
        h = hashlib.sha256(f"{path.resolve()}|{st.st_size}|{st.st_mtime}|{fe}".encode()).hexdigest()[:16]
        return ROOT / "outputs" / "cache" / f"{path.stem}.{h}.recognition.json"

    def _existing(self, path: Path) -> Path | None:
        c = self._cache_path(path)
        if c.is_file():
            return c
        return self.runs.get(path.resolve())

    # ---- loading ---------------------------------------------------------------------
    def get_item(self, iid: str) -> Item:
        for it in self.items:
            if it.id == iid:
                return it
        raise KeyError(f"unknown song id {iid!r}")

    def _load_now(self, it: Item) -> RecognitionResult | None:
        if it.kind == "recognition":
            return RecognitionResult.from_dict(json.loads(it.path.read_text(encoding="utf-8")))
        if it.kind == "symbolic":
            if it.path.suffix == ".lab":
                return lab_to_recognition(it.path)
            return parse_progression(it.path.read_text(encoding="utf-8"), on_error="flag")
        ex = self._existing(it.path)
        if ex is not None:
            return RecognitionResult.from_dict(json.loads(ex.read_text(encoding="utf-8")))
        return None

    def _transcribe(self, it: Item) -> None:
        with self.worker:
            with self.lock:
                self.status[it.id] = "transcribing"
            try:
                from ..frontend.pipeline import transcribe  # optional 'audio' extra
                print(f"ui: transcribing {it.path.name} ...", file=sys.stderr)
                rec = transcribe(it.path, self.cfg)
                cache = self._cache_path(it.path)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")
                with self.lock:
                    self.songs[it.id] = Song(it, rec, self.cfg, self.key)
                    self.status[it.id] = "ready"
            except Exception as e:  # surfaced to the UI, never swallowed
                traceback.print_exc()
                with self.lock:
                    self.status[it.id] = f"error: {e!r}"

    def open(self, iid: str) -> str:
        it = self.get_item(iid)
        with self.lock:
            st = self.status[iid]
            if st == "ready" or st in ("queued", "transcribing"):
                return st
        rec = self._load_now(it)
        if rec is not None:
            with self.lock:
                self.songs[iid] = Song(it, rec, self.cfg, self.key)
                self.status[iid] = "ready"
            return "ready"
        with self.lock:
            self.status[iid] = "queued"
        threading.Thread(target=self._transcribe, args=(it,), daemon=True).start()
        return "queued"

    def song(self, iid: str) -> Song:
        with self.lock:
            s = self.songs.get(iid)
        if s is None:
            raise LookupError(self.status.get(iid, "unknown"))
        return s

    def listing(self) -> list[dict]:
        with self.lock:
            return [{"id": it.id, "name": it.name, "group": it.group, "kind": it.kind, "status": self.status[it.id]}
                    for it in self.items]


def _make_handler(lib: Library):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if args and "/api/" in str(args[0]) and "/api/library" not in str(args[0]):
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
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    body = resources.files("harmonia.ui").joinpath("index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/library":
                    self._json({"items": lib.listing()})
                elif url.path == "/api/state":
                    s = lib.song(q["id"])
                    self._json({"id": q["id"], "name": s.item.name, "has_audio": s.audio is not None,
                                "recognition": s.recognition.to_dict(), "analysis": s.analyse([], None)})
                elif url.path == "/audio":
                    s = lib.song(q["id"])
                    if s.audio is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                    else:
                        self._send_audio(s.audio)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except LookupError as e:
                self._json({"error": "song not loaded", "status": str(e)}, 409)
            except KeyError as e:
                self._json({"error": str(e)}, 404)

        def _send_audio(self, path: Path) -> None:
            size = path.stat().st_size
            ctype = _MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(size - int(b), 0)
                end = min(int(b) if (a and b) else size - 1, size - 1)
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
                if self.path == "/api/open":
                    self._json({"status": lib.open(req["id"])})
                    return
                s = lib.song(req["id"])
                edits, key = req.get("edits", []), req.get("key")
                if self.path == "/api/analyze":
                    self._json({"analysis": s.analyse(edits, key)})
                elif self.path == "/api/export":
                    self._json(_export(s, edits, key))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except ChordParseError as e:
                self._json({"error": f"chord: {e}"}, 400)
            except LookupError as e:
                self._json({"error": "song not loaded", "status": str(e)}, 409)
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def _export(s: Song, edits: list[dict], key: str | None) -> dict:
    res = analyze(s.apply_edits(edits), config=s.cfg, key=key or s.key)
    out = ROOT / "outputs" / "annotations"
    out.mkdir(parents=True, exist_ok=True)
    lab = out / f"{s.item.name}.lab"
    lab.write_text("".join(f"{x.start:.3f}\t{x.end:.3f}\t{x.chord_harte}\n" for x in res.segments), encoding="utf-8")
    aj = out / f"{s.item.name}.analysis.json"
    aj.write_text(res.to_json(), encoding="utf-8")
    return {"lab": str(lab.relative_to(ROOT)), "analysis": str(aj.relative_to(ROOT)), "n_edits": len(edits)}


def serve(inputs: list[str | Path] | None = None, port: int = 8765, key: str | None = None,
          config: str | None = None, open_browser: bool = True) -> None:
    cfg = load_config(config)
    paths = [Path(p) for p in (inputs or [])] or [ROOT / p for p in cfg["ui"]["library"]]
    lib = Library(paths, cfg, key)
    if not lib.items:
        raise SystemExit(f"no songs found in {[str(p) for p in paths]}")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(lib))
    url = f"http://127.0.0.1:{port}/"
    print(f"Harmonia UI: {url}  ({len(lib.items)} songs; Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
