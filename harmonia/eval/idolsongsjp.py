"""Import IdolSongsJp (Suda et al., ISMIR 2025) into the data/eval layout.

    python -m harmonia.eval.idolsongsjp            # writes data/eval/{dev,test}/<song_id>/

Raw corpus (gitignored, never modified): data/external/idolsongsjp, fetched from HuggingFace
``imprt/idol-songs-jp`` (gated). Only the plain mixes ``master_48k32b_-9LUFS/<id>.wav`` plus
``chords/<id>.txt`` and ``keys/<id>.txt`` are used. Both annotation files are already
time-aligned Harte ``start end label`` rows, so they are copied verbatim as chords.lab /
keys.lab (time-varying key reference) next to a meta.toml with provenance.

LICENSE: non-commercial research; using the corpus for model TRAINING is prohibited.
Here it is evaluation data only (no fitting of model weights; config calibration on dev only).

Split: the same stable hash as ChoCo (``choco.split_of``: sha256(song id) → 60 % dev / 40 % test).
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from .choco import split_of
from .dataset import key_label_to_mir_eval

RAW = Path("data/external/idolsongsjp")
AUDIO_DIR = "master_48k32b_-9LUFS"
SOURCE = "IdolSongsJp corpus (HuggingFace imprt/idol-songs-jp; Suda et al., ISMIR 2025)"
LICENSE = "IdolSongsJp license: non-commercial research; model TRAINING prohibited -> evaluation only"


def read_spans(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 3:
            out.append((float(p[0]), float(p[1]), p[2]))
    return out


def global_key(keys: list[tuple[float, float, str]]) -> str:
    """Key with the most annotated time (home key of a modulating song)."""
    dur: dict[str, float] = defaultdict(float)
    for s, e, k in keys:
        dur[k] += e - s
    return key_label_to_mir_eval(max(dur, key=dur.__getitem__))


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def import_corpus(raw: Path = RAW, out_root: Path = Path("data/eval")) -> dict[str, list[str]]:
    if not (raw / "chords").is_dir():
        raise FileNotFoundError(f"{raw}/chords missing — download the corpus first (see CLAUDE.md)")
    splits: dict[str, list[str]] = {"dev": [], "test": []}
    for chords in sorted((raw / "chords").glob("*.txt")):
        sid = chords.stem
        keys, audio = raw / "keys" / f"{sid}.txt", raw / AUDIO_DIR / f"{sid}.wav"
        if not keys.is_file() or not audio.is_file():
            raise FileNotFoundError(f"{sid}: needs {keys} and {audio}")
        split = split_of(sid)
        d = out_root / split / sid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chords, d / "chords.lab")
        shutil.copyfile(keys, d / "keys.lab")
        meta = [
            f"title = {_toml_str(sid)}",
            f"source = {_toml_str(SOURCE)}",
            f"license = {_toml_str(LICENSE)}",
            f"audio = {_toml_str(str(audio))}",
            f"key = {_toml_str(global_key(read_spans(keys)))}   # longest-held key; keys.lab = time-varying",
            f"raw_chords = {_toml_str(str(chords))}",
            f"raw_keys = {_toml_str(str(keys))}",
        ]
        (d / "meta.toml").write_text("\n".join(meta) + "\n", encoding="utf-8")
        splits[split].append(sid)
    return splits


if __name__ == "__main__":
    for split, ids in import_corpus().items():
        print(f"{split}: {len(ids)}  {' '.join(ids)}")
