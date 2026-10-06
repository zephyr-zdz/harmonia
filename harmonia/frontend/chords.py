"""Frame-level chord evidence backends.

A backend returns per-frame log-likelihoods over a chord vocabulary plus per-frame bass
probabilities. Decoding (beat-synchronous HMM, top-k) lives in decode.py so that backends
are interchangeable.

"lv_chordia": ISMIR 2019 large-vocabulary model (Jiang et al., MIT), 5-model ensemble on CQT.
    We tap its chord-structure-decomposition heads and its own observation function
    (XHMMDecoder.get_chord_tag_obs) instead of its final Viterbi labels, so we keep soft
    evidence for top-k candidates. Runs on CPU (the wrapper refuses MPS).
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameEvidence:
    times: np.ndarray          # frame centre times (s), shape [T]
    labels: list[str]          # chord vocabulary (Harte), length C; "N" allowed
    loglik: np.ndarray         # [T, C] unnormalised log-likelihoods
    bass: np.ndarray | None    # [T, 13] probabilities: index 0 = no bass, 1 + pitch class
    duration: float
    backend: str


_ENSEMBLE = None


def frame_evidence(audio_path: str | Path, cfg: dict) -> FrameEvidence:
    fc = cfg["frontend"]["chords"]
    if fc["backend"] == "lv_chordia":
        return _lv_chordia(str(audio_path), fc)
    raise ValueError(f"unknown chord backend {fc['backend']!r}")


def _lv_chordia(audio_path: str, fc: dict) -> FrameEvidence:
    global _ENSEMBLE
    import contextlib
    import io as _io

    from lv_chordia.chord_recognition import load_ensemble
    from lv_chordia.extractors.cqt import CQTV2
    from lv_chordia.extractors.xhmm_ismir import XHMMDecoder
    from lv_chordia.mir import DataEntry, io
    from lv_chordia.settings import DEFAULT_HOP_LENGTH, DEFAULT_SR

    if _ENSEMBLE is None:
        _ENSEMBLE = load_ensemble(False)
    entry = DataEntry()
    entry.prop.set("sr", DEFAULT_SR)
    entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
    entry.append_file(audio_path, io.MusicIO, "music")
    entry.append_extractor(CQTV2, "cqt")
    cqt = entry.cqt
    with contextlib.redirect_stderr(_io.StringIO()):  # silence per-model progress prints
        outs = [net.inference(cqt) for net in _ENSEMBLE]
    probs = [np.mean([o[i] for o in outs], axis=0) for i in range(len(outs[0]))]
    with importlib.resources.path("lv_chordia.data", f"{fc['vocabulary']}_chord_list.txt") as d:
        hmm = XHMMDecoder(template_file=str(d))
    labels, loglik = hmm.get_chord_tag_obs(probs)
    hop = DEFAULT_HOP_LENGTH / DEFAULT_SR
    T = loglik.shape[0]
    return FrameEvidence(times=(np.arange(T) + 0.5) * hop, labels=list(labels),
                         loglik=np.asarray(loglik, dtype=np.float64), bass=np.asarray(probs[1], dtype=np.float64),
                         duration=T * hop, backend=f"lv_chordia/{fc['vocabulary']}")
