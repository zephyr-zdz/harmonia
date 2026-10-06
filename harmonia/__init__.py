"""Harmonia: local harmonic analysis for pop music (J-pop / City Pop).

Recognition layer (audio -> RecognitionResult) and analysis layer
(RecognitionResult -> AnalysisResult) communicate only through harmonia.schema.
"""

from __future__ import annotations

from typing import Any

from .analysis import analyze
from .io import parse_progression
from .schema import AnalysisResult, RecognitionResult
from .theory.key import Key

__version__ = "0.1.0"


def analyze_progression(text: str, key: str | Key | None = None, config: dict[str, Any] | None = None,
                        beats_per_bar: int = 4) -> AnalysisResult:
    """Convenience: analyse a chord-progression string (see harmonia.io.progression)."""
    return analyze(parse_progression(text, beats_per_bar=beats_per_bar), config=config, key=key)


__all__ = ["AnalysisResult", "RecognitionResult", "analyze", "analyze_progression", "parse_progression"]
