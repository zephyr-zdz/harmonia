"""Rule registry: config section name -> detector(ctx) -> list[Event].

Each rule is a pure function of an AnalysisContext and can be tested on its own
(see tests/test_rules.py, which builds contexts with harmonia.analysis.pipeline.build_context).
"""

from __future__ import annotations

from collections.abc import Callable

from ...schema import Event
from ..context import AnalysisContext
from . import borrowed, ii_v_i, patterns, secondary, tritone

RULES: dict[str, Callable[[AnalysisContext], list[Event]]] = {
    "ii_V_I": ii_v_i.detect,
    "secondary_dominant": secondary.detect_dominants,
    "secondary_leading_tone": secondary.detect_leading_tone,
    "tritone_sub": tritone.detect,
    "borrowed": borrowed.detect,
    "aeolian_cadence": patterns.detect_aeolian,
    "modulation": patterns.detect_modulations,
}

__all__ = ["RULES"]
