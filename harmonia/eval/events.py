"""Harmonic-event matching: precision / recall / F1 per event type, and error attribution.

Matching: a prediction matches a gold event of the SAME TYPE when their time spans overlap
by at least ``min_overlap`` of the shorter span (point events such as modulations match
within ``point_tolerance``). Greedy, highest-confidence prediction first, one-to-one.
Label correctness (e.g. "V7/vi" vs "V7/ii") is scored separately among matched pairs.

Attribution (needs gold events): the analysis layer is run twice — on the reference chords
("oracle") and on the recognised chords ("system").
    gold event found by oracle and system   -> ok
    found by oracle, missed by system       -> recognition_miss
    missed by oracle                        -> analysis_miss
    system false positive also produced by oracle  -> analysis_fp
    system false positive not produced by oracle   -> recognition_fp
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .dataset import GoldEvent


@dataclass
class Span:
    type: str
    start: float
    end: float
    label: str | None = None
    confidence: float = 1.0


@dataclass
class MatchResult:
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (gold_idx, pred_idx)
    missed: list[int] = field(default_factory=list)
    false_pos: list[int] = field(default_factory=list)


def _overlap_ok(a: Span, b: Span, min_overlap: float, tol: float) -> float:
    la, lb = a.end - a.start, b.end - b.start
    if la <= 0 or lb <= 0:
        d = abs(a.start - b.start)
        return 1.0 - d / (tol + 1e-9) if d <= tol else -1.0
    inter = min(a.end, b.end) - max(a.start, b.start)
    ratio = inter / min(la, lb)
    return ratio if ratio >= min_overlap else -1.0


def match(gold: list[Span], pred: list[Span], min_overlap: float = 0.5,
          point_tolerance: float = 2.0) -> MatchResult:
    res = MatchResult()
    used: set[int] = set()
    for pi in sorted(range(len(pred)), key=lambda i: -pred[i].confidence):
        p = pred[pi]
        best, best_q = None, -1.0
        for gi, g in enumerate(gold):
            if gi in used or g.type != p.type:
                continue
            q = _overlap_ok(g, p, min_overlap, point_tolerance)
            if q > best_q:
                best, best_q = gi, q
        if best is None or best_q < 0:
            res.false_pos.append(pi)
        else:
            used.add(best)
            res.pairs.append((best, pi))
    res.missed = [gi for gi in range(len(gold)) if gi not in used]
    return res


def prf(gold: list[Span], pred: list[Span], **kw: Any) -> dict[str, Any]:
    """Micro P/R/F1 overall and per type, plus label accuracy among matches."""
    def score(g: list[Span], p: list[Span]) -> dict[str, Any]:
        m = match(g, p, **kw)
        tp = len(m.pairs)
        prec = tp / len(p) if p else None
        rec = tp / len(g) if g else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        labelled = [(gi, pi) for gi, pi in m.pairs if g[gi].label]
        lab_acc = (sum(1 for gi, pi in labelled if g[gi].label == p[pi].label) / len(labelled)) if labelled else None
        return {"tp": tp, "n_gold": len(g), "n_pred": len(p), "precision": prec, "recall": rec, "f1": f1,
                "label_accuracy": lab_acc}

    out = {"overall": score(gold, pred), "by_type": {}}
    for t in sorted({x.type for x in gold} | {x.type for x in pred}):
        out["by_type"][t] = score([x for x in gold if x.type == t], [x for x in pred if x.type == t])
    return out


def attribute(gold: list[Span], oracle: list[Span], system: list[Span], **kw: Any) -> dict[str, Any]:
    mo = match(gold, oracle, **kw)
    ms = match(gold, system, **kw)
    o_found = {gi for gi, _ in mo.pairs}
    s_found = {gi for gi, _ in ms.pairs}
    cats: Counter[str] = Counter()
    details = []
    for gi, g in enumerate(gold):
        if gi in o_found and gi in s_found:
            c = "ok"
        elif gi in o_found:
            c = "recognition_miss"
        elif gi in s_found:
            c = "analysis_miss_but_system_hit"
        else:
            c = "analysis_miss"
        cats[c] += 1
        if c != "ok":
            details.append({"category": c, "type": g.type, "label": g.label, "start": g.start, "end": g.end})
    oracle_fp = [oracle[pi] for pi in mo.false_pos]
    for pi in ms.false_pos:
        p = system[pi]
        same = match(oracle_fp, [p], **kw)
        c = "analysis_fp" if same.pairs else "recognition_fp"
        cats[c] += 1
        details.append({"category": c, "type": p.type, "label": p.label, "start": p.start, "end": p.end,
                        "confidence": p.confidence})
    return {"counts": dict(cats), "details": details}


def gold_spans(gold: list[GoldEvent]) -> list[Span]:
    return [Span(g.type, g.start, g.end, g.label) for g in gold]
