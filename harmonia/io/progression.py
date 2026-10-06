"""Symbolic chord-progression text -> RecognitionResult (time unit = beats).

Formats:
  * Bar notation:   ``| C | G | Am Em | F G |``  — chords in a bar share it equally;
                    ``.`` extends the previous chord by one slot (``| C . . G |`` = 3+1 beats);
                    ``%`` repeats the previous bar.
  * Plain list:     ``C G Am Em``  — one chord per bar.
  * ``[Verse]``-style markers set the section of following chords; ``//`` starts a comment,
    as does ``#`` at the start of a line (elsewhere ``#`` is a sharp).
Every chord token is validated: an unparseable symbol raises ChordParseError (with its bar),
unless on_error="flag", which keeps it as an X frame plus a warning.
"""

from __future__ import annotations

import re

from ..schema import ChordCandidate, Frame, RecognitionResult
from ..theory.chord import ChordParseError, parse_chord

_SECTION_RE = re.compile(r"\[([^\]]*)\]")


def _tokens(text: str) -> list[str]:
    # Comments: "//" anywhere, or "#" as the first non-blank character of a line
    # ("#" elsewhere is a sharp, e.g. C#dim7).
    lines = ["" if ln.lstrip().startswith("#") else ln.split("//", 1)[0] for ln in text.splitlines()]
    body = "\n".join(lines)
    body = _SECTION_RE.sub(lambda m: " §" + m.group(1).strip().replace(" ", "_") + " ", body)
    return body.replace("|", " | ").split()


def parse_progression(text: str, beats_per_bar: int = 4, on_error: str = "raise") -> RecognitionResult:
    if on_error not in ("raise", "flag"):
        raise ValueError("on_error must be 'raise' or 'flag'")
    toks = _tokens(text)
    bar_mode = "|" in toks

    # Group into bars of (token, section) items.
    bars: list[list[tuple[str, str | None]]] = []
    section: str | None = None
    current: list[tuple[str, str | None]] = []
    for t in toks:
        if t.startswith("§"):
            section = t[1:].replace("_", " ") or None
        elif t == "|":
            if current:
                bars.append(current)
            current = []
        elif not bar_mode:
            bars.append([(t, section)])
        elif t == "%":
            if current:
                raise ChordParseError("'%' must stand alone in its bar")
            if not bars:
                raise ChordParseError("'%' with no previous bar")
            current = [(tok, section) for tok, _ in bars[-1]]
        else:
            current.append((t, section))
    if current:
        bars.append(current)

    frames: list[Frame] = []
    warnings: list[str] = []
    time = 0.0
    for bi, bar in enumerate(bars, start=1):
        slot = beats_per_bar / len(bar)
        pos = 0.0
        for tok, sec in bar:
            if tok == ".":
                if not frames:
                    raise ChordParseError("'.' with no previous chord")
                frames[-1].duration += slot
                frames[-1].beats = frames[-1].duration
            else:
                # Symbolic bass lives in the chord symbol itself (slash / on-chord), so no
                # separate BassObservation is attached.
                try:
                    parse_chord(tok)
                except ChordParseError as e:
                    if on_error == "raise":
                        raise ChordParseError(f"bar {bi}: {e}") from None
                    warnings.append(f"bar {bi}: unparseable chord {tok!r} ({e})")
                frames.append(Frame(time=time + pos, duration=slot, beats=slot,
                                    candidates=[ChordCandidate(tok, 1.0)],
                                    bar=bi, beat_in_bar=pos + 1, section=sec))
            pos += slot
        time += beats_per_bar
    return RecognitionResult(frames=frames, time_unit="beat", beats_per_bar=beats_per_bar,
                             source={"type": "symbolic"}, warnings=warnings)
