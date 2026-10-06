# Intermediate JSON schema (v0.1.0)

Source of truth: `harmonia/schema.py` (dataclasses). This page explains the fields.
Bump `SCHEMA_VERSION` on any incompatible change.

## 1. RecognitionResult — recognition layer → analysis layer

```jsonc
{
  "schema_version": "0.1.0",
  "time_unit": "second",          // "second" (audio, .lab) | "beat" (symbolic input)
  "beats_per_bar": 4,
  "source": {"type": "audio", "path": "song.mp3"},
  "warnings": [],                 // recognition-side problems, passed through to the output
  "frames": [                     // time-ordered; audio: one per beat
    {
      "time": 12.48, "duration": 0.51,
      "beats": 1.0,               // beats this frame spans (weights the key model)
      "candidates": [             // top-k; probs need not sum to 1 (rest = "other")
        {"label": "F:maj7", "prob": 0.62},
        {"label": "A:min7", "prob": 0.21}
      ],
      "bass": {"note": "F", "prob": 0.8},       // optional, from bass transcription
      "key": {"label": "C major", "prob": 0.7}, // optional local-key guess (currently unused)
      "bar": 7, "beat_in_bar": 1, "section": "Chorus"
    }
  ]
}
```

Chord labels may be Harte (`Bb:hdim7`, `C:maj/3`) or pop/Japanese (`Bbm7-5`, `C(onE)`), see
`harmonia/theory/chord.py`. `N` = no chord, `X` = unknown. An unparseable candidate is dropped
with a warning; a frame with no parseable candidate becomes `X` with confidence 0.

## 2. AnalysisResult — analysis layer → UI / evaluation

| field | meaning |
|---|---|
| `global_key` | `{key, alternatives[], ambiguous, source}`; `source` = `estimated` or `given`. `key.prob` = share of posterior-weighted time. `ambiguous` = over half the beats have two competing keys (ratio ≥ `key.ambiguity_ratio`). |
| `key_regions[]` | `{start, end, segment_range:[a,b), key}` from the HMM path. |
| `segments[]` | merged chord spans, see below |
| `events[]` | harmonic events, see below |
| `warnings[]` | parse problems, key ambiguity, missing beat info… never silently dropped |

### Segment
`chord` (as written in the input), `chord_harte`, `candidates` (merged top-k), `root`, `bass`,
`key` (`label`, `tonic`, `mode`, `prob` = posterior), `roman` (`numeral`, `display` with
inversion e.g. `I/3`, `degree`, `diatonic`, `confidence` = chord prob × key prob),
`functions` (applied readings from events, e.g. `["V7/vi"]`), `event_ids`, `confidence`
(top candidate prob), `low_confidence` + `warnings` (why), `section`, `bar`.

### Event
```jsonc
{
  "id": "ev1", "type": "ii_V_I", "label": "ii–V–I → IV",
  "start": 12.0, "end": 24.0, "segment_indices": [3, 4, 5],
  "key": "C major",
  "chords":    ["Gm7", "C7", "Fmaj7"],
  "numerals":  ["vm7", "I7", "IVmaj7"],          // relative to the local key
  "functions": ["iim7/IV", "V7/IV", "IVmaj7"],   // functional reading
  "confidence": 0.9, "low_confidence": false,
  "rule": "ii_V_I",
  "evidence": [                                   // product of factors = confidence
    {"kind": "root_motion", "detail": "roots rise by a 4th twice (ii → V → I)", "factor": 1.0},
    {"kind": "target", "detail": "tonicises a diatonic chord of C major", "factor": 0.9}
  ],
  "attributes": {"mode": "major", "target": "IV", "tonicization": true},
  "explains": [3, 4],            // chromatic chords accounted for by this event
  "alternatives": [{"type": "borrowed_chord", "label": "vm7 (borrowed: …)", "confidence": 1.0}],
  "part_of": null                // e.g. a V7/IV inside this ii–V points here
}
```

Event types: `ii_V_I`, `secondary_dominant` (`attributes.resolution` = resolved / deceptive /
unresolved), `secondary_leading_tone`, `tritone_sub`, `borrowed_chord`
(`attributes.source`, `entry`, `backdoor`), `aeolian_cadence`, `modulation`
(`attributes.from`, `to`, `interval`, `semitones`; `start == end` = boundary time).
