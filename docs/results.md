# Results log

Every reported number states split, data, git state and what changed. Tuning uses dev only;
test numbers are looked at once per decision.

## 2026-10-06 — analysis layer on ChoCo expert annotations (symbolic input, no audio)

Data: ChoCo v1.0.0 (Zenodo 7706751). Split = sha256(song id) → 60 % dev / 40 % test
(`harmonia.eval.choco.split_of`). rock-corpus = de Clercq & Temperley (jams-converted; keys are
tonic-only → tonic accuracy is the meaningful key metric there); isophonics / robbie-williams
have time-varying keys with mode. Metrics defined in `harmonia/eval/choco.py`.

| split | config | rock key | rock tonic | rock numeral deg. | applied R / P | iso key | robbie key |
|---|---|---|---|---|---|---|---|
| dev (112/126/29) | baseline (Phase 1 priors) | 0.498 | 0.601 | 0.567 | 0.64 / 0.18 | 0.652 | 0.744 |
| dev | **cadence_triad_factor 0.3** | 0.584 | 0.684 | 0.629 | 0.69 / 0.22 | 0.696 | 0.829 |
| test (88/99/33) | baseline | 0.663 | 0.770 | 0.704 | 0.76 / 0.20 | 0.684 | 0.765 |
| test | **cadence_triad_factor 0.3** | 0.701 | 0.808 | 0.750 | 0.76 / 0.24 | 0.736 | 0.763 |

Dev error analysis (baseline): most global-key errors put the key a 4th too high
(rock 29/112, isophonics 28/126) — I→IV of the true key looks like V→I of its subdominant,
and ♭VII-heavy (Mixolydian) rock is diatonic in that subdominant key.

Tried on dev and NOT adopted (conflict with J-pop decisions encoded in unit tests):
* boundary tonic bonus (first/last chord): dev rock tonic up to 0.82, but 丸サ / 王道 loops
  open on IV and the bonus moves them to the IV key → breaks decisions Q1/Q2. Kept, default off. [ASK]
* ♭VII penalty −1.2 → −0.6: small dev gain; makes F–G–Em–Am read as G Mixolydian (breaks Q2).

Open issue: applied-dominant precision ≈ 0.2 — many predicted V/x where experts write a plain
numeral (e.g. rock "II", unresolved). To investigate on dev.
