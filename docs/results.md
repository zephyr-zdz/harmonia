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
  open on IV and the bonus moves them to the IV key → breaks decisions Q1/Q2. (Switched on 2026-10-07 with idiom exemption, below.)
* ♭VII penalty −1.2 → −0.6: small dev gain; makes F–G–Em–Am read as G Mixolydian (breaks Q2).

Open issue (addressed 2026-10-07, below): applied-dominant precision ≈ 0.2 — many predicted V/x where experts write a plain
numeral (e.g. rock "II", unresolved). To investigate on dev.

## 2026-10-07 — boundary tonic bonus ON, graded deceptive cadence, stricter applied chords

Same data / split / metrics as above. Changes (user decisions 2026-10-07, see CLAUDE.md):
1. `key.boundary_tonic_bonus` 0 → 3.0, with chords inside or restarting a named progression
   exempt (`idioms.loop_segments`) — this resolves the earlier conflict with 丸サ / 王道.
2. Unresolved `ii_V` of a non-tonic target needs a chromatic V (P ≥ 0.5) and a minor-family ii
   (P ≥ 0.5); unresolved secondary dominants of a non-diatonic target (V/♭VII …) are dropped.
3. Deceptive cadence: IV → V (triad) → vi gets 0.5, IV → V7 → vi 0.8, ii → V → vi 1.0.

Dev sweep of the bonus (rule changes 2–3 not yet applied; rock tonic / iso key / robbie key):
0 → .684/.696/.829, 1 → .698/.712/.850, 2 → .714/.745/.850, **3 → .770/.758/.850**,
4 → .770/.773/.850, 6 → .777/.786/.847, 10 → .779/.787/.848; "last chord only" 3 → .706/.718/.820.
Chose 3.0: most of the gain, smallest value on the plateau (a larger bonus lets one chord
override more of the song).

| split | config | rock key | rock tonic | rock numeral deg. | applied R / P | iso key | robbie key |
|---|---|---|---|---|---|---|---|
| dev | previous (cadence_triad_factor 0.3) | 0.584 | 0.684 | 0.629 | 0.69 / 0.22 | 0.696 | 0.829 |
| dev | + boundary bonus 3.0 | 0.638 | 0.770 | 0.689 | 0.69 / 0.23 | 0.758 | 0.850 |
| dev | **+ applied-chord rules (2)** | 0.638 | 0.770 | 0.689 | 0.69 / 0.29 | 0.758 | 0.850 |
| test | previous | 0.701 | 0.808 | 0.750 | 0.76 / 0.25 | 0.736 | 0.763 |
| test | **all 2026-10-07 changes** | 0.716 | 0.829 | 0.780 | 0.76 / 0.28 | 0.767 | 0.815 |

Remaining applied false positives on dev (rock): mostly key errors (one song read in D♭ instead
of F contributes 70 spans; "V/V" where the key is a 5th low) and notation differences (rock
writes v–I–IV where we write ii–V–I/IV). Not rule problems; no further rule changes.

User recordings (outputs/runs/user_music_v1, no references → counts only, not accuracy):
deceptive cadences 44 → 38, of which 36 now low-confidence; secondary dominants 57 → 50
(21 low-confidence); unresolved ii–V 11 → 6.

## 2026-10-07 — first real-audio evaluation: IdolSongsJp (15 J-pop idol songs)

Data: IdolSongsJp (Suda et al., ISMIR 2025; HF `imprt/idol-songs-jp`, gated, evaluation only —
training prohibited by its license). Plain mixes `master_48k32b_-9LUFS`, expert chord + time-varying
key annotations. Imported by `python -m harmonia.eval.idolsongsjp`; split by `choco.split_of`:
dev = f04 f05 f06 f07 m02 m03 m04 m07, test = f01 f02 f03 f08 m01 m05 m06.
System: front end option A (Beat This! + lv-chordia + our decoder, config as committed in
4d92908, **nothing tuned on this corpus**), recognitions in `outputs/runs/idolsongsjp_v1`.
Command: `harmonia eval --split <dev|test> --estimates outputs/runs/idolsongsjp_v1`.
Oracle = analysis layer on the reference chords (no recognition errors).

| split | root | majmin | sevenths | CER majmin | numeral agree strict / degree | global key = oracle | local key weighted / exact: oracle | system |
|---|---|---|---|---|---|---|---|---|
| dev (8) | 0.807 | 0.812 | 0.623 | 0.299 | 0.540 / 0.794 | 8/8 | 0.950 / 0.932 | 0.932 / 0.915 |
| test (7, looked at once) | 0.788 | 0.771 | 0.594 | 0.336 | 0.446 / 0.632 | 6/7 | 0.678 / 0.553 | 0.697 / 0.568 |

Events system vs oracle (dev): P 0.66 / R 0.48 / F1 0.56 — modulation 0.88, aeolian 0.86,
secondary dominant 0.61, ii–V–I 0.61, borrowed 0.52 (recall 0.37), deceptive 0.33, tritone 0.
No gold events yet (IdolSongsJp has none), so these measure recognition-induced change only.

Findings:
* Strict vs degree numeral agreement: the recogniser mostly drops 7ths / 9ths (maj7(9) → maj);
  degree agreement ≈ root accuracy, i.e. numerals track recognition, not analysis.
* Dev: the key model is excellent on real J-pop (local 0.95 with reference chords; modulations
  up a semitone in f04 / m07 and the 5-key m04 found). One system error: m07's D♭ section
  read as F♯ (= G♭, a 4th up — the known subdominant confusion).
* Test: 3/7 songs are **relative major / minor confusions** (m01 C major → A minor; m05 A minor →
  oracle C major, system correct; m06 B major → G♯ minor). Dev had none. This is the
  relative-key ambiguity of decision Q2 (mild major prior). Not tuned on test — needs a
  theory-level answer from the user / more dev material.

## 2026-10-07 — minor leading-tone bonus (user request): NOT adopted

`key.minor_leading_tone_bonus` = extra minor-key evidence per V / V7 / vii°(7). Dev only:

| bonus | rock tonic | iso key | robbie key | IdolSongsJp dev local key: oracle | system |
|---|---|---|---|---|---|
| **0.0** | 0.770 | 0.758 | 0.850 | 0.950 | 0.932 |
| 0.3 | 0.775 | 0.752 | 0.847 | 0.950 | 0.929 |
| 0.6 | 0.775 | 0.744 | 0.845 | 0.950 | 0.898 |
| 1.0 | 0.774 | 0.702 | 0.839 | 0.950 | 0.894 |
| 1.5 | 0.738 | 0.672 | 0.804 | 0.873 | 0.892 |

Monotone loss: the leading-tone chord is just as often III(7) = V/vi of the relative major
(J-pop III7→vim), which the emission table already scores (+1.5 for minor). Kept at 0 (off).

## 2026-10-07 — decoder calibration on IdolSongsJp dev (user request)

`python -m harmonia.frontend.calibrate --out outputs/calibration/decode_v1` — 192-point grid
(obs_weight 0.5–2, change_penalty 1–4, change_extra_offbeat 0.5–3, new `seventh_bias` 0–2) on
the 8 dev songs, network features cached, objective fixed in advance: mean(majmin, sevenths)
with local key ≥ default − 0.01. Best on the grid: obs 1.5, change 1.0, offbeat 0.5, 7th bias 1.0.
Two values were on the grid edge; a refinement beyond them (`decode_v1_edge`, change 0.25–1,
offbeat 0–0.5, bias 0.75–1.25) gained ≤ 0.003 — flat plateau, noise for 8 songs — so the main
grid point was kept. seventh_bias 2.0 over-predicts 7ths (sevenths 0.44 at obs 0.5).

Recognitions re-run with the new defaults: `outputs/runs/idolsongsjp_v2`.

| split | run | root | majmin | sevenths | seg | CER majmin | numeral strict / degree | local key system | events F1 vs oracle |
|---|---|---|---|---|---|---|---|---|---|
| dev | v1 (old defaults) | 0.807 | 0.812 | 0.623 | 0.819 | 0.299 | 0.540 / 0.794 | 0.932 | 0.555 |
| dev | **v2 (calibrated)** | 0.818 | 0.826 | 0.664 | 0.847 | 0.273 | 0.578 / 0.819 | 0.952 | 0.603 |
| test | v1 | 0.788 | 0.771 | 0.594 | 0.776 | 0.336 | 0.446 / 0.632 | 0.697 | 0.499 |
| test | **v2** | 0.799 | 0.784 | 0.635 | 0.802 | 0.303 | 0.440 / 0.603 | 0.823 | 0.534 |

Test, looked at once: every chord metric improves, consistent with dev. The test local-key jump
(0.697 → 0.823) is ONE song (m06: 0.30 → 1.00, while its global key still reads G♯ minor) —
luck, not a calibrated effect. Test numeral agreement drops slightly because it is measured
against the oracle, whose keys are wrong on m01 / m05 / m06 (relative major/minor; open issue).
Noticed (not tuned): m06's global key (posterior mass) disagrees with its Viterbi local keys.
