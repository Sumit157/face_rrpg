# Methodology

## rPPG extraction methods

| Method | Reference | Idea |
|---|---|---|
| Green | Verkruysse et al., 2008 | Use the raw green channel; haemoglobin absorbs green light most strongly, so it carries the strongest plethysmographic AC component. Simplest, most replicated, most motion-sensitive. |
| CHROM | de Haan & Jeanne, 2013 | Project RGB onto two chrominance axes designed to cancel specular reflection / illumination-driven intensity changes, then combine using their std-dev ratio to suppress remaining motion. |
| POS | Wang et al., 2017 ("Algorithmic Principles of Remote PPG") | Project temporally-normalised RGB onto the plane orthogonal to the assumed skin-tone/illumination vector, computed over a sliding ~1.6s window. Tends to generalise better across skin tones/lighting than CHROM. Used as the default method. |

All three are implemented in `services/ml/app/rppg/` and share the same downstream
preprocessing (`filtering.py`): smoothness-priors detrending (Tarvainen et al., 2002) →
z-score normalisation → Butterworth band-pass restricted to 0.7–4.0 Hz (42–240 BPM).

## Signal quality

Signal-to-noise ratio is computed as the fraction of spectral power (Welch's method PSD)
within ±0.1 Hz of the dominant peak and its first harmonic, versus everything else in the
0.7–4.0 Hz band. This SNR (dB) is mapped to a 0–100 "quality score" via a clipped linear scale
calibrated so roughly −10dB → 0 and +12dB → 100. **This mapping is a documented heuristic, not
a clinically validated scale** — see `app/rppg/quality.py` and `docs/limitations.md`.

## Cross-region physiological consistency

Real facial blood flow pulses near-synchronously across the forehead and both cheeks (they
share the same cardiac cycle). `app/features/consistency.py` computes pairwise Pearson
correlation between each region's preprocessed rPPG signal, plus whether each pair's dominant
frequency agrees within 0.15 Hz. Low correlation/frequency agreement is treated as evidence
*against* a physiologically coherent signal — not proof of manipulation by itself, since poor
lighting, motion, or camera quality can equally cause it (which is exactly why motion and
illumination are scored and surfaced as separate confounds, not folded silently into the
verdict).

## Classifier

`app/models/baseline.py` ships two things:

1. **The real path**: load a `scikit-learn` classifier (Logistic Regression / Random Forest /
   SVM, selected by cross-validated AUC — see `scripts/train_baseline.py`) trained on the
   11-dimensional feature vector in `app/features/feature_vector.py`.
2. **The fallback**: if no trained model file exists at `MODEL_PATH`, a transparent weighted
   heuristic over signal quality, cross-region correlation/frequency agreement, and BPM
   stability — explicitly labelled `isHeuristicFallback: true` in every API response and in the
   UI. This exists only so the full application is exercisable before a labelled dataset is
   available; it is not a validated classifier and should never be presented or compared as one.

## Training protocol (once a dataset exists)

`scripts/train_baseline.py` uses **GroupKFold on subject ID**, not plain K-fold — this is
required for a valid subject-independent evaluation (a model that has seen a person's face in
training and sees a different clip of the same person at test time will look artificially
good). See spec Objective 12 / Definition of Done.

## Additional features: harmonics, phase coherence, amplitude consistency

Three features (`harmonic_ratio`, `phase_coherence`, `amplitude_cv`) were adapted from an
external notebook shared during development ("rPPG-Based Deepfake Detector v2" by Gauraav Kate
& Atharav Shelar, MIT WPU) and integrated into `app/features/harmonics.py`,
`app/features/phase_coherence.py`, and `app/features/amplitude_consistency.py`:

- **Harmonic ratio** — real pulse waveforms aren't a pure sinusoid; energy should also appear
  at 2x and 3x the dominant cardiac frequency. **Caveat found during integration**: on pure
  noise (no true dominant peak), this ratio can score *higher* than on a real structured
  signal — see `docs/limitations.md`. Mitigated by signal-quality gating, not fixed at the
  source.
- **Phase coherence** — cross-region phase stability via the Hilbert transform. Verified by
  testing to correctly separate synchronised synthetic signals (coherence > 0.8) from pure
  noise (coherence < 0.3) — see `tests/test_notebook_derived_features.py`.
- **Amplitude CV** — coefficient of variation of peak spectral amplitude across regions; real
  blood flow scales roughly proportionally across the face.

The same source notebook also motivated `app/face/haar_fallback.py` (a Haar cascade face
detector as a fallback when MediaPipe's Face Landmarker is unavailable) and
`app/video/enhancement.py` (CLAHE contrast enhancement for dark frames). The Haar cascade is
notable because it ships bundled inside `opencv-python` — unlike MediaPipe's Face Landmarker,
it requires no network download, making it the one face-detection path that was fully testable
in a network-restricted environment during this project's development.

## Explainability

`app/explainability/evidence.py` converts computed features into plain-language evidence and
warning strings using fixed, documented thresholds (not learned) — e.g. signal quality below
40 always produces a warning that the result is unreliable, regardless of what the classifier
outputs. `app/explainability/confidence.py` discounts the model's raw probability toward 0.5 in
proportion to how far signal quality falls below a "trustworthy" threshold, so a confident
model output computed from a noisy signal is never presented as confident to the user.
