# Limitations

Read this before trusting, presenting, or building further on any output of this system.

## No validated real-video accuracy is available

The repository includes `models/production/baseline.joblib`, a Random Forest trained on
synthetically generated rPPG features. Its 94% accuracy and 0.982 AUC are from a random holdout
of that same synthetic signal generator. They do not measure accuracy on genuine or deepfake
videos. No labeled real/fake video dataset is included, and no cross-dataset test has been run.
The service marks this optional model `synthetic_unvalidated`; a trained estimator alone is not
evidence of validated accuracy. The live rPPG decision does not use this artifact.
**Do not present any REAL/FAKE output as a validated forensic result.**

Use `scripts/train_baseline.py` with verified video-derived features and distinct subject IDs.
It uses stratified group folds and stops promotion below 80% mean balanced accuracy. Then run
`scripts/evaluate.py` on a held-out feature table with no subject overlap. Accuracy on that test
set is meaningful only if it represents the deployment population and deepfake methods.

## rPPG is not a deepfake detector on its own

A physiologically plausible, consistent pulse signal is evidence the video is *not obviously*
a class of deepfake that fails to preserve blood-flow cues — it is not proof of authenticity.
Some synthesis/reenactment methods do preserve or approximate physiological signal; a video
could also be REAL but score poorly here due to bad lighting, compression, or motion. The live
output is an rPPG-based directional assessment, not proof of authenticity or manipulation.

## Signal-quality gating is a heuristic

The SNR→quality-score mapping in `app/rppg/quality.py` and the confidence-discounting curve in
`app/explainability/confidence.py` and rPPG cue weights are heuristics, not fitted to labelled
ground truth (e.g. Platt scaling against known real/fake outcomes). They should be tuned and
calibrated against a verified video benchmark before being interpreted as probabilities.

## Face detection dependency

`app/face/detector.py` uses MediaPipe's Face Landmarker (Tasks API), which requires a
downloaded `.task` model file not bundled with the `mediapipe` package as of the version
pinned here. Videos with no visible/detectable face, extreme angles, or heavy occlusion will
produce a "no usable signal" failure rather than a wrong prediction — this is intentional
fail-closed behaviour (see `app/pipeline/end_to_end.py`), but it means coverage on
low-quality/uncooperative footage is inherently limited.

## Motion, lighting, and compression sensitivity

rPPG methods (Green/CHROM/POS) are all fundamentally sensitive to head motion, non-uniform or
changing illumination, and heavy video compression — these degrade signal quality regardless of
whether the video is real or fake, which is why `warnings` in every result explicitly separates
"reasons this measurement might be unreliable" from "evidence for/against authenticity."
Robustness experiments across compression levels/motion/lighting (spec Milestone in
"Robustness & Generalisation") have not been run in this build — the pipeline supports running
them (feed the same video processed at different bitrates/resolutions through
`scripts/extract_features.py`), but no results are included.

## Harmonic ratio feature is unreliable without a real dominant peak

`app/features/harmonics.py` (ported from an external notebook, see
`docs/methodology.md`) can score *higher* on pure noise than on a real
structured pulse signal, because "energy near 2x/3x the peak" is only
meaningful when the peak itself is a real cardiac frequency rather than
a noise spike. This was caught by testing (see
`tests/test_notebook_derived_features.py`) rather than assumed away. It
is currently neutralised by signal-quality gating (a related bug was
caught the same way: an early version of the CLAHE dark-frame
enhancement applied contrast correction to the same frame later used
for RGB signal measurement, corrupting the small pulsatile amplitude
it was trying to detect — fixed in `app/pipeline/extraction.py` so
enhancement is now used only to help detection, never to derive the
measured signal) — the classifier and
evidence layer only weight harmonic-ratio evidence when
`signal_quality_mean` clears a minimum threshold, and pure noise
reliably scores near-zero quality — but the underlying feature
computation itself has not been made robust to this case. Treat
`featureVector.harmonicRatio` as informative only when reported
alongside a reasonable `signalQuality` score, not on its own.

## No cross-dataset generalisation testing performed

Any trained model should be evaluated on datasets/generation methods it wasn't trained on
before being trusted on novel deepfake techniques — see spec Objective 12. This has not been
done here since no dataset is included.

## Environment this codebase was generated in

The scientific core (`services/ml/app/{video,face,rppg,features,models,explainability,
pipeline}/`) was written and *tested end-to-end* in the environment that produced this
repository — see `services/ml/tests/`, which run and pass with only NumPy/SciPy/OpenCV/
MediaPipe/scikit-learn (already present). The Node/Express API and React frontend were written
as complete, internally consistent source code but could not be `npm install`-ed, type-checked
against their real dependency trees, or run in that environment (no network access). Run
`pnpm install` and `pnpm dev:api` / `pnpm dev:web` locally to exercise them for the first time —
treat the first local run as the point at which integration bugs (import typos, minor API
mismatches) are most likely to surface, and expect to fix a handful.
