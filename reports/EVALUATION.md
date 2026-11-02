# Technical evaluation plan and report

**Current status:** evaluation code and unit tests are implemented, but no real dataset manifest or trained checkpoint is present in this workspace. Consequently there are no ASVspoof/WaveFake scores, figures, or benchmark comparisons to report here. Random synthetic test fixtures validate code paths only and are not model performance results.

## Reproducible procedure

1. Obtain ASVspoof 2019 Logical Access data and protocols through the official challenge access process. Generate and quality-check the manifest using the instructions in the repository README. Preserve protocol `train`, `dev`, and `eval` rows as separate partitions.
2. Run training smoke checks, then full training. Use the selected `checkpoints/best.pt`; do not select a checkpoint after viewing evaluation metrics.
3. Evaluate once with a development manifest to select the spoof threshold by maximum spoof F1, then apply that frozen threshold to the untouched official evaluation partition:

```powershell
python -m training.evaluate `
  --checkpoint checkpoints/best.pt `
  --dev-manifest data/manifests/asvspoof2019_la.csv `
  --dev-split dev `
  --evaluation-manifest data/manifests/asvspoof2019_la.csv `
  --evaluation-split eval `
  --output-dir reports/asvspoof2019_la_eval `
  --device auto
```

For a training setup using separate train/dev manifest CSVs, pass the dev CSV to `--dev-manifest`; pass the combined official manifest containing eval rows to `--evaluation-manifest`. The evaluator rejects `train`/`dev` as the final evaluation split and reads labels from the chosen non-train/dev partition only after threshold selection. Use a project-generated selected checkpoint and do not reuse evaluation results for tuning.

## Scores and metrics

Class IDs are `bona_fide=0`, `spoof=1`. For each recording, the system extracts LFCCs for all configured windows, computes raw two-class logits, averages logits across windows, applies softmax, and uses class index 1 as the continuous spoof score. Predict spoof at `spoof_score >= threshold`. The default threshold is selected on development data by maximizing spoof-class F1; ties prefer the higher threshold. If `--threshold` is supplied explicitly, it must have been determined using development data and is recorded as caller-supplied. A development manifest is still required to independently calibrate the amplitude baseline threshold.

The confusion matrix is `[[TN, FP], [FN, TP]]`, with true labels as rows and predicted labels as columns in order bona fide, spoof. Precision, recall, and F1 are spoof-positive; FPR is `FP/(FP+TN)` and FNR is `FN/(FN+TP)`. ROC-AUC is trapezoidal area under the continuous-score ROC curve. EER is estimated by linearly interpolating adjacent ROC points where FPR crosses FNR (`1 - TPR`), and reporting their mean. AUC/EER are unavailable when either class is absent.

Metrics by attack type are reported only when the manifest provides non-empty attack labels. Attack-specific subsets may lack bona-fide examples; in that case AUC/EER are null and rates with zero denominator are reported as zero by convention. Interpret support alongside every attack result.

## Baseline and latency

The comparison baseline is mean absolute waveform amplitude, using the exact shared resampling/window path. Its threshold is independently chosen on development data using the same spoof-F1 rule; it is a simple diagnostic comparator, not a validated detector.

The reported model latency measures CUDA-synchronized model scoring time per window, including tensor transfer and forward passes. Offline wall time includes preprocessing and file iteration. Neither value measures live microphone buffering, WebSocket transport, queueing, or end-to-end alert latency; these require a separate live-stream experiment.

## External WaveFake evaluation

WaveFake is an external dataset, not an ASVspoof partition. The published generated-audio archive is spoof-only and does not redistribute its source/reference corpora. The manifest generator therefore cannot form a labeled two-class test set from WaveFake alone. Binary metrics require compatible bona-fide reference records with documented licensing, provenance, and an explicit paired external protocol. Add those bona-fide rows (with their true source) to a separate evaluation CSV alongside WaveFake spoof rows; the evaluator accepts non-WaveFake rows only as bona fide. When such a manifest exists, use:

```powershell
python -m training.evaluate `
  --checkpoint checkpoints/best.pt `
  --dev-manifest data/manifests/asvspoof2019_la.csv `
  --dev-split dev `
  --evaluation-manifest data/manifests/wavefake_paired_external.csv `
  --evaluation-split external_eval `
  --external-dataset `
  --output-dir reports/wavefake_external_eval `
  --device auto
```

WaveFake's generated clips (16-bit PCM WAV, including English LJSpeech- and Japanese JSUT-related material) differ from ASVspoof LA's VCTK-derived challenge material and generation conditions. Record the exact paired-reference construction, languages, generator identities, sampling rates, and licenses. No unseen-generator generalization is claimed unless held-out generator identities are explicitly separated before model development and evaluated without tuning.

## Generated artifacts

A completed run writes `metrics.json`, per-recording `predictions.csv`, a development threshold record when fitted, and SVG ROC/confusion figures under the chosen output directory. Preserve the checkpoint, manifests/protocol version, config, data-access terms, and report together for reproducibility. Do not commit raw audio or sensitive manifests.
