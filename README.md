# A.U.D.I.O — AI-Driven Uncertainty and Detection for Impersonation Overwatch

## Federated Voice Threat Intelligence Platform

## A.U.D.I.O local inference API

Start the FastAPI server from the repository root after installing `requirements.txt` and ensuring FFmpeg/ffprobe are on `PATH`:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The service loads `checkpoints/user_audio_human_plus_ai/best.pt` once at startup. Override it with `SONARA_CHECKPOINT_PATH`; choose a device with `SONARA_DEVICE` (`auto`, `cpu`, or a CUDA device). `GET /api/health` reports backend and model availability separately. `POST /api/analyze` accepts multipart field `file`, supports M4A/WAV/FLAC/MP3/OGG/AIFF, and enforces a 50 MB limit. It validates and converts the original recording to mono 16 kHz PCM WAV, then invokes `LocalVoicePredictor` using the checkpoint's embedded preprocessing configuration. Temporary source and conversion files are removed after analysis.

For local Vite development, create `frontend/.env.local` with `VITE_API_BASE_URL=http://127.0.0.1:8000`, then run:

```powershell
cd frontend
npm run dev
```

The frontend sends the original file only after the user requests inference. Scores and class results are emitted only when the actual predictor runs. Results are experimental; the development split informed checkpoint selection and there is no independent untouched test split. This API does not persist history or provide a chat endpoint.

This repository contains a locally trained, experimental voice anti-spoofing detector and a local inference API. It does not depend on a hosted inference service or pretrained voice-detection model. The model has no independent untouched test split, so its returned estimates are not independently accuracy-validated and are not proof of identity or fraud.

## Architecture

- **Audio and features:** the local console capture uses sounddevice and a bounded in-memory ring; shared preprocessing validates/resamples and computes LFCCs with NumPy and SciPy. Avoid retaining raw audio by default.
- **Model and training:** implement a new PyTorch classifier with random initialization; train from a documented real-versus-spoof dataset manifest and save model checkpoints and evaluation reports locally. No pretrained weights.
- **Streaming and API:** a FastAPI backend accepts authenticated, bounded WebSocket audio frames, runs local inference, and returns per-window scores. Add transport security, input limits, session cleanup, and rate limits before real deployment.
- **Risk and prevention:** combine calibrated model outputs over multiple windows with configurable policy thresholds; surface a risk score and alert. Keep any high-impact action human-confirmed; a score is not proof of fraud.
- **User interface:** React and TypeScript, built with Vite, captures microphone audio only after user consent and displays session status, alerts, and recommended verification steps.
- **Validation:** unit and integration tests, subject/disjoint train-validation-test splits, threshold calibration, and reporting of false-accept and false-reject rates. Evaluate robustness and demographic performance before operational use.

## Initial directory layout

```text
.
├── backend/
│   └── app/
│       └── api/
├── checkpoints/          # Generated model artifacts; not committed
├── configs/
│   └── baseline.yaml
├── data/
│   ├── manifests/        # Dataset split/index files; inspect for sensitive metadata
│   ├── processed/        # Derived features; not committed
│   └── raw/              # User-provided licensed audio; not committed
├── frontend/             # React + TypeScript + Vite application
├── inference/            # Local model loading and scoring
│   └── predict.py
├── reports/              # Evaluation outputs; not committed
├── streaming/             # Audio framing and live microphone capture
│   ├── audio_buffer.py
│   └── microphone.py
├── tests/
│   ├── integration/
│   └── unit/
├── training/              # Dataset, model, training, and evaluation code
│   ├── audio_preprocessing.py # Shared waveform normalization and LFCC
│   ├── dataset_manifest.py
│   └── examples/
│       └── inspect_lfcc.py
├── .gitignore
└── requirements.txt
```

Directories are scaffolding only; application modules and the frontend package have not yet been generated.

## Environment observed

- Windows PowerShell; Python 3.12.10 (`py` and `python` are available).
- NVIDIA GeForce RTX 3050 Laptop GPU, 6 GiB reported memory; NVIDIA driver 576.93 reports CUDA 12.9 support. The installed PyTorch `2.11.0+cu128` build was verified with CUDA 12.8: CUDA availability, tensor allocation, and a GPU matrix multiplication all passed.
- Node.js 26.7.0, npm 11.19.0, and Git 2.56.0 are available.

The existing `.venv` uses Python 3.12.10 and has the pinned audio/backend dependencies installed. Its PyTorch `2.11.0+cu128` reports CUDA runtime 12.8, and `check_gpu.py` passed on the RTX 3050 Laptop GPU. No separate CUDA Toolkit installation is normally required for PyTorch wheels.

## Windows PowerShell setup

From the repository root in PowerShell (the existing `.venv` was found and is Python 3.12.10; the guard preserves it):

```powershell
if (-not (Test-Path .\.venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_gpu.py
```

`requirements.txt` pins the tested dependency set and selects the PyTorch CUDA 12.8 wheel index. The wheel bundles the CUDA runtime it needs. Do not install `torchaudio` separately; it is not required by this scaffold. If it becomes necessary later, install the exact matching torchaudio release for the selected torch release from the same official wheel index. For another platform, GPU, or driver, use the official PyTorch installation selector and revise the pin deliberately.

`check_gpu.py` reports Python/PyTorch versions, CUDA runtime and availability, device, tensor allocation, and a small CUDA matrix multiplication. It exits nonzero if CUDA is missing or the GPU computation fails. Do not consider GPU setup complete unless it prints `CUDA verification: PASS`.

### Common Windows setup fixes

- **PowerShell says script execution is disabled:** for the current user, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, confirm the prompt, then activate again. Alternatively, skip activation and call `\.venv\Scripts\python.exe -m pip ...` directly.
- **`pip` is not found or installs into another Python:** use `python -m pip ...` after activation, then confirm `python -c "import sys; print(sys.executable)"` points into this repository's `.venv`.
- **Access denied / wrong interpreter:** close terminals, activate the project `.venv`, and avoid administrator installs. Recreate the environment only if needed, after preserving any user data.
- **`torch.cuda.is_available()` is false or the wheel says CPU-only:** run `python -m pip show torch` and check `python -c "import torch; print(torch.__version__, torch.version.cuda)"`. Reinstall the pinned wheel with `python -m pip install --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0`. Then rerun `python check_gpu.py`.
- **Driver/runtime error or `CUDA driver version is insufficient`:** update the NVIDIA display driver from NVIDIA, restart Windows, and retry. The CUDA runtime bundled with PyTorch does not require the standalone CUDA Toolkit for ordinary PyTorch use.
- **Out of memory:** close GPU-heavy applications, reduce batch size/window size, and retry. The detected GPU reports 6 GiB VRAM.
- **Torch/torchaudio version conflict:** torchaudio is not needed here. Do not install its latest version blindly; match its release to the installed torch release and use the same CUDA wheel index.
- **Network or wheel resolution errors:** run `python -m pip install --upgrade pip`, confirm network access to PyPI and `download.pytorch.org`, and retry. Avoid mixing CPU and CUDA package indexes or using `--no-deps` as a first fix.

After a successful installation, capture the exact resolved Windows environment with `python -m pip freeze > requirements-lock.txt`. Recreate it with `python -m pip install -r requirements-lock.txt`; install using the CUDA index in `requirements.txt` first if the lock does not include index directives.

If CUDA is unavailable, use CPU for initial development; do not assume the GPU will be used. A compatible GPU build and sufficient free VRAM must be confirmed before training.

The frontend can later be initialized from `frontend/` with Vite's React + TypeScript template, followed by `npm install` and `npm run dev`. Backend, training, and test commands will be documented once their entry points are implemented.

## Data, evaluation, and privacy

- Obtain and document dataset licenses, provenance, labels, and consent before placing audio under `data/raw/`; do not commit audio or identifiable manifests.
- Keep raw audio in memory for inference and discard it after each session unless explicit consent and a documented retention need justify otherwise. Do not log audio, transcripts, or unnecessary identifiers.
- Split by speaker/source and, where possible, generation system to reduce leakage. Report the protocol, class balance, calibration, false-accept/false-reject rates, and limitations. Do not choose a production threshold from training data.
- Treat model scores as uncertain signals. Provide a human review and independent verification workflow; do not automatically deny access or transfer funds based only on the detector.

## Audio preprocessing and LFCC

The exact shared preprocessing parameters are in [configs/baseline.yaml](configs/baseline.yaml). Both training and inference must load this same configuration and call `training.audio_preprocessing.process_audio_file` (or `process_waveform` for already-loaded samples); do not duplicate preprocessing in separate paths.

- Decode WAV, FLAC, OGG, or AIFF via SoundFile; reject unsupported suffixes, missing paths, undecodable files, and empty waveforms with `AudioProcessingError`.
- Downmix sample-by-channel audio by averaging channels. Replace NaN and infinities with zero, clip finite amplitudes to `[-1, 1]`, and **do not peak-normalize** individual files.
- Resample deterministically to mono 16 kHz with SciPy `resample_poly`, fixed rational factors, Hann resampling window, and constant boundary extension.
- Window at 2.0 seconds (32,000 samples), hop 0.5 seconds (8,000 samples). Short clips are right-zero-padded. Long clips use left-aligned overlapping windows; a final partial tail is zero-padded, without dropping the tail.
- LFCC: 25 ms frame (`win_length=400`), 10 ms hop (`hop_length=160`), 512-point RFFT, periodic Hann, no centered padding, pre-emphasis 0.97, power spectrum, 64 triangular filters linearly spaced from 0 to 8 kHz, log energy floor `1e-10`, orthonormal type-II DCT, first 40 coefficients including C0. For the configured 2-second window this yields **40 x 198** values per window. The output is finite `float32` shaped `[windows, coefficients, frames]`.
- Normalization is `none`: there are no corpus-level or evaluation-fitted statistics. If later experiments add learned normalization, estimate it on training data only, save it with the checkpoint/config, and apply the same frozen values to development/evaluation/inference.

Run tests (temporary synthetic files only):

```powershell
python -m pytest tests/unit/test_audio_preprocessing.py tests/unit/test_dataset_manifest.py -v
python -m ruff check training tests/unit
```

To inspect a **real locally downloaded** benchmark file (replace the example name with an existing file):

```powershell
python -m training.examples.inspect_lfcc data/raw/ASVspoof2019_LA_train/flac/LA_T_XXXXXXXX.flac
```

The example prints the fixed-window waveform tensor and LFCC tensor shapes. It does not download data or train a model.

## Dataset acquisition and manifest preparation

No benchmark data is included in this repository, downloaded by the tooling, or fabricated by the tests. Keep downloaded archives and extracted audio outside Git (the `data/raw/` tree and common audio extensions are ignored). Review and retain each dataset's current terms and attribution with your research records before use.

### ASVspoof 2019 Logical Access (primary benchmark)

- Start at the [official ASVspoof 2019 challenge page](https://www.asvspoof.org/index2019.html) and follow its license/access instructions; the challenge page indicates that dataset access is provided through its licensing process. The [University of Edinburgh DataShare record](https://datashare.ed.ac.uk/handle/10283/3336) is the catalog record for the database and lists LA/PA archives, protocol/readme material, and the evaluation plan. Follow the current access terms and download the **LA** archive and its protocol files from the official sources; do not use unofficial mirrors as a substitute.
- The DataShare metadata describes FLAC-compressed PCM. It documents train, development, and evaluation speaker partitions as disjoint; preserve those official splits. Do not merge or reshuffle them. Protocol labels are authoritative: bona fide maps to `0`, spoof to `1`. The parser rejects unknown/missing keys rather than guessing labels.
- **Storage:** consult the current official archive listing for exact LA and protocol archive sizes; the captured listing did not provide reliable byte sizes. Reserve space for both the downloaded archive and its extracted files. On this machine, the earlier disk check observed about 289 GB free on C:, but recheck before downloading.
- Extract the LA audio and protocols while preserving names. Expected layout (directory names can vary slightly by archive version):

```text
data/raw/ASVspoof2019_LA_train/flac/<utterance-id>.flac
data/raw/ASVspoof2019_LA_dev/flac/<utterance-id>.flac
data/raw/ASVspoof2019_LA_eval/flac/<utterance-id>.flac
data/raw/ASVspoof2019_LA_cm_protocols/
	ASVspoof2019.LA.cm.train.trn.txt
	ASVspoof2019.LA.cm.dev.trl.txt
	ASVspoof2019.LA.cm.eval.trl.txt
```

Run from the repository root after extraction:

```powershell
python -m training.dataset_manifest --dataset asvspoof2019-la `
	--protocol-dir data/raw/ASVspoof2019_LA_cm_protocols `
	--audio-root data/raw `
	--output data/manifests/asvspoof2019_la.csv `
	--report reports/asvspoof2019_la_data_quality.json `
	--strict
```

The generator reads each official train/dev/eval protocol, matches utterance IDs to extracted audio, retains protocol speaker/attack metadata when present, checks audio decode and the expected 16 kHz sample rate, and reports missing/ambiguous paths, duplicates, class counts, and cross-split speaker/utterance overlap. It writes the manifest and report even when quality issues are found; `--strict` then returns a nonzero exit code. Fix layout or metadata issues and rerun before using the manifest. Audio is not resampled or modified.

### WaveFake (secondary external generalization benchmark)

- Use the [WaveFake Zenodo record](https://zenodo.org/records/5642694) (version 1.2.0) and the authors' [code/repository documentation](https://github.com/RUB-SysSec/WaveFake). The record lists `generated_audio.zip` at 28.9 GB and the dataset license as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Read the record's datasheet/license and attribution requirements. Its description says it contains 104,885 generated 16-bit PCM WAV clips and does **not** redistribute the reference corpora.
- Download the archive through the Zenodo record's file interface and extract it outside Git. Reserve at least the 28.9 GB archive plus additional extraction space; the exact expanded size depends on the archive and must be checked before extraction. The archive is organized by generated model/source in release-specific folders; the script indexes recursively and records the nearest non-generic directory as attack-type metadata when available.
- WaveFake's generated archive does not provide ASVspoof-style official train/dev/eval partitions or a complete bona-fide counterpart. The manifest tool therefore labels these records as spoof `1` in `external_eval` only. Do not use them as bona-fide examples or silently create train/dev splits. For a defensible generalization metric, define an explicit held-out protocol and obtain compatible bona-fide reference audio under its own license; report domain and speaker limitations.

```powershell
python -m training.dataset_manifest --dataset wavefake `
	--audio-root data/raw/WaveFake/generated_audio `
	--output data/manifests/wavefake_external_eval.csv `
	--report reports/wavefake_data_quality.json `
	--expected-sample-rate 16000 `
	--strict
```

The `--expected-sample-rate` option only validates; it never resamples. WaveFake may contain more than one source/model sampling rate, so inspect the report's observed sample-rate counts and choose a documented preprocessing/resampling policy later, rather than treating validation warnings as successful normalization.

### Synthetic test fixtures (not benchmark records)

Run `python -m pytest tests/unit/test_dataset_manifest.py -v` to exercise protocol mapping, split preservation, path/audio checks, leakage reporting, and WaveFake's external-evaluation behavior. The tests create temporary audio and protocol rows at runtime under pytest's temporary directory; these are synthetic fixtures, not real ASVspoof/WaveFake records, benchmark results, or model evaluation.

## Evaluation

The evaluation implementation, class-score convention, EER definition, threshold rules, baseline, latency scope, WaveFake caveats, and reproducible commands are documented in [reports/EVALUATION.md](reports/EVALUATION.md). Run it only after a full training run has produced `checkpoints/best.pt` and the official held-out evaluation manifest has passed quality checks. No real evaluation scores are present until that experiment is run.

## Live local microphone inference

`streaming/microphone.py` captures microphone input with `sounddevice`, requests mono 16 kHz float audio, and places callback chunks into a bounded two-second ring buffer. The audio callback only copies/buffers samples and signals a worker; LFCC extraction and model inference happen on the worker thread. Completed windows are processed every 500 ms. The local predictor loads only a project-owned training checkpoint and reconstructs the exact audio/LFCC settings stored in it. Inference uses CUDA when available and falls back to CPU. Audio remains in memory and is not recorded to disk.

Install the newly pinned microphone dependency in the activated project environment:

```powershell
python -m pip install -r requirements.txt
```

List audio input devices:

```powershell
python -m streaming.microphone --checkpoint checkpoints/best.pt --list-devices
```

Start the console stream with the Windows default microphone. If the default input is unavailable, use a listed input index:

```powershell
python -m streaming.microphone --checkpoint checkpoints/best.pt --device auto
```

Select a specific input device by index if needed:

```powershell
python -m streaming.microphone --checkpoint checkpoints/best.pt --microphone-device 1 --device auto
```

This requires a trained, selected `checkpoints/best.pt`; no checkpoint is currently present in this workspace. The console labels results as **model estimates, not verified identity**. Scores are not identity verification or proof of fraud. This console implementation does not initiate transactions, block calls, or take other automatic high-impact actions. Input overflow, stale inference windows, silence, and clipping are reflected in the audio-quality status; missing devices, Windows microphone permissions, unsupported audio rates, and invalid checkpoints stop with an actionable error. Buffering and drop counters are bounded/volatile and no microphone samples are persisted.

## Continuous spoof-risk policy

`inference/risk_engine.py` consumes per-window scores and audio-quality metadata, but its output is a workflow triage signal only. It does not estimate real-world fraud probability, verify identity, and must not be used to automatically approve/reject a transaction, block a call, or deny access. Never treat caller ID, a matching name, or contextual claims as evidence that speech is genuine.

All policy knobs are in the `risk` section of [configs/baseline.yaml](configs/baseline.yaml). The defaults are **uncalibrated placeholders**, not production thresholds. Before operational use, generate predictions for a labeled development split with the same selected checkpoint, then calibrate thresholds against bona-fide development scores:

```powershell
python -c "from inference.risk_engine import calibrate_from_development_csv, save_calibration; r=calibrate_from_development_csv('reports/asvspoof_dev/predictions.csv', target_false_positive_rate=0.01, development_source='ASVspoof2019_LA:dev'); save_calibration('configs/risk_calibration.json', r); print(r)"
```

The CSV must contain `label`, `spoof_score`, and an explicit `split` column containing only `dev`/`development` rows. The calibration selects the lowest score threshold meeting the requested development bona-fide false-positive-rate target; the high-risk threshold is a configured margin above it. This does not convert model scores into fraud probabilities or guarantee deployment error rates. Load settings with `policy_from_config('configs/baseline.yaml', 'configs/risk_calibration.json')` and pass that policy into `VoiceRiskEngine`.

The engine keeps a bounded history, groups windows whose timestamps are within the configured correlation interval (or share an explicit `correlation_group`), and median-aggregates each group. Persistence and evidence counts advance per group, not per overlapping window. Repeated alert notifications are cooldown-limited; an active alert recovers only after persistent low-risk groups. Missing predictions or non-`ok` audio-quality status return `insufficient_evidence`, never a “safe” result. Events include timestamp, smoothed model score, alert level/emission, reason codes, and recommended independent verification actions. Human review and a separate trusted-channel verification remain required for sensitive requests.

## Supervised training

`training/dataset.py` reads only rows marked `train` or `dev` and delegates waveform processing/LFCC extraction to `training.audio_preprocessing`. Training samples one deterministic window per recording per epoch; development evaluates all windows and averages logits per recording. Class IDs are fixed (`bona_fide=0`, `spoof=1`). The default imbalance strategy is inverse-frequency weighted cross-entropy; alternatives are `none` and `weighted_sampler`.

The trainer uses a randomly initialized `LFCCCNNDetector`, cross-entropy, AdamW, an optional CUDA-only mixed-precision path, development macro-F1 checkpoint selection (lower dev loss breaks ties), a ReduceLROnPlateau scheduler, and early stopping. The evaluation split is not loaded. Checkpoints preserve model/optimizer/scheduler/scaler states, model architecture, full config/preprocessing settings, class mapping, seed, manifest paths, environment metadata, and history. Resume only from a project checkpoint with matching preprocessing, model/class configuration, seed, and train/dev manifest paths. Optimizer settings and total epoch count may be adjusted.

1. Generate a validated ASVspoof manifest as described above. Confirm it contains both `train` and `dev` labels and that the generated report has no unresolved audio issues. There are currently no real dataset manifests in this workspace.
2. Run a **smoke test first** on a small per-class subset of those real train/dev records (this checks pipeline execution, not model quality):

```powershell
python -m training.train --config configs/baseline.yaml --smoke --smoke-limit-per-class 4
```

Smoke checkpoints go only under `checkpoints/smoke/`. The trainer prints losses/metrics for the selected fixture, but these values are not benchmark or performance evidence. The automated smoke test uses generated synthetic audio in pytest temporary directories and is explicitly not real dataset evidence.

3. Only after inspecting the smoke logs and data-quality report, start full training:

```powershell
python -m training.train --config configs/baseline.yaml
```

Resume a full run from its own last checkpoint with:

```powershell
python -m training.train --config configs/baseline.yaml --resume checkpoints/last.pt
```

Edit `training.train_manifest`, `training.dev_manifest`, batch size, epoch count, device, and imbalance policy in the baseline YAML as needed. The default manifest path is a combined CSV from the manifest generator. The full training command cannot run successfully until real, licensed audio and its manifests are present.

## Assumptions and open decisions

- Python 3.12 is the current target; the pinned dependencies and CUDA wheel have been verified in this workspace.
- The RTX 3050 Laptop GPU has 6 GiB reported VRAM; model size, batch size, and audio window length must be tuned to available memory.
- ASVspoof 2019 LA is the primary benchmark and WaveFake the secondary external benchmark; dataset access and terms must be completed by the user. Latency, deployment topology, authentication, and retention policy still need decisions. Baseline audio settings in `configs/baseline.yaml` are starting points, not validated requirements.
- The frontend is intended to run locally during development. Production microphone access requires HTTPS (except localhost), explicit browser permission, and a security-reviewed deployment.
- This scaffold creates no model, makes no detection claims, and is not ready to protect a real workflow until training, independent evaluation, security review, and operational testing are complete.

## Safe local podcast processing

Use the unified pipeline from the repository root after placing authorized media files locally. Supply an explicit `SOURCE_ID=PATH` mapping for each recording; YouTube URLs are never downloaded or treated as files. The pipeline preserves source recordings, diarizes each complete recording before making 10-second mono 16 kHz PCM WAV clips, aligns speaker turns into clip metadata, and updates the processing manifest only after processing succeeds. Existing outputs are never overwritten; a rerun skips sources already marked processed.

```powershell
.\.venv\Scripts\python.exe process_podcast_dataset.py `
	--media "P001=PodcastDataset/originals/P001/authorized-recording.mp4" `
	--media "P002=PodcastDataset/originals/P002/authorized-recording.wav"
```

FFmpeg is discovered in this order: `--ffmpeg <path>`, the `FFMPEG_PATH` environment variable, then system `PATH`; its `-version` command is checked before conversion. If FFmpeg is installed but not on `PATH`, set `$env:FFMPEG_PATH = "<full path to ffmpeg.exe>"` in PowerShell, or pass `--ffmpeg "<full path to ffmpeg.exe>"`. If Pyannote needs access to a model not already cached or otherwise accessible, set `HF_TOKEN` in the PowerShell environment. The token is read from the environment only and is never written to project files or printed. Pyannote is used only for speaker diarization. Running without local media prints a per-source `WAITING` report and does not change the manifest or metadata. Outputs are written to the manifest's MP3 and clip paths, with speaker turns under `PodcastDataset/diarization/` and clip QA/alignment in `PodcastDataset/clip_metadata.csv`.
