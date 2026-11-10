# Training dataset metadata and readiness

The readiness utility audits explicitly labeled local audio and can write a new source-/speaker-disjoint manifest. It never labels audio, downloads media, modifies source audio, trains the detector, or overwrites an existing manifest.

## Required input metadata

| Field | Meaning / rule |
|---|---|
| `file_path` | Existing local WAV/FLAC/OGG/AIFF audio; paths are relative to the metadata CSV or absolute. |
| `label` | Evidence-backed integer: `0` bona fide, `1` synthetic/spoof. Never inferred from a podcast, filename, or synthesis-looking content. |
| `split` | Must be empty, `unassigned`, or `pending` in the input. The output uses `train`, `dev`, `eval`; these mean train, validation, and test respectively and match the current trainer/evaluator. |
| `source_id` | Stable ID for the originating recording/source. All clips from one recording remain together. |
| `source` | Dataset/source name with provenance identifiable in project records. |
| `speaker_id` | Stable pseudonymous speaker identity, not a local diarization label that restarts for each recording. Same speaker IDs are kept in one split. |
| `utterance_id` | Stable utterance/clip identifier where available. |
| `language` | Language label; use `unknown` only when genuinely unknown (reported as a warning). |
| `permission_status` | Must be `granted` before inclusion. Keep actual permission evidence/records separately. |
| `permission_evidence_ref` | Non-placeholder reference to permission evidence; `granted` by itself is insufficient. |
| `generation_method` | For label 1, record the evidenced synthesis method/system. For label 0 use `none` or `not_applicable`. |
| `label_source` | Authority used for the class decision, such as a verified protocol or documented human review. |
| `label_provenance` | Enough detail to trace the label to its source/evidence; non-empty text is required but is not independently authenticated by this tool. |
| `attack_type` | Optional attack/generator category when evidenced; blank for bona fide or unknown. |

The output also adds `audio_sha256`, calculated from fully decoded audio blocks. Repeated paths, identical decoded audio, missing audio, corrupt audio, missing classes, class imbalance, and cross-split source/speaker overlap are reported. Class imbalance above 3:1 is a warning by default; it is not silently corrected by duplicating records.

## Split constraints

`source_id` and `speaker_id` form indivisible connected groups: records sharing either identifier are assigned to a single split. This is deliberately stricter than clip-level random splitting. At least three independent groups carrying each class are needed to place both classes in all three partitions. The split ratios are targets, not a guarantee of exact sample proportions when groups differ in size.

Diarization turn tags from one recording (for example `SPEAKER_00`) are not global speaker identity across recordings. Populate a globally stable speaker key only when supported by the dataset protocol or an independently documented identity mapping. Otherwise the source-level grouping still prevents clips from one recording leaking across splits, but a true cross-recording speaker-disjoint claim cannot be made.

## Current workspace audit

The current `PodcastDataset/clip_metadata.csv` has only its legacy clip columns and no rows; it has no binary label, generation method, or label provenance. The 16 podcast records remain in their link/processing manifests, marked `granted` and `waiting_for_audio`, but there are no local podcast media files or processed clips in the inspected workspace. `data/manifests/` has no training manifest. Two local files exist under `data/raw/genuine/spk001/`: `spk001_1.wav` and `spk001_2.wav`; their headers report mono 16 kHz, PCM-16 WAV, 8 seconds each. No accompanying metadata or provenance record was found to verify recording identity, consent, source, speaker identity, language, or bona-fide label authority. The `genuine/spk001` folder names alone are not label or provenance evidence, so neither file is eligible for training until documented in the required schema. No synthetic examples or synthetic-label evidence are present. A `granted` podcast CSV value is metadata and does not itself prove or document licensing/consent.

The existing ASVspoof builder derives class labels from its official protocol and records dataset speaker IDs, but its output lacks the extended `source_id`, permission, generation-method, and label-provenance fields required here. Those fields must be populated from authoritative dataset documentation/permissions before this readiness tool accepts that data. WaveFake's generated-only records are not sufficient for a two-class split by themselves.

## Evidence curation workflow

`training.dataset_curation` reads `podcast_links.csv`, `processing_manifest.csv`, and the existing legacy `clip_metadata.csv` without changing them. Podcast URLs remain source metadata only. Records without actual local audio or curation evidence are reported as pending. It also inspects local WAVs technically (sample rate, channels, duration, subtype, and decoded-content SHA-256), without inferring their source, speaker, permission, or class from folder names.

When evidence has been independently reviewed, create a separate curation CSV with these exact columns:

```text
audio_id,source_id,speaker_id,audio_path,binary_label,language,permission_status,permission_evidence_ref,label_source,label_provenance,synthetic_generation_method,split_assignment
```

`binary_label` must be explicit `0` or `1`; `permission_status` must be `granted` with a non-empty local evidence document; `label_source` and `label_provenance` must point to non-empty local documents describing the label authority/evidence. External URLs are never fetched and remain pending until their evidence is made locally available and reviewed. Label 1 additionally requires the evidenced synthesis method. `split_assignment` must be blank, `unassigned`, or `pending`; the splitter assigns partitions after checking leakage and independent-group counts. The program checks that evidence files exist and are non-empty, not that their contents prove the claim; human review is still mandatory.

First audit without generating any manifest:

```powershell
.\.venv\Scripts\python.exe -m training.dataset_curation
```

To audit a separate curated table and save a new report:

```powershell
.\.venv\Scripts\python.exe -m training.dataset_curation `
  --curation-csv path\to\curation_metadata.csv `
  --report-output reports\dataset_curation.json
```

Only when the report says data are ready, a new training manifest may be written with `--manifest-output <new-path>`. Existing report or manifest paths are never overwritten. The pipeline requires both classes and at least three independent connected source/speaker groups per class to populate train/dev/eval without leakage. Accepted individual records do not imply the complete dataset is ready.

## Usage

Run an audit without requesting a split manifest while curating metadata:

```powershell
.\.venv\Scripts\python.exe -m training.dataset_readiness `
  --input path\to\labeled_metadata.csv `
  --output data\manifests\training_ready.csv `
  --audit-only
```

The audit-only mode does not create the `--output` file. It reports the current labeled class coverage, permissions, duplicates, missing/corrupt audio, unknown language, and imbalance.

After preparing an evidence-backed labeled metadata CSV with the schema above, create new output paths and run:

```powershell
.\.venv\Scripts\python.exe -m training.dataset_readiness `
  --input path\to\labeled_metadata.csv `
  --output data\manifests\training_ready.csv `
  --report reports\training_data_readiness.json `
  --seed 42
```

The tool refuses to split incomplete, ambiguous, or pre-split data and refuses to overwrite either output. This is a readiness check; it does not train a model or claim detector performance.

The generated manifest keeps the training module's required `file_path`, `label`, `split`, and `source` columns and includes the additional provenance/disjointness metadata. Map `source_id` to the trainer's `source` column for compatibility; the readiness utility writes both fields.
