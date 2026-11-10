import csv
from pathlib import Path

root = Path("PodcastDataset")
originals = root / "originals"
clips = root / "clips"
metadata = root / "clip_metadata.csv"

print("DATASET VALIDATION")
print("-" * 35)

print("Originals folder:", originals.exists())
print("Clips folder:", clips.exists())
print("Metadata CSV:", metadata.exists())

wav_files = list(clips.rglob("*.wav"))
print("Total WAV clips:", len(wav_files))

if metadata.exists():
    with open(metadata, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ids = [row["clip_id"] for row in rows]
    duplicates = len(ids) - len(set(ids))

    print("Metadata records:", len(rows))
    print("Duplicate clip IDs:", duplicates)

print("-" * 35)
print("Validation completed.")