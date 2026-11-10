
import sounddevice as sd
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 8
CHANNELS = 1

speaker_id = input("Enter speaker ID (e.g. spk001): ").strip()
sample_number = input("Enter sample number (e.g. 001): ").strip()

if not speaker_id or not sample_number:
    raise ValueError("Speaker ID and sample number are required.")

output_dir = Path("data/raw/genuine") / speaker_id
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{speaker_id}_{sample_number}.wav"

input("Press ENTER when you are ready to record...")

print("Recording started. Speak naturally!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16"
)

sd.wait()

sf.write(output_file, audio, SAMPLE_RATE, subtype="PCM_16")

print(f"Recording saved: {output_file}")
