import soundfile as sf
import numpy as np
from pathlib import Path

CLIP_DIR = Path("PodcastDataset/clips")

def check_audio(file_path):
    try:
        audio, sample_rate = sf.read(file_path)

        duration = len(audio) / sample_rate
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))

        issues = []

        if sample_rate != 16000:
            issues.append("Incorrect sample rate")

        if duration < 9:
            issues.append("Clip shorter than 9 seconds")

        if rms < 0.001:
            issues.append("Silent or very quiet audio")

        if peak >= 0.999:
            issues.append("Possible clipping")

        status = "PASS" if not issues else "CHECK"

        print(f"\nFile: {file_path.name}")
        print(f"Duration: {duration:.2f}s")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Peak: {peak:.4f}")
        print(f"RMS: {rms:.4f}")
        print(f"Status: {status}")

        for issue in issues:
            print("Issue:", issue)

    except Exception as error:
        print(f"INVALID FILE: {file_path} - {error}")


def main():
    files = list(CLIP_DIR.rglob("*.wav"))

    if not files:
        print("No WAV clips found. Add authorised audio and run preprocessing first.")
        return

    for file in files:
        check_audio(file)


if __name__ == "__main__":
    main()