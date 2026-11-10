"""Compatibility entry point; audio processing is centralized in the safe pipeline."""

from process_podcast_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
