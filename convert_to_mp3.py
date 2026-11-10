"""Compatibility entry point; use process_podcast_dataset.py --media instead."""

from process_podcast_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
