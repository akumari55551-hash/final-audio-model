"""Compatibility entry point; metadata is written only after pipeline success."""

from process_podcast_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
