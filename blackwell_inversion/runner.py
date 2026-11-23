"""Compatibility entry point: `python -m blackwell_inversion.runner` equals `python -m blackwell_inversion`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
