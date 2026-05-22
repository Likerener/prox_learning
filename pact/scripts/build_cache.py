"""CLI wrapper; delegates to prox_encoder.cache so the package stays importable
without scripts on sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prox_encoder.cache import main

if __name__ == "__main__":
    raise SystemExit(main())
