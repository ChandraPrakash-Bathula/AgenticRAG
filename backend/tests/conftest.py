import sys
from pathlib import Path

# Make `app.*` importable when running `pytest backend/tests` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
