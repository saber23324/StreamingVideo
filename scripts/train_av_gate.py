"""Compatibility entry for the independent gate trainer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.experiment import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    main()
