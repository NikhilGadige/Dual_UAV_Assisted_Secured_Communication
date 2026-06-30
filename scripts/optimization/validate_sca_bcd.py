"""Validation entry point for SCA-BCD optimization."""
import sys
import os
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from sca_bcd_exp.validate_sca_bcd import main

if __name__ == "__main__":
    main()
