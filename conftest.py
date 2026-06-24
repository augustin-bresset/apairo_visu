"""Make the package importable when running the test suite from a source tree
without an editable install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
