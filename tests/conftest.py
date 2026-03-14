"""Test configuration for pytest.

Ensure the project root is on sys.path during test collection so imports like
`from main import main` resolve reliably when pytest changes the import context.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the project root at the front of sys.path to ensure local imports work.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
