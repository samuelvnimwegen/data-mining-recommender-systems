"""Shim package so `import main` works in tests.

This module dynamically loads the top-level `main.py` file and re-exports
its symbols so tests that do `from main import main` can import reliably.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType

_root_main_path = Path(__file__).resolve().parent.parent / "main.py"
if not _root_main_path.exists():
    raise ImportError(f"Expected top-level main.py at {_root_main_path!s}")

_spec = util.spec_from_file_location("_top_level_main", str(_root_main_path))
_module = ModuleType("_top_level_main")
if _spec and _spec.loader:
    _spec.loader.exec_module(_module)  # type: ignore[arg-type]
else:
    raise ImportError(f"Could not load main module from {_root_main_path!s}")

# Re-export main function and useful helpers for tests
try:
    main = getattr(_module, "main")
except AttributeError as exc:  # pragma: no cover - defensive
    raise ImportError("Loaded main.py but it does not define `main`") from exc

# Export any helpers tests might want to use
build_argument_parser = getattr(_module, "build_argument_parser", None)
create_cleaner_config_from_arguments = getattr(_module, "create_cleaner_config_from_arguments", None)

__all__ = ["main", "build_argument_parser", "create_cleaner_config_from_arguments"]
