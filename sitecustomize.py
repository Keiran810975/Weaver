"""Optional Weaver auto-bootstrap.

This module is imported by CPython when the repository root is on PYTHONPATH.
It stays inert unless WEAVER_AUTO_PROFILE=1 is present, which lets the launcher
turn on the native CPython profile collector without touching user training code.
"""

import os

try:
    from weaver.collector.py_runtime import enable_from_env

    enable_from_env()
except Exception:
    if os.environ.get("WEAVER_REQUIRE_NATIVE_PY", "0") in ("1", "true", "TRUE", "on", "ON"):
        raise
    # Profiling must never prevent the target program from starting.
    pass
