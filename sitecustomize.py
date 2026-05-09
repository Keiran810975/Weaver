"""Optional Weaver auto-bootstrap.

This module is imported by CPython when the repository root is on PYTHONPATH.
It stays inert unless WEAVER_AUTO_PROFILE=1 is present, which lets the launcher
turn on CPython profile collection without touching user training code.
"""

try:
    from weaver.collector.py_runtime import enable_from_env

    enable_from_env()
except Exception:
    # Profiling must never prevent the target program from starting.
    pass
