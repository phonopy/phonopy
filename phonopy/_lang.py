# SPDX-License-Identifier: BSD-3-Clause
"""Dispatch logging helper for the C/Rust backend switch.

The dedicated ``phonopy.lang`` logger records which backend is selected
when a lang-aware routine is entered.  It stays silent by default.

To see the messages, either:

* set the environment variable ``PHONOPY_TRACE_LANG=1`` before running
  (this configures a ``StreamHandler`` to stderr at DEBUG level), or
* configure ``logging.getLogger("phonopy.lang")`` from client code.

"""

from __future__ import annotations

import logging
import os
from typing import Literal

_logger = logging.getLogger("phonopy.lang")

if os.environ.get("PHONOPY_TRACE_LANG"):
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("[phonopy.lang] %(message)s"))
        _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)


def log_dispatch(lang: str, name: str) -> None:
    """Emit a dispatch-level trace line for a lang-aware call site."""
    _logger.debug("dispatch name=%s lang=%s", name, lang)


def have_c_ext() -> bool:
    """Return True when the ``phonopy._phonopy`` C extension is importable."""
    try:
        import phonopy._phonopy  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        return False
    return True


def have_phonors() -> bool:
    """Return True when the ``phonors`` Rust extension is importable."""
    try:
        import phonors  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        return False
    return True


def c_use_openmp() -> bool:
    """Return ``phonoc.use_openmp()`` or ``False`` if C extension is absent."""
    try:
        import phonopy._phonopy as phonoc  # type: ignore[import-untyped]
    except ImportError:
        return False
    return bool(phonoc.use_openmp())


def c_omp_max_threads() -> int:
    """Return ``phonoc.omp_max_threads()`` or ``0`` if C extension is absent."""
    try:
        import phonopy._phonopy as phonoc  # type: ignore[import-untyped]
    except ImportError:
        return 0
    return int(phonoc.omp_max_threads())


def rust_rayon_max_threads() -> int:
    """Return ``phonors.rayon_max_threads()`` or ``0`` if Rust backend is absent."""
    try:
        import phonors  # type: ignore[import-untyped]
    except ImportError:
        return 0
    fn = getattr(phonors, "rayon_max_threads", None)
    if fn is None:
        return 0
    return int(fn())


_fallback_warned = False


def resolve_lang(lang: Literal["C", "Rust"]) -> Literal["C", "Rust"]:
    """Pick the best available backend, falling back from C to Rust.

    When ``lang == "C"`` but the C extension is not installed (e.g. a
    ``PHONOPY_NO_C_EXT=1`` build), flip to ``"Rust"`` and emit a one-time
    informational message.  Raise ``ImportError`` if neither backend is
    available, or if Rust is requested without ``phonors`` installed.

    """
    global _fallback_warned

    if lang == "Rust":
        if not have_phonors():
            raise ImportError(
                "lang='Rust' was requested but the `phonors` package is not "
                "installed.  Install it (e.g. `pip install phonors`) or use "
                "lang='C'."
            )
        return "Rust"

    if have_c_ext():
        return "C"

    if have_phonors():
        if not _fallback_warned:
            print(
                "[phonopy] C extension `phonopy._phonopy` is not available; "
                "falling back to lang='Rust' via the `phonors` package."
            )
            _fallback_warned = True
        return "Rust"

    raise ImportError(
        "Neither the `phonopy._phonopy` C extension nor the `phonors` Rust "
        "package is importable.  Reinstall phonopy with C support, or "
        "install `phonors`."
    )
