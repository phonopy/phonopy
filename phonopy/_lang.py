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
