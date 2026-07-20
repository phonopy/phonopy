#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause

from phonopy.cui.phonopy_script import main


def run():
    """Run phonopy-load script."""
    argparse_control = {
        "load_phonopy_yaml": True,
        "mode": "run",
        "deprecated_command": "phonopy-load",
    }
    main(**argparse_control)
