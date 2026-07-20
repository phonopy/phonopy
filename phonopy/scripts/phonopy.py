# SPDX-License-Identifier: BSD-3-Clause
from phonopy.cui.phonopy_script import main


def run():
    """Run phonopy script."""
    argparse_control = {
        "load_phonopy_yaml": True,
        "mode": "run",
    }
    main(**argparse_control)
