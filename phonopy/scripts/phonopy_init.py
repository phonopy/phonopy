# SPDX-License-Identifier: BSD-3-Clause
from phonopy.cui.phonopy_script import main


def run():
    """Run phonopy-init script."""
    argparse_control = {
        "load_phonopy_yaml": False,
        "mode": "init",
    }
    main(**argparse_control)
