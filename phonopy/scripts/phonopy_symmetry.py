# SPDX-License-Identifier: BSD-3-Clause
from phonopy.cui.phonopy_script import main


def run():
    """Run phonopy-symmetry script."""
    argparse_control = {
        "load_phonopy_yaml": False,
        "mode": "symmetry",
    }
    main(**argparse_control)
