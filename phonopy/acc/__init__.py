try:
    from phonopy.acc.numba_imports import use_acc
    from phonopy.acc.phonon.qpoints import run_qpoints_phonon
except ImportError:
    from phonopy.acc.placeholder import use_acc, run_qpoints_phonon

__all__ = ["use_acc", "run_qpoints_phonon"]
