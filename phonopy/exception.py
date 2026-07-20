# SPDX-License-Identifier: BSD-3-Clause
"""Phonopy exceptions."""


class ForcesetsNotFoundError(RuntimeError):
    """Exception when forces not found in Phonopy class instance."""

    pass


class ForceCalculatorRequiredError(RuntimeError):
    """Exception when force calculator is required to compute force constants."""

    pass


class ForceConstantsCalculatorNotFoundError(RuntimeError):
    """Exception when force constants calculator is not found."""

    pass


class CellNotFoundError(RuntimeError):
    """Exception when unit cell not found."""

    pass


class MagmomValueError(ValueError):
    """Exception when magnetic moment value is not valid."""

    pass


class PypolymlpFileNotFoundError(RuntimeError):
    """Exception when pypolymlp file is not found."""

    pass


class PypolymlpTrainingDatasetNotFoundError(RuntimeError):
    """Exception when pypolymlp dataset is not found."""

    pass


class PypolymlpRelaxationError(RuntimeError):
    """Exception when relaxation of atomic positions by pypolymlp fails."""

    pass


class PypolymlpDevelopmentError(RuntimeError):
    """Exception when development of pypolymlp fails."""

    pass


class BORNFileParseError(RuntimeError):
    """Exception when parsing BORN file fails."""

    pass
