# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the phonopy-mlpsscha command."""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.cui.phonopy_mlpsscha_script import main
from phonopy.sscha.run import SSCHARun, read_sscha_run_hdf5

cwd = pathlib.Path(__file__).parent

ITERATIONS = 2
TEMPERATURE = 300.0
TRANSIENT = 1


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[SSCHARun, str]:
    """Run the command once and return the file it wrote and what it printed.

    The run costs an MLP evaluation per snapshot and a force-constant fit per
    iteration, so it is made once and read by every test below.

    """
    pytest.importorskip("pypolymlp")
    output = tmp_path_factory.mktemp("mlpsscha") / "sscha.hdf5"
    argv = [
        "phonopy-mlpsscha",
        str(cwd / ".." / "phonopy_KCl.yaml"),
        "--mlp",
        str(cwd / ".." / "polymlp_KCL-120.yaml"),
        "-t",
        str(TEMPERATURE),
        "--snapshots",
        "10",
        "--iterations",
        str(ITERATIONS),
        "--mesh",
        "10",
        "--random-seed",
        "42",
        "--transient",
        str(TRANSIENT),
        "-o",
        str(output),
        "-v",
    ]
    log = io.StringIO()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        with contextlib.redirect_stdout(log):
            main()
    return read_sscha_run_hdf5(output), log.getvalue()


def test_cli_writes_one_row_per_iteration(cli_run: tuple[SSCHARun, str]) -> None:
    """The file holds every iteration and no average.

    phonopy_KCl.yaml carries no force constants, so the run opens with an
    initialization step. That step draws its displacements at --distance
    rather than from a canonical ensemble and records no free energy, which
    leaves one row per --iterations.

    """
    run, _ = cli_run
    assert run.n_iterations == ITERATIONS
    for name in SSCHARun.PER_ITERATION:
        values = getattr(run, name)
        assert values.shape == (ITERATIONS,)
        assert np.all(np.isfinite(values))
    assert np.all(run.errors > 0)


def test_cli_records_the_temperature_and_the_cell(
    cli_run: tuple[SSCHARun, str], ph_kcl: Phonopy
) -> None:
    """-t and the cell of the input file reach the file.

    The lattice lengths are what lets a sweep place the run on its grid, and
    they are the input cell's, not the supercell's.

    """
    run, _ = cli_run
    assert run.temperature == pytest.approx(TEMPERATURE)
    np.testing.assert_allclose(
        run.lattice_lengths, np.linalg.norm(ph_kcl.unitcell.cell, axis=1)
    )
    assert np.isfinite(run.reference_energy)


def test_cli_lists_the_iterations_and_marks_the_transient(
    cli_run: tuple[SSCHARun, str],
) -> None:
    """-v prints the listing the transient is chosen from.

    The listing is the command's only report: which iterations to average is
    decided from it afterwards, and --transient only marks them here.

    """
    _, log = cli_run
    assert "(F - mean)/error" in log
    # "  {i + 1:4d}{mark}": iteration 1 is inside the transient, 2 is not.
    assert "     1*" in log
    assert "     2 " in log
    assert "left out as the transient" in log
    assert "Wrote " in log
