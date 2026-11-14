"""Tests of phonopy settings parser."""

from __future__ import annotations

import pathlib
import re

import pytest

from phonopy.cui.phonopy_argparse import get_init_parser
from phonopy.cui.settings import PhonopyConfParser, parse_magmom
from phonopy.exception import MagmomValueError

_magmom_expansion_cases = [
    ("1 1 -1 -1", [1.0, 1.0, -1.0, -1.0]),
    ("2*1 2*-1", [1.0, 1.0, -1.0, -1.0]),
    ("3*0 1", [0.0, 0.0, 0.0, 1.0]),
    ("2*0 2", [0.0, 0.0, 2.0]),
    ("1*0.5", [0.5]),
]

_magmom_invalid_tokens = ["2*", "*1", "a*1", "2.5*1", "0*1", "1*2*3", "abc"]


def _parse_magmom_conf_file(magmom: str, tmp_path: pathlib.Path) -> PhonopyConfParser:
    conf_filename = tmp_path / "phonopy.conf"
    conf_filename.write_text(f"MAGMOM = {magmom}\n")
    return PhonopyConfParser(filename=conf_filename)


def _parse_magmom_option(magmom: str) -> PhonopyConfParser:
    parser, _ = get_init_parser()
    args = parser.parse_args(["--magmom", *magmom.split()])
    return PhonopyConfParser(args=args)


@pytest.mark.parametrize("magmom,expected", _magmom_expansion_cases)
def test_parse_magmom(magmom: str, expected: list[float]):
    """parse_magmom expands n*value."""
    assert parse_magmom(magmom) == expected


@pytest.mark.parametrize("token", _magmom_invalid_tokens)
def test_parse_magmom_invalid(token: str):
    """Invalid token given to parse_magmom raises MagmomValueError."""
    with pytest.raises(MagmomValueError, match=re.escape(f'"{token}"')):
        parse_magmom(f"1 {token}")


@pytest.mark.parametrize("magmom,expected", _magmom_expansion_cases)
def test_magmom_conf_file(magmom: str, expected: list[float], tmp_path):
    """MAGMOM tag in configuration file expands n*value."""
    conf_parser = _parse_magmom_conf_file(magmom, tmp_path)
    assert conf_parser.settings.magnetic_moments == expected


@pytest.mark.parametrize("magmom,expected", _magmom_expansion_cases)
def test_magmom_option(magmom: str, expected: list[float]):
    """--magmom option expands n*value."""
    conf_parser = _parse_magmom_option(magmom)
    assert conf_parser.settings.magnetic_moments == expected


@pytest.mark.parametrize("token", _magmom_invalid_tokens)
def test_magmom_conf_file_invalid(token: str, tmp_path):
    """Invalid token of MAGMOM tag raises MagmomValueError."""
    with pytest.raises(MagmomValueError, match=re.escape(f'"{token}"')):
        _parse_magmom_conf_file(f"1 {token}", tmp_path)


@pytest.mark.parametrize("token", _magmom_invalid_tokens)
def test_magmom_option_invalid(token: str):
    """Invalid token of --magmom option raises MagmomValueError."""
    with pytest.raises(MagmomValueError, match=re.escape(f'"{token}"')):
        _parse_magmom_option(f"1 {token}")


_modulation_header_cases = [
    ("2 2 2", [2, 2, 2], None),
    ("2 2 2 0.01 0 0", [2, 2, 2], [0.01, 0.0, 0.0]),
    ("-1 1 1 1 -1 1 1 1 -1", [-1, 1, 1, 1, -1, 1, 1, 1, -1], None),
    (
        "-1 1 1 1 -1 1 1 1 -1 0.01 0 0",
        [-1, 1, 1, 1, -1, 1, 1, 1, -1],
        [0.01, 0.0, 0.0],
    ),
]


def _parse_modulation_conf_file(header: str, tmp_path: pathlib.Path) -> dict:
    conf_filename = tmp_path / "phonopy.conf"
    conf_filename.write_text(f"MODULATION = {header}, 0 1/2 1/2 1 2\n")
    modulation = PhonopyConfParser(filename=conf_filename).settings.modulation
    assert modulation is not None
    return modulation


@pytest.mark.parametrize("header,dimension,delta_q", _modulation_header_cases)
def test_modulation_conf_file(
    header: str, dimension: list[int], delta_q: list[float] | None, tmp_path
):
    """MODULATION header gives supercell dimension and optional delta_q."""
    modulation = _parse_modulation_conf_file(header, tmp_path)
    assert modulation["dimension"] == dimension
    assert modulation.get("delta_q") == delta_q
    assert modulation["modulations"] == [[[0.0, 0.5, 0.5], 0, 2.0, 0]]


@pytest.mark.parametrize(
    "header", ["2 2 2 0.01", "2 2 2 0.01 0 0 2", "-1 1 1 1 -1 1 1 1 -1 0.01 0 0 2"]
)
def test_modulation_conf_file_wrong_header(header: str, tmp_path):
    """MODULATION header with a wrong number of values is rejected.

    The header used to accept the derivative order of the dynamical matrix
    after delta_q.  This value is no longer accepted.

    """
    with pytest.raises(SystemExit):
        _parse_modulation_conf_file(header, tmp_path)
