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
