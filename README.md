[![Version Badge](https://anaconda.org/conda-forge/phonopy/badges/version.svg)](https://anaconda.org/conda-forge/phonopy)
[![Downloads Badge](https://anaconda.org/conda-forge/phonopy/badges/downloads.svg)](https://anaconda.org/conda-forge/phonopy)
[![PyPI](https://img.shields.io/pypi/dm/phonopy.svg?maxAge=2592000)](https://pypi.python.org/pypi/phonopy)
[![codecov](https://codecov.io/gh/phonopy/phonopy/branch/develop/graph/badge.svg)](https://codecov.io/gh/phonopy/phonopy)

# Phonopy

Phonon code. Phonopy user documentation is found at
http://phonopy.github.io/phonopy/

## Installation

See https://phonopy.github.io/phonopy/install.html.

## Dependency

- python>=3.8
- numpy>=1.17.0
- PyYAML>=5.3
- matplotlib>=2.2.2
- h5py>=3.0
- spglib>=2.0
- scipy (optional)
- seekpath (optional)

## Mailing list for questions

Usual phonopy questions should be sent to phonopy mailing list
(https://sourceforge.net/p/phonopy/mailman/).

## Development

The development of phonopy is managed on the `develop` branch of github phonopy
repository.

- Github issues is the place to discuss about phonopy issues.
- Github pull request is the place to request merging source code.

### Formatting

Formatting rule is written in `pyproject.toml`.

### pre-commit

Pre-commit (https://pre-commit.com/) is mainly used for applying the formatting
rule automatically. Therefore, the use is strongly encouraged at or before
git-commit. Pre-commit is set-up and used in the following way:

- Installed by `pip install pre-commit`, `conda install pre_commit` or see
  https://pre-commit.com/#install.
- pre-commit hook is installed by `pre-commit install`.
- pre-commit hook is run by `pre-commit run --all-files`.

Unless running pre-commit, pre-commit.ci may push the fix at PR by github
action. In this case, the fix should be merged by the contributor's repository.
### VSCode setting
- Not strictly, but VSCode's `settings.json` may be written like

  ```json
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=88", "--ignore=E203,W503"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.pycodestyleEnabled": false,
  "python.linting.pydocstyleEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.sortImports.args": ["--profile", "black"],
  "[python]": {
      "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
  }
  ```

## Documentation

Phonopy user documentation is written using python sphinx. The source files are
stored in `doc` directory. Please see how to write the documentation at
`doc/README.md`.

## Tests

Tests are written using pytest. To run tests, pytest has to be installed. The
tests can be run by

```bash
% pytest
```
