[![Version Badge](https://anaconda.org/conda-forge/phonopy/badges/version.svg)](https://anaconda.org/conda-forge/phonopy)
[![Downloads Badge](https://anaconda.org/conda-forge/phonopy/badges/downloads.svg)](https://anaconda.org/conda-forge/phonopy)
[![PyPI](https://img.shields.io/pypi/dm/phonopy.svg?maxAge=2592000)](https://pypi.python.org/pypi/phonopy)
[![codecov](https://codecov.io/gh/phonopy/phonopy/branch/develop/graph/badge.svg)](https://codecov.io/gh/phonopy/phonopy)

# Phonopy

Phonon code. Phonopy user documentation is found at
http://phonopy.github.io/phonopy/

## Mailing list for questions

Usual phonopy questions should be sent to phonopy mailing list
(https://sourceforge.net/p/phonopy/mailman/).

## Development

The development of phonopy is managed on the `develop` branch of github phonopy
repository.

- Github issues is the place to discuss about phonopy issues.
- Github pull request is the place to request merging source code.
- Python 3.7 will be the minimum requirement soon.
- Formatting is written in `pyproject.toml`.
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

- Use of pre-commit (https://pre-commit.com/) is encouraged.
  - Installed by `pip install pre-commit`, `conda install pre_commit` or see
    https://pre-commit.com/#install.
  - pre-commit hook is installed by `pre-commit install`.
  - pre-commit hook is run by `pre-commit run --all-files`.

## Documentation

Phonopy user documentation is written using python sphinx. The source files are
stored in `doc` directory. Please see how to write the documentation at
`doc/README.md`.

## How to run tests

You need pytest. At home directory of phonopy after setup,

```bash
% pytest
```
