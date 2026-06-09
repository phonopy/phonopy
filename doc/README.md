# How to write phonopy documentation

This directory contains the python-sphinx documentation source.

## How to compile

```
make html
```

This is equivalent to what the CI runs:

```
sphinx-build doc docs_build
```

## Source files

* `conf.py` contains the sphinx configuration.
* `*.md` are the documentation source written in MyST Markdown (the main
  format). The filenames without `.md` are the keys used to link from the
  toctree, mainly in `index.md`.
* `*.rst` are reStructuredText source files, still used for some pages such
  as the calculator interfaces.
* `*.inc` are files included from other source files.

## How to publish

Publishing is automated. Pushing to the `publish-gh-pages` branch triggers the
`publish-gh-pages.yml` GitHub Actions workflow, which builds the documentation
with `sphinx-build` and deploys it to the `gh-pages` branch. No manual copying
is needed.
