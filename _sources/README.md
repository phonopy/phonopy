# How to write phonopy documentation

This directory contains python-sphinx documentation source.

## How to compile

```
make html
```

## Source files

* `conf.py` contains the sphinx setting confiuration.
* `*.rst` are the usual sphinx documentation source and the filenames without `.rst` are the keys to link from toctree mainly in `index.rst`.
* `*.inc` are the files included in the other `*.rst` files.

## How to publish

Web page files are copied to `gh-pages` branch. At the phonopy github top directory,
```
git checkout gh-pages
rm -r .buildinfo .doctrees *
```

From the directory the sphinx doc is complied,
```
rsync -avh _build/ <phonopy-repository-directory>/
```

Again, at the phonopy github top directory,
```
git add .
git commit -a -m "Update documentation ..."
git push
```
