#!/bin/bash
set -e
SRC=/mnt/c/Users/Administrator/open-qingyi/paper/arr
BUILD=/root/arr-build
rm -rf "$BUILD"
mkdir -p "$BUILD"
cp "$SRC"/main.tex "$SRC"/custom.bib "$SRC"/acl.sty "$SRC"/acl_natbib.bst "$BUILD"/
cd "$BUILD"
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
bibtex main || true
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
pdflatex -interaction=nonstopmode main.tex | tail -5
echo "=== page count ==="
pdfinfo main.pdf | grep Pages
echo "=== errors ==="
grep -c "^!" main.log || true
echo "=== undefined refs/citations ==="
grep -c "undefined" main.log || true
echo "=== overfull ==="
grep -c "Overfull" main.log || true
cp main.pdf "$SRC"/main-arr.pdf
echo "copied back: $SRC/main-arr.pdf"
