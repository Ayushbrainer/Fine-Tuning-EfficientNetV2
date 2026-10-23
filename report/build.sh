#!/bin/bash
# Compile report.tex -> report.pdf with tectonic from DeepEnv (no latexmk/pdflatex on this machine).
cd "$(dirname "$0")"
/home/msds/ayushnik001/.conda/envs/DeepEnv/bin/tectonic report.tex
pdfinfo report.pdf | grep Pages
