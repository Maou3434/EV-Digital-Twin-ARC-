# EV Battery Thermal Digital Twin - IEEE Conference Paper

This directory contains the LaTeX source files and configuration for the publication-ready IEEE conference paper documenting the EV Battery Thermal Digital Twin.

## Directory Structure

```text
paper/
├── main.tex             # Main LaTeX file
├── references.bib       # References database
├── IEEEtran.cls         # IEEEtran document class file
├── Makefile             # Automated build configuration
├── README.md            # This documentation file
├── sections/            # Individual paper sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── future_work.tex
│   └── conclusion.tex
├── figures/             # Figure assets (loss, trajectory, residuals, etc.)
└── tables/              # Structural tables
```

## Compilation

To compile the paper into a PDF, you must have a LaTeX distribution (such as TeX Live, MacTeX, or MiKTeX) installed.

### Standard Build (Makefile)

If you have `make` installed:
```bash
make
```

### Direct Compilation (latexmk)

Compile using `latexmk` (recommended):
```bash
latexmk -pdf main.tex
```

### Manual Compilation Sequence

If `latexmk` is not available, run the following sequence:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Cleaning Auxiliary Files

To clean up standard LaTeX intermediate files (such as `.aux`, `.log`, `.bbl`):
```bash
make clean
```
or
```bash
latexmk -c
```
