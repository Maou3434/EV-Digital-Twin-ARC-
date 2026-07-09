.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex
	latexmk -pdf main.tex || (pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex)

clean:
	latexmk -c
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.run.xml *.bcf
