# Build the manuscript.  Requires pdflatex + bibtex (TeX Live / TinyTeX).
MAIN = main_en

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib numbers.tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Refresh numbers.tex and the figures from results/ before rebuilding.
refresh:
	cd .. && $(PYTHON) src/make_plots.py && $(PYTHON) src/make_numbers.py
PYTHON ?= python3

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg $(MAIN).abs

.PHONY: all refresh clean
