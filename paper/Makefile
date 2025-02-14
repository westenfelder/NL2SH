# apt install latexmk texlive-full
# pip install git+https://github.com/acl-org/aclpubcheck
# aclpubcheck --paper_type long paper.pdf

PAPER = paper
TEXFILES = $(wildcard *.tex)
FIGURES = $(wildcard figs/*.png)
BIBFILE = $(wildcard *.bib)

${PAPER}.pdf: ${PAPER}.tex ${TEXFILES} ${BIBFILE} ${FIGURES}
	latexmk -pdf ${PAPER}

clean:
	latexmk -C ${PAPER}
