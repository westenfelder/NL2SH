# apt install latexmk texlive-full
# pip install git+https://github.com/acl-org/aclpubcheck
# aclpubcheck --paper_type long paper.pdf

all:
	latexmk -pdf paper.tex

clean:
	latexmk -C paper.tex

.PHONY: all clean