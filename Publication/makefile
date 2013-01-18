#Makefile for assignment4 document

all: pmiind.pdf

pmiind.pdf: *.tex 
	latex pmiind
	bibtex pmiind
	latex pmiind 
	latex pmiind 
	dvipdf pmiind

clean: 
	rm -f *.toc *.log *.aux *~
