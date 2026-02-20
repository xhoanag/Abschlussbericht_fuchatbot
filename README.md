# Abschlussbericht_fuchatbot

Abschlussbericht – FU Chatbot

Setup

All team members should use the same setup:

-Clone the GitHub repository in VS Code

-Install LaTeX

Windows: MiKTeX

macOS: MacTeX

-Install the VS Code extension “LaTeX Workshop”

-Install Strawberry Perl 64bit

Run the following commands in the project root to compile the PDF:

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

This generates:

main.pdf
