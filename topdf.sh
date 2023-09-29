pdf="hw2.pdf"
md="README.md"

pandoc -V documentclass:article -V geometry:margin=.75in -V mainfont:'TeX Gyre Pagella' -V monofont:'TeX Gyre Cursor' -Vcolorlinks:true --pdf-engine=xelatex -o ${pdf} ${md}
