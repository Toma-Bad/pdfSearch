# PDF Keyword Search and Highlight (using PyMuPDF)

Command-line tool to search a PDF for keywords, highlight matches, and add bookmarks (TOC) showing total hits per keyword and per-page hit counts.  
Multi-word phrases are matched robustly using the PDF word list. You can optionally allow suffixes on the last word and allow concatenated/hyphenated forms.

python main.py -i <input.pdf> [-o <output.pdf>] \
  [-k <kw1> <kw2> ...] [-f keywords.txt] \
  [-c] [--no-whole-word] [--allow-concat]

-i, --input (required): Input PDF.

-o, --output: Output PDF. Defaults to <input>_k.pdf.

-k, --keywords: Space-separated keywords. Quote multi-word phrases.

-f, --keywords-file: Text file with one keyword per line (# starts a comment).

-c, --case-sensitive: Case-sensitive search (default is case-insensitive).

--no-whole-word: For phrases, also match suffixes on the last word
(e.g., "molecular cloud" matches "molecular clouds").

--allow-concat: For phrases, also match glued/hyphenated forms inside one token
(e.g., photo-dissociation, photodissociation).
