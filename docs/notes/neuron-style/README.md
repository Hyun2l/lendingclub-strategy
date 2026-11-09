# Neuron document style

- Fonts: Charter (Latin text and math letters, `fonts/*.otf`) + Noto Serif CJK KR (Hangul, system font). Sans/mono: Latin Modern.
- Boxes: white with thin lines only (no filled or grey boxes). keybox / formulabox / workout / qa / warnbox / whybox / prof / insight / career.
- Grayscale design. Booktabs tables. Bold headings, header shows chapter name in italics.
- Notes: `notes_template.tex` (report, 11pt, 24mm margins). Cheat/problem sheets: `sheet_template.tex` (10pt, 2 columns, 12mm margins, answers right under each problem with \Q / \A).

Build: `xelatex notes_template.tex` (twice). Requires Noto Serif CJK KR installed.
If fonts/*.otf are missing, rebuild with `python3 fonts/build_charter.py` (needs afdko, fonttools, TeX Live bchr8a fonts).

Gotchas
- Run after writing: `perl -CSD -Mutf8 -pi -e 's/(\\[a-zA-Z]+)(?=[\x{AC00}-\x{D7A3}])/$1\{\}/g' file.tex` (macro followed by Hangul).
- Charter lacks arrows/Greek/★: use \ensuremath{\to}, \ensuremath{\star}, $\sigma$ etc. Never put Korean or \ensuremath inside lstlisting.
