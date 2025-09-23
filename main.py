#!/usr/bin/env python3
import argparse
import os
import fitz  # PyMuPDF
from pathlib import Path
import re
# --- reuse the search & annotate functions from earlier ---
DASH_CLASS = r"\-\u2010\u2011\u2012\u2013\u2014"  # -, ‐, -, ‒, –, —
SEP_CLASS = rf"(?:\s|[{DASH_CLASS}])"            # any whitespace or dash

def _compile_patterns(keywords, allow_concatenated=True, pluralize_last=False):
    """
    Build regex patterns that accept space/hyphen/whitespace between tokens.
    If allow_concatenated=True, tokens may also have *no* separator between them.
    If pluralize_last=True, add 's?' to the last token (helpful for 'clouds').
    """
    patterns = {}
    for kw in keywords:
        raw = kw.strip()
        # split on spaces/dashes (incl. Unicode dashes)
        tokens = [t for t in re.split(rf"[{DASH_CLASS}\s]+", raw) if t]
        if not tokens:
            continue
        esc = [re.escape(t) for t in tokens]
        sep = SEP_CLASS + (r"?" if allow_concatenated else r"+")
        patt = esc[0] + "".join(sep + t for t in esc[1:])
        if pluralize_last and len(esc) >= 1:
            patt += r"s?"  # very simple plural
        patterns[kw] = patt
    return patterns

# Allow separators between tokens (space or dashes), optionally allow no separator (concatenation),
# and permit trailing *suffix* on the LAST token (e.g., 'cloud' -> 'clouds', 'cloudy', etc.).
def _build_relaxed_phrase_pattern(phrase: str, allow_concatenated: bool = True) -> str:
    tokens = [t for t in re.split(rf"[{DASH_CLASS}\s]+", phrase.strip()) if t]
    if not tokens:
        return ""
    esc = [re.escape(t) for t in tokens]
    # separators between tokens
    sep = SEP_CLASS + (r"*" if allow_concatenated else r"+")  # * => also match concatenation
    # any non-separator suffix on the last token (letters/digits/punct until space/dash)
    SUFFIX = rf"[^\s{DASH_CLASS}]*"
    pattern = esc[0] + "".join(sep + t for t in esc[1:-1])
    pattern += sep + esc[-1] + SUFFIX if len(esc) > 1 else esc[-1] + SUFFIX
    return pattern


def _find_phrase_rects(page, phrase, case_sensitive=False, allow_suffix=False, allow_concat=False):
    """
    Find phrase occurrences using the page's word list (robust across weird spaces).
    If allow_suffix=True: last token may have extra trailing chars (e.g., 'cloud' -> 'clouds', 'cloudy').
    If allow_concat=True: also match when tokens are glued (or hyphenated) inside ONE word.
    """
    tokens = [t for t in re.split(r"\s+", phrase.strip()) if t]
    if not tokens:
        return []

    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
    words.sort(key=lambda w: (round(w[3], 1), w[0]))  # reading order

    def norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    toks = [norm(t) for t in tokens]
    n = len(words)
    m = len(toks)
    rects: list[fitz.Rect] = []

    # Unicode dash set for inside-a-word concatenations
    DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2212"  # -, ‐, -, ‒, –, —, −
    sep_inside = f"[{DASHES}]"
    between = f"{sep_inside}*" if allow_concat else f"{sep_inside}+"

    # pattern for "all tokens inside a single word", with optional suffix on last token
    if m >= 2:
        pat = re.escape(tokens[0])
        for t in tokens[1:]:
            pat += f"(?:{between}){re.escape(t)}"
        if allow_suffix:
            pat += r".*"  # allow anything after the last token
        pat_singleword = re.compile(pat, 0 if case_sensitive else re.IGNORECASE)
    else:
        pat_singleword = None

    i = 0
    while i < n:
        wtxt = norm(words[i][4])

        # 1) Try normal multi-word span across m consecutive words
        if m >= 2 and i + m - 1 < n:
            # first m-1 must match exactly
            ok = True
            for k in range(m - 1):
                if norm(words[i + k][4]) != toks[k]:
                    ok = False
                    break
            if ok:
                last = norm(words[i + m - 1][4])
                last_ok = last.startswith(toks[-1]) if allow_suffix else (last == toks[-1])
                if last_ok:
                    x0 = min(words[i + k][0] for k in range(m))
                    y0 = min(words[i + k][1] for k in range(m))
                    x1 = max(words[i + k][2] for k in range(m))
                    y1 = max(words[i + k][3] for k in range(m))
                    rects.append(fitz.Rect(x0, y0, x1, y1))
                    i += m
                    continue

        # 2) If allowed, try the "single glued/hyphenated word" variant
        if allow_concat and m >= 2 and pat_singleword and pat_singleword.match(wtxt):
            x0, y0, x1, y1 = words[i][0], words[i][1], words[i][2], words[i][3]
            rects.append(fitz.Rect(x0, y0, x1, y1))
            i += 1
            continue

        i += 1

    return rects


def search_and_highlight(
    doc, keywords, case_sensitive=False, color_map=None, opacity=0.3,
    allow_concatenated=False, no_whole_word=False
):
    """
    Return structure:
      {
        "molecular cloud": {23: 5, 41: 2, ...},
        "CO":               {12: 3, 19: 1, ...},
      }
    Also applies highlights (with optional per-keyword color/opacity).
    """
    IGNORECASE = getattr(fitz, "TEXT_IGNORECASE", 0)
    flags = 0 if case_sensitive else IGNORECASE

    # kw -> {page -> count}
    results = {kw: {} for kw in keywords}

    for pno, page in enumerate(doc, start=1):
        for kw in keywords:
            s = kw.strip()

            # MULTI-WORD (space-separated) → robust word-list matcher
            if re.search(r"\s", s):
                rects = _find_phrase_rects(
                    page, s,
                    case_sensitive=case_sensitive,
                    allow_suffix=no_whole_word,      # allow suffix on last word if requested
                    allow_concat=allow_concatenated  # allow glued/hyphenated inside one token
                )
            else:
                # SINGLE WORD: PyMuPDF literal search (substring by default, i.e., 'abc'→'abcd')
                rects = page.search_for(s, flags=flags)

            if rects:
                # store count for this page
                results[kw][pno] = len(rects)

                # add highlights (respect per-keyword color / opacity if provided)
                rgb = (color_map or {}).get(kw)
                for r in rects:
                    annot = page.add_highlight_annot(r)
                    if rgb:
                        annot.set_colors(stroke=rgb, fill=rgb)
                        annot.set_opacity(opacity)
                        annot.update()
    return results



def add_toc(doc, matches):
    """
    matches: {keyword: {page: count}}
    Creates level-1 bookmark per keyword with total hits,
    and level-2 bookmarks per page: 'page P — N hits'.
    """
    toc = doc.get_toc(simple=True) or []
    for kw, page_counts in matches.items():
        if not page_counts:
            continue
        first_page = min(page_counts.keys())
        total_hits = sum(page_counts.values())
        toc.append([1, f"Keyword: {kw} — {total_hits} hits", first_page])
        for p in sorted(page_counts.keys()):
            n = page_counts[p]
            toc.append([2, f"{kw}, page: {p}, hits: {n}", p])
    doc.set_toc(toc)

# --- main CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Search keywords in a PDF, highlight them, and add bookmarks."
    )
    parser.add_argument("-i", "--input", required=True, help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output PDF file (default: input_k.pdf)")
    parser.add_argument(
        "-k", "--keywords", nargs="+", help="List of keywords to search (quoted if multi-word)"
    )
    parser.add_argument(
        "-f", "--keywords-file", help="Text file with keywords (one per line)"
    )
    parser.add_argument(
        "-c", "--case-sensitive", action="store_true", help="Case sensitive search"
    )
    parser.add_argument(
        "--no-whole-word",
        action="store_true",
        help="For multi-word phrases, use flexible substring/regex matching instead of whole-word-only. (e.g. 'molecular cloud' finds also 'molecular clouds'"
    )
    parser.add_argument(
        "--allow-concat",
        dest="allow_concat",
        action="store_true",
        help="Allow concatenation between phrase tokens (e.g.,'photo dissociation' also finds 'photo-dissociation', 'photodissociation').",
    )

    args = parser.parse_args()

    # collect keywords from CLI and/or file
    keywords = []
    if args.keywords:
        keywords.extend(args.keywords)
    if args.keywords_file:
        with open(args.keywords_file, encoding="utf-8") as f:
            # strip whitespace, skip blanks and comments
            for line in f:
                kw = line.strip()
                if kw and not kw.startswith("#"):
                    keywords.append(kw)

    if not keywords:
        parser.error("No keywords specified. Use -k or -f.")

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_k.pdf")

    doc = fitz.open(in_path)
    matches = search_and_highlight(
        doc, keywords,
        case_sensitive=args.case_sensitive,
        no_whole_word=getattr(args, "no_whole_word", False),  # if you added this earlier
        allow_concatenated=args.allow_concat,
    )
    add_toc(doc, matches)
    doc.save(out_path, garbage=4, deflate=True)
    doc.close()

    print(f"Processed {in_path} -> {out_path}")
    for kw, pages in matches.items():
        print(f"'{kw}' found on pages {list(pages.keys())}")

if __name__ == "__main__":
    main()
