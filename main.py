#!/usr/bin/env python3
import argparse
import os
import fitz  # PyMuPDF
from pathlib import Path
import re
# --- reuse the search & annotate functions from earlier ---
DASH_CLASS = r"\-\u2010\u2011\u2012\u2013\u2014"  # -, ‐, -, ‒, –, —
SEP_CLASS = rf"(?:\s|[{DASH_CLASS}])"            # any whitespace or dash
import unicodedata

ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}  # ZWSP, ZWNJ, ZWJ, BOM
def _find_singleword_rects(page, term, case_sensitive=False):
    """
    Single-word finder with ligature-safe normalization and substring match.
    Keeps your 'abc' -> 'abcd' behavior.
    """
    term_n = normalize_text(term, case_sensitive=case_sensitive)
    if not term_n:
        return []
    words = page.get_text("words")
    words.sort(key=lambda w: (round(w[3], 1), w[0]))
    rects = []
    for (x0, y0, x1, y1, txt, *_rest) in words:
        w_n = normalize_text(txt, case_sensitive=case_sensitive)
        if term_n in w_n:
            rects.append(fitz.Rect(x0, y0, x1, y1))
    return rects

def normalize_text(s: str, *, case_sensitive: bool) -> str:
    """
    Compatibility-normalize to expand ligatures etc. and drop zero-width chars.
    Example: 'difﬁcult' (d i ﬁ c …) -> 'difficult'
    """
    if not s:
        return s
    t = unicodedata.normalize("NFKC", s)  # expands ﬁ, ﬂ, ﬀ, œ, æ, …
    t = "".join(ch for ch in t if ch not in ZERO_WIDTH)
    return t if case_sensitive else t.lower()
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
    Multi-word phrase finder using normalized tokens so ligatures (ﬁ) match 'fi'.
    """
    tokens_raw = [t for t in re.split(r"\s+", phrase.strip()) if t]
    if not tokens_raw:
        return []

    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
    words.sort(key=lambda w: (round(w[3], 1), w[0]))

    # normalized word texts (ligatures expanded, zero-width removed, case folded if needed)
    w_norm = [normalize_text(w[4], case_sensitive=case_sensitive) for w in words]
    toks   = [normalize_text(t, case_sensitive=case_sensitive) for t in tokens_raw]

    rects: list[fitz.Rect] = []
    n, m = len(words), len(toks)

    DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2212"
    sep_inside = f"[{DASHES}]"
    between = f"{sep_inside}*" if allow_concat else f"{sep_inside}+"

    # single-word-glued pattern on *normalized* tokens
    if m >= 2:
        pat = re.escape(toks[0])
        for t in toks[1:]:
            pat += f"(?:{between}){re.escape(t)}"
        if allow_suffix:
            pat += r".*"
        flags = 0 if case_sensitive else re.IGNORECASE
        pat_singleword = re.compile(pat, flags)
    else:
        pat_singleword = None

    i = 0
    while i < n:
        # 1) span across m consecutive words (normalized equality)
        if m >= 2 and i + m - 1 < n:
            ok = True
            for k in range(m - 1):
                if w_norm[i + k] != toks[k]:
                    ok = False
                    break
            if ok:
                last = w_norm[i + m - 1]
                last_ok = last.startswith(toks[-1]) if allow_suffix else (last == toks[-1])
                if last_ok:
                    x0 = min(words[i + k][0] for k in range(m))
                    y0 = min(words[i + k][1] for k in range(m))
                    x1 = max(words[i + k][2] for k in range(m))
                    y1 = max(words[i + k][3] for k in range(m))
                    rects.append(fitz.Rect(x0, y0, x1, y1))
                    i += m
                    continue

        # 2) single glued/hyphenated token (normalized regex)
        if allow_concat and m >= 2 and pat_singleword and pat_singleword.match(w_norm[i]):
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
                    allow_suffix=no_whole_word,
                    allow_concat=allow_concatenated
                )
            else:
                # WAS: rects = page.search_for(s, flags=flags)
                rects = _find_singleword_rects(page, s, case_sensitive=case_sensitive)

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

# --- term discovery (per your rules) ---
CHARACTERS_TO_STRIP = "()'\":,.”“‘?;-•’—…[]!.´"
 # strip from word edges for checks
PHRASES_TO_STRIP = ["'s", "’s", "'re", "'ve", "'t",
                    "[0]", "[1]", "[2]", "[3]", "[4]", "[5]", "[6]"]

_URL_RE   = re.compile(r"^(?:https?://|www\.)", re.I)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", re.I)
_DOI_RE   = re.compile(r"^10\.\d{4,9}/", re.I)
_DASHES   = "-\u2010\u2011\u2012\u2013\u2014\u2212"  # -, ‐, -, ‒, –, —, −

def _load_common_words(path: str) -> set[str]:
    """Load common words; normalize (NFKC + lower) so ligatures and case match discovery."""
    common = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            w = normalize_text(raw, case_sensitive=False)
            if w:
                common.add(w)
    return common


def _strip_edges_and_suffixes(token: str) -> str:
    """Strip configured edge chars and trailing phrases before checks/length."""
    if not token:
        return token
    s = token
    # strip leading/trailing edge characters
    while s and s[0] in CHARACTERS_TO_STRIP:
        s = s[1:]
    while s and s[-1] in CHARACTERS_TO_STRIP:
        s = s[:-1]
    # strip one trailing configured phrase if present
    for suf in PHRASES_TO_STRIP:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s

def _is_linkish(s: str) -> bool:
    return bool(_URL_RE.match(s) or _EMAIL_RE.match(s) or _DOI_RE.match(s) or s.endswith(".com"))

def _join_hyphenated_word_tokens(words):
    """
    Merge end-of-line hyphenations from page.get_text('words'):
    ('molecu-', 'lar') -> 'molecular'
    """
    out = []
    i = 0
    while i < len(words):
        x0, y0, x1, y1, txt, b, ln, wn = words[i]
        if txt and txt[-1] in _DASHES and i + 1 < len(words):
            nx0, ny0, nx1, ny1, ntxt, nb, nln, nwn = words[i + 1]
            merged_txt = txt[:-1] + ntxt  # drop hyphen, join
            out.append((x0, y0, max(x1, nx1), max(y1, ny1), merged_txt, b, ln, wn))
            i += 2
            continue
        out.append(words[i])
        i += 1
    return out

from collections import Counter

from collections import Counter

def discover_terms(
    doc,
    *,
    common_words_file: str,
    min_len: int = 3,
    case_sensitive: bool = False,
    limit_terms: int | None = None,   # stop after this many UNIQUE terms
    max_pages: int | None = None,     # scan at most this many pages (from start_page)
    start_page: int | None = None,    # 1-based page to start scanning
) -> list[str]:
    """
    Discover candidate terms with:
      - length >= min_len AFTER stripping + normalization
      - NOT in common words (dictionary normalized the same way)
      - not starting with a digit, not linkish
      - hyphenated line-breaks joined
    Honors:
      - start_page (1-based), max_pages
      - limit_terms (unique term cap; early returns as soon as reached)
    """
    common = _load_common_words(common_words_file)
    freq: Counter[str] = Counter()

    start = start_page or 1
    scanned = 0

    for pno, page in enumerate(doc, start=1):
        if pno < start:
            continue
        if max_pages is not None and scanned >= max_pages:
            break
        scanned += 1

        words = page.get_text("words")
        words.sort(key=lambda w: (round(w[3], 1), w[0]))
        words = _join_hyphenated_word_tokens(words)

        for (_x0, _y0, _x1, _y1, raw, *_rest) in words:
            if not raw:
                continue

            token = _strip_edges_and_suffixes(raw)
            if not token:
                continue

            # Normalize for checks (ligatures, zero-width, case-fold)
            canon_ci = normalize_text(token, case_sensitive=False)
            canon_cs = normalize_text(token, case_sensitive=True)

            if not canon_ci:
                continue
            if _is_linkish(canon_cs):
                continue
            if canon_ci[0].isdigit():
                continue
            if len(canon_ci) < min_len:
                continue
            if canon_ci in common:
                continue

            # Choose counting key (keep all-caps acronyms if desired)
            key = canon_cs if (case_sensitive or (canon_cs.isupper() and 2 <= len(canon_cs) <= 6)) else canon_ci

            # EARLY-STOP: if this would introduce a new unique term beyond the cap, stop NOW
            if limit_terms is not None and key not in freq and len(freq) >= limit_terms:
                return [t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]

            # Count it
            freq[key] += 1

    return [t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]



# --- main CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Search keywords in a PDF, highlight them, and add bookmarks."
    )
    parser.add_argument("-i", "--input", required=True, help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output PDF file (default: input_k.pdf)")
    parser.add_argument("-k", "--keywords", nargs="+", help="List of keywords to search (quoted if multi-word)")
    parser.add_argument("-f", "--keywords-file", help="Text file with keywords (one per line)")
    parser.add_argument("-c", "--case-sensitive", action="store_true", help="Case sensitive search")
    parser.add_argument(
        "--no-whole-word",
        action="store_true",
        help="For multi-word phrases, allow suffix on the last word (e.g., 'molecular clouds')."
    )
    parser.add_argument(
        "--allow-concat",
        dest="allow_concat",
        action="store_true",
        help="Allow concatenation/hyphenation inside one token (e.g., 'photodissociation', 'photo-dissociation').",
    )
    # NEW: discover technical terms using a common-words file
    parser.add_argument(
        "--discover",
        metavar="COMMON_WORDS_FILE",
        help="Discover non-common terms directly from the PDF using COMMON_WORDS_FILE (one word per line).",
    )
    parser.add_argument(
        "--discover-limit", type=int, metavar="N",
        help="Stop discovery after finding N unique terms."
    )
    parser.add_argument(
        "--discover-max-pages", type=int, metavar="P",
        help="Only scan the first P pages during discovery."
    )
    parser.add_argument(
        "--discover-start", type=int, metavar="P",
        help="Start discovery at page P (1-based)."
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_k.pdf")

    doc = fitz.open(in_path)

    # collect keywords from CLI and/or file
    keywords = []
    if args.keywords:
        keywords.extend(args.keywords)
    if args.keywords_file:
        with open(args.keywords_file, encoding="utf-8") as f:
            for line in f:
                kw = line.strip()
                if kw and not kw.startswith("#"):
                    keywords.append(kw)

    # ---- DISCOVER TERMS (if requested) ----
    discovered = []
    if args.discover:
        discovered = discover_terms(
            doc,
            common_words_file=args.discover,
            min_len=3,
            case_sensitive=args.case_sensitive,
            limit_terms=args.discover_limit,
            max_pages=args.discover_max_pages,
            start_page=args.discover_start,  # <-- NEW
        )
        if discovered:
            # Merge discovered + provided keywords, preserving order
            merged, seen = [], set()
            for term in discovered + keywords:
                if term not in seen:
                    merged.append(term);
                    seen.add(term)
            keywords = merged
        else:
            print(f"[info] No terms discovered using {args.discover}")
        print(f"[info] Discovery used {len(discovered)} terms"
              + (f", early-stop at {args.discover_limit}" if args.discover_limit else "")
              + (f", pages≤{args.discover_max_pages}" if args.discover_max_pages else ""))

    if not keywords:
        parser.error("No keywords to search. Use -k/-f or pass --discover COMMON_WORDS_FILE.")

    matches = search_and_highlight(
        doc, keywords,
        case_sensitive=args.case_sensitive,
        no_whole_word=getattr(args, "no_whole_word", False),
        allow_concatenated=args.allow_concat,
    )
    print(f"Processed {in_path} -> {out_path}")
    if args.discover:
        print(f"Discovered {len(discovered)} terms.")
    for kw, pages in matches.items():
        print(f"'{kw}' found on pages {sorted(pages.keys())}")

    add_toc(doc, matches)
    try:
        doc.save(out_path, garbage=4, deflate=True)
    except Exception as e:
        print(f"Exception saving output: {e}")
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_k2.pdf")
        doc.save(out_path, garbage=4, deflate=True)
    doc.close()


if __name__ == "__main__":
    main()


