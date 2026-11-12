"""Feature rows for the MGL P_self ranker (frontier lane, fetch-free).

The layout must match the offline trainer exactly (fable-crawl-selection
`scripts/features.py`, groups lex + tok + link): 22 URL-shape columns, 11 link
columns, then NHASH hashed path/query token columns. Nothing here reads golden
data, url_score or domain_score, so the model generalises to domains that were
never in a golden batch.
"""
from __future__ import annotations

import math
import re
import zlib
from urllib.parse import urlsplit

FEATURE_VERSION = 1
NHASH = 1 << 14
TOK_RE = re.compile(r"[a-z]+|\d+")
DATE_RE = re.compile(r"/(19|20)\d{2}[/-](0?[1-9]|1[0-2])([/-]|$)|(19|20)\d{2}(0[1-9]|1[0-2])[0-3]\d")
LONGNUM_RE = re.compile(r"\d{5,}")
EXT_RE = re.compile(r"\.([a-z0-9]{1,5})$")
EXTS = {"html": 1, "htm": 1, "php": 2, "aspx": 2, "asp": 2, "jsp": 2, "pdf": 3, "jpg": 4, "png": 4, "gif": 4,
        "xml": 5, "json": 5, "js": 6, "css": 6, "mp4": 7, "zip": 8}

LEX = ["https", "host_labels", "is_www", "is_mobile_sub", "lang_sub", "path_depth", "path_len", "url_len",
       "n_digits", "has_query", "n_params", "ext_cat", "trail_slash", "is_home", "has_date", "has_longnum",
       "last_seg_len", "last_seg_hyphens", "pct_enc", "upper", "tld_len", "is_gov_edu"]
LINK = ["log_inlink", "log_inlink_ext", "has_anchor", "anchor_len", "anchor_words", "dst", "has_parent",
        "parent_depth", "parent_is_home", "parent_same_host", "parent_query"]
NUM_COLS = LEX + LINK
N_FEATURES = len(NUM_COLS) + NHASH


def _split(url: str):
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _lex(url: str, p) -> list[float]:
    host = (p.hostname or "").lower()
    labels = host.split(".")
    path = p.path or ""
    segs = [s for s in path.split("/") if s]
    last = segs[-1] if segs else ""
    m = EXT_RE.search(last.lower())
    return [p.scheme == "https", len(labels), labels[0] == "www", labels[0] in ("m", "mobile", "amp"),
            len(labels[0]) == 2 and len(labels) > 2, len(segs), len(path), len(url),
            sum(ch.isdigit() for ch in path), bool(p.query), p.query.count("&") + 1 if p.query else 0,
            EXTS.get(m.group(1), 9) if m else 0, path.endswith("/"), len(segs) == 0 and not p.query,
            bool(DATE_RE.search(path)), bool(LONGNUM_RE.search(path)), len(last), last.count("-") + last.count("_"),
            url.count("%"), sum(ch.isupper() for ch in path), len(labels[-1]),
            any(x in ("gov", "edu", "ac", "org") for x in labels[-2:])]


def _link(p, inlink, inlink_ext, anchor, dst, parent) -> list[float]:
    anchor = anchor or ""
    pp = _split(parent) if parent else None
    pseg = [s for s in (pp.path or "").split("/") if s] if pp else []
    return [math.log1p(inlink or 0), math.log1p(inlink_ext or 0), bool(anchor.strip()), len(anchor),
            len(anchor.split()), dst if dst is not None else -1, pp is not None,
            len(pseg) if pp else -1, (pp is not None and not pseg), pp is not None and pp.hostname == p.hostname,
            bool(pp.query) if pp else False]


def _tokens(p) -> set[int]:
    text = (p.path or "").lower() + " " + (p.query or "").lower()
    toks = {zlib.crc32(("p:" + t).encode()) % NHASH for t in TOK_RE.findall(text) if not t.isdigit() and len(t) > 1}
    segs = [s for s in (p.path or "").split("/") if s]
    if segs:
        toks.add(zlib.crc32(("s1:" + segs[0].lower()[:30]).encode()) % NHASH)
    return toks


def feature_row(row: dict) -> tuple[list[float], set[int]] | None:
    """(numeric columns, hashed token columns offset from len(NUM_COLS)), or None for an unparsable URL.

    `row` carries url, inlink_count_approx, inlink_count_external, anchor_text,
    discovery_source_type, discovered_from (missing keys count as unknown).
    """
    url = row["url"]
    p = _split(url)
    if p is None:
        return None
    num = _lex(url, p) + _link(p, row.get("inlink_count_approx"), row.get("inlink_count_external"),
                               row.get("anchor_text"), row.get("discovery_source_type"),
                               row.get("discovered_from"))
    return [float(x) for x in num], _tokens(p)
