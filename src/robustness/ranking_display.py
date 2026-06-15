"""
Compact terminal ranking dashboard — pure Python, no extra dependencies.
Usage:  python ranking_display.py
        python ranking_display.py --no-color     (plain text, e.g. for logging)
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable

import pandas as pd

# ── ANSI helpers ────────────────────────────────────────────────────────────

USE_COLOR = "--no-color" not in sys.argv


def _c(*codes: int) -> str:
    return f"\033[{';'.join(map(str, codes))}m" if USE_COLOR else ""


RESET = _c(0)
BOLD = _c(1)
DIM = _c(2)


# named colours (foreground)
def fg(r, g, b):
    return _c(38, 2, r, g, b)


def bg(r, g, b):
    return _c(48, 2, r, g, b)


# Palette
C_HEADER = fg(74, 85, 104)
C_DIM = fg(45, 55, 72)
C_TEXT = fg(180, 190, 210)
C_MONO = fg(140, 160, 190)

C_GREEN = fg(34, 211, 160)
C_YELLOW = fg(250, 204, 21)
C_RED = fg(240, 82, 82)
C_BLUE = fg(59, 130, 246)
C_PURPLE = fg(99, 102, 241)
C_GREY = fg(100, 116, 139)

BG_ROW_A = bg(13, 17, 23)
BG_ROW_B = bg(10, 14, 22)
BG_BAR = bg(26, 31, 46)

# ── Bar drawing ─────────────────────────────────────────────────────────────

BLOCK_FULL = "█"
BLOCK_HALF = "▌"
BLOCK_EMPTY = "░"


def mini_bar(
    value: float, lo: float, hi: float, width: int, color: str = C_BLUE, show_val: bool = False, decimals: int = 2
) -> str:
    """Render a compact filled bar with optional right-aligned value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        empty = C_DIM + (BLOCK_EMPTY * width) + RESET
        return empty + (f" {'—':>{decimals + 3}}" if show_val else "")
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.5
    filled = int(frac * width * 2)  # half-block resolution
    full_blocks = filled // 2
    half_block = filled % 2
    empty_blocks = width - full_blocks - half_block
    bar = (
        BG_BAR
        + color
        + BLOCK_FULL * full_blocks
        + (BLOCK_HALF if half_block else "")
        + C_DIM
        + BLOCK_EMPTY * empty_blocks
        + RESET
    )
    if show_val:
        val_str = f"{value:>{decimals + 4}.{decimals}f}"
        return bar + C_MONO + val_str + RESET
    return bar


def score_color(v: float) -> str:
    if v >= 0.65:
        return C_GREEN
    if v >= 0.52:
        return C_YELLOW
    return C_GREY


def oos_color(v: float) -> str:
    if v > 0.1:
        return C_GREEN
    if v < -0.3:
        return C_RED
    return C_GREY


def dd_color(v: float) -> str:
    if v > 25:
        return C_RED
    if v > 15:
        return C_YELLOW
    return C_GREEN


# ── Formatting helpers ───────────────────────────────────────────────────────


def fmt(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{d}f}"


def time_window(sh, sm, eh, em) -> str:
    """Format like  1:00–11:00  or  1:30–9:30"""
    return f"{sh}:{sm:02d}–{eh}:{em:02d}"


# ── Column layout ────────────────────────────────────────────────────────────


# Each col: (header, width, align)  — align 'l' | 'r' | 'c'
def _is_header(metric_policy: str) -> str:
    if metric_policy == "blend":
        return "CPCV bl▸med/p25"
    if metric_policy == "median":
        return "CPCV med▸p25"
    if metric_policy == "p25":
        return "CPCV p25▸med"
    return "CPCV mean▸med"


def _build_cols(
    metric_policy: str,
    include_train_sh: bool = False,
    include_rrr: bool = True,
    include_portion: bool = True,
    include_nmb_c: bool = True,
):
    cols = [
        ("#", 3, "r"),
        ("Window", 20, "l"),
        (_is_header(metric_policy), 20, "l"),
    ]
    if include_rrr:
        cols.insert(2, ("RRR", 4, "r"))
    if include_portion:
        cols.insert(3 if include_rrr else 2, ("Por", 4, "r"))
    if include_nmb_c:
        idx = 2 + int(include_rrr) + int(include_portion)
        cols.insert(idx, ("Cnd", 4, "r"))
    if include_train_sh:
        cols.append(("Train Sh", 8, "r"))
    cols.extend(
        [
            ("Stab", 10, "l"),
            ("Boot", 10, "l"),
            ("OOS Sh", 7, "r"),
            ("MaxDD%", 13, "l"),
            ("Score", 13, "l"),
        ]
    )
    return cols


def _pad(text_bare: str, display_width: int, total_width: int, align: str) -> str:
    """Pad a string that may contain ANSI codes (bare = stripped for width calc)."""
    pad = total_width - display_width
    if align == "r":
        return " " * pad + text_bare
    if align == "c":
        lp = pad // 2
        return " " * lp + text_bare + " " * (pad - lp)
    return text_bare + " " * pad


def _strip(s: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", s)


def _truncate_plain(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 1:
        return "…"
    return text[: max_len - 1] + "…"


def _is_missing(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return v is None


def _has_any_nonempty_value(series: pd.Series) -> bool:
    for value in series:
        if _is_missing(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return True
    return False


def _fmt_pair_value(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.2f}"
    try:
        as_float = float(v)
        if math.isnan(as_float):
            return "—"
        if as_float.is_integer():
            return str(int(as_float))
        return f"{as_float:.2f}"
    except (TypeError, ValueError):
        return str(v)


def _resolve_identity(
    d: dict[str, object],
    use_time_window: bool,
    fallback_keys: Iterable[str],
    width: int,
) -> str:
    if use_time_window and all(not _is_missing(d.get(k)) for k in ("start_hour", "start_min", "end_hour", "end_min")):
        return _truncate_plain(
            time_window(d["start_hour"], d["start_min"], d["end_hour"], d["end_min"]),
            width,
        )

    pairs: list[str] = []
    for key in fallback_keys:
        value = d.get(key)
        if _is_missing(value):
            continue
        pairs.append(f"{key}={_fmt_pair_value(value)}")
        if len(pairs) >= 2:
            break
    if not pairs:
        return "—"
    return _truncate_plain(" | ".join(pairs), width)


def _format_is_cell(d: dict, metric_policy: str) -> str:
    med = float(d.get("cpcv_med", float("nan")))
    p25 = float(d.get("cpcv_p25", float("nan")))
    blend = d.get("cpcv_blend", d.get("cpcv_sharpe_blend", float("nan")))
    blend = float(blend) if blend is not None else float("nan")
    mean = float(d.get("mean_sharpe", float("nan")))

    # Fallback-compatible primary metric value by policy.
    if metric_policy == "blend":
        primary = blend
        if math.isnan(primary):
            if not math.isnan(med) and not math.isnan(p25):
                primary = 0.7 * med + 0.3 * p25
            elif not math.isnan(med):
                primary = med
            elif not math.isnan(p25):
                primary = p25
            else:
                primary = mean
        primary_s = C_TEXT + f"{primary:4.2f}" + RESET
        bar_s = mini_bar(primary, 0.5, 1.5, 6, C_BLUE)
        pair_s = C_DIM + f"{med:4.2f}/{p25:4.2f}" + RESET
        return primary_s + " " + bar_s + " " + pair_s

    if metric_policy == "median":
        primary_s = C_TEXT + f"{med:4.2f}" + RESET
        bar_s = mini_bar(med, 0.5, 1.5, 6, C_BLUE)
        second_s = C_DIM + f"{p25:4.2f}" + RESET
        return primary_s + " " + bar_s + " " + second_s

    if metric_policy == "mean":
        primary_s = C_TEXT + f"{mean:4.2f}" + RESET
        bar_s = mini_bar(mean, 0.5, 1.5, 6, C_BLUE)
        second_s = C_DIM + f"{med:4.2f}" + RESET
        return primary_s + " " + bar_s + " " + second_s

    # p25 default
    p25_s = C_DIM + f"{p25:4.2f}" + RESET
    bar_s = mini_bar(med, 0.5, 1.5, 6, C_BLUE)
    med_s = C_TEXT + f"{med:4.2f}" + RESET
    return p25_s + " " + bar_s + " " + med_s


def _format_train_sharpe(v: object) -> str:
    if _is_missing(v):
        return C_DIM + "  —" + RESET
    try:
        val = float(v)
    except (TypeError, ValueError):
        return C_DIM + "  —" + RESET
    if math.isnan(val):
        return C_DIM + "  —" + RESET
    return C_MONO + f"{val:+.2f}" + RESET


def render_row(
    d: dict,
    idx: int,
    use_time_window: bool,
    fallback_keys: Iterable[str],
    cols,
    metric_policy: str,
    include_train_sh: bool = False,
    include_rrr: bool = True,
    include_portion: bool = True,
    include_nmb_c: bool = True,
) -> str:
    sc = score_color(d["composite"])
    bg = BG_ROW_A if idx % 2 == 0 else BG_ROW_B

    # Left-border accent
    accent = sc + "▌" + RESET + bg

    cells = []

    # # rank
    cells.append(C_DIM + fmt(d["rank"], 0) + RESET)

    # Window
    window_width = next(w for h, w, _ in cols if h == "Window")
    w = _resolve_identity(d, use_time_window=use_time_window, fallback_keys=fallback_keys, width=window_width)
    cells.append(C_TEXT + w + RESET)

    if include_rrr:
        cells.append(C_MONO + fmt(d.get("rrr"), 0) + RESET)

    if include_portion:
        cells.append(C_MONO + fmt(d.get("portion"), 0) + RESET)

    if include_nmb_c:
        cells.append(C_MONO + fmt(d.get("nmb_c"), 0) + RESET)

    # IS performance block is policy-aware.
    cells.append(_format_is_cell(d, metric_policy=metric_policy))

    # Single full-train Sharpe diagnostic (optional).
    if include_train_sh:
        cells.append(_format_train_sharpe(d.get("full_train_sharpe")))

    # Stability bar
    cells.append(mini_bar(d["stab"], 0.4, 0.9, 8, C_PURPLE))

    # Bootstrap stability bar
    cells.append(mini_bar(d["boot_stab"], 0.4, 0.9, 8, C_BLUE))

    # OOS Sharpe
    oc = oos_color(d["oos_sharpe"])
    cells.append(oc + f"{d['oos_sharpe']:+.2f}" + RESET)

    # MaxDD bar
    cells.append(mini_bar(d["oos_maxdd"], 0, 40, 8, dd_color(d["oos_maxdd"]), show_val=True, decimals=1))

    # Composite score bar
    cells.append(mini_bar(d["composite"], 0.3, 0.8, 8, sc, show_val=True, decimals=2))

    # Build row string with padding
    parts = []
    for cell, (_, col_w, align) in zip(cells, cols, strict=True):
        bare_len = len(_strip(cell))
        parts.append(_pad(cell, bare_len, col_w, align))

    row_body = "  ".join(parts)
    return accent + bg + row_body + RESET


def render_header(cols) -> str:
    parts = []
    for hdr, col_w, align in cols:
        parts.append(_pad(C_HEADER + BOLD + hdr + RESET, len(hdr), col_w, align))
    return C_DIM + "  " + RESET + "  ".join(parts)


def render_separator(total_width: int, char="─") -> str:
    return C_DIM + "  " + char * (total_width - 2) + RESET


def render_title(n: int, include_train_sh: bool = False) -> str:
    title = "  STRATEGY RANKING"
    sub = f"  {n} candidates · CPCV=fold in-sample metric · Stab=proximity · Boot=bootstrap · OOS=holdout"
    if include_train_sh:
        sub += " · Train Sh=single run on original train split"
    return "\n" + BOLD + C_TEXT + title + RESET + "\n" + C_DIM + sub + RESET + "\n"


def render_legend(include_train_sh: bool = False) -> str:
    parts = [
        f"{C_GREEN}█{RESET} score ≥0.65",
        f"{C_YELLOW}█{RESET} score ≥0.52",
        f"{C_GREY}█{RESET} score <0.52",
        f"  OOS: {C_GREEN}+{RESET}=positive  {C_RED}−{RESET}=negative",
        f"  DD:  {C_GREEN}low{RESET}  {C_YELLOW}>15%{RESET}  {C_RED}>25%{RESET}",
    ]
    if include_train_sh:
        parts.append("  Train Sh: full-train single backtest Sharpe")
    return C_DIM + "  " + RESET + "  ".join(parts)


# ── Main display function ────────────────────────────────────────────────────


def _normalize_metric_policy(metric_policy: object) -> str:
    val = str(metric_policy).strip().lower()
    if val in {"blend", "median", "p25", "mean"}:
        return val
    return "p25"


def display_ranking(df: pd.DataFrame, metric_policy: str = "p25") -> None:
    """
    Render a compact terminal dashboard for ranking results.

    Expected columns (subset used):
      rank, start_hour, start_min, end_hour, end_min,
      rrr, portion, nmb_c,
      cpcv_p25 (or cpcv_sharpe_p25), cpcv_med (or cpcv_sharpe_median),
      stab (or stability_score), boot_stab (or bootstrap_stability_score),
      oos_sharpe, oos_maxdd, composite (or composite_score),
      optional: full_train_sharpe, full_train_basis
    """
    required_direct = ("rank", "oos_sharpe", "oos_maxdd")
    required_alias_groups = {
        "cpcv_p25": ("cpcv_p25", "cpcv_sharpe_p25"),
        "cpcv_med": ("cpcv_med", "cpcv_sharpe_median"),
        "stab": ("stab", "stability_score"),
        "boot_stab": ("boot_stab", "bootstrap_stability_score"),
        "composite": ("composite", "composite_score"),
    }
    cols = set(df.columns)
    missing_direct = [c for c in required_direct if c not in cols]
    missing_groups = [name for name, opts in required_alias_groups.items() if not any(opt in cols for opt in opts)]
    if missing_direct or missing_groups:
        missing_parts = []
        if missing_direct:
            missing_parts.append("columns: " + ", ".join(missing_direct))
        if missing_groups:
            group_expr = ", ".join(f"{name} ({' or '.join(required_alias_groups[name])})" for name in missing_groups)
            missing_parts.append("aliases: " + group_expr)
        raise ValueError("display_ranking missing required schema: " + "; ".join(missing_parts))

    # Normalise column names (accept both short and full names)
    col_map = {
        "cpcv_sharpe_p25": "cpcv_p25",
        "cpcv_sharpe_median": "cpcv_med",
        "cpcv_sharpe_blend": "cpcv_blend",
        "stability_score": "stab",
        "bootstrap_stability_score": "boot_stab",
        "composite_score": "composite",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    policy = _normalize_metric_policy(metric_policy)
    if "is_perf_metric" in df.columns:
        valid = df["is_perf_metric"].dropna().astype(str)
        if not valid.empty:
            policy = _normalize_metric_policy(valid.iloc[0])
    include_train_sh = "full_train_sharpe" in df.columns
    include_rrr = "rrr" in df.columns and _has_any_nonempty_value(df["rrr"])
    include_portion = "portion" in df.columns and _has_any_nonempty_value(df["portion"])
    include_nmb_c = "nmb_c" in df.columns and _has_any_nonempty_value(df["nmb_c"])
    cols = _build_cols(
        policy,
        include_train_sh=include_train_sh,
        include_rrr=include_rrr,
        include_portion=include_portion,
        include_nmb_c=include_nmb_c,
    )
    total_width = sum(w for _, w, _ in cols) + len(cols) * 2 + 1
    use_time_window = all(c in df.columns for c in ("start_hour", "start_min", "end_hour", "end_min"))

    excluded_exact = {
        "rank",
        "start_hour",
        "start_min",
        "end_hour",
        "end_min",
        "cpcv_p25",
        "cpcv_med",
        "stab",
        "boot_stab",
        "oos_sharpe",
        "oos_maxdd",
        "composite",
        "tier",
        "recommendation",
        "full_train_sharpe",
        "full_train_basis",
    }
    excluded_prefixes = (
        "mean_",
        "std_",
        "cv_",
        "fold_",
        "norm_",
        "pass_",
        "degradation_",
        "bootstrap_",
        "stability_",
        "oos_",
        "is_",
        "cpcv_",
        "neigh_",
    )
    raw_fallback = [
        c for c in df.columns if c not in excluded_exact and not any(c.startswith(pfx) for pfx in excluded_prefixes)
    ]
    preferred = ["nmb_c", "rrr", "portion"]
    fallback_keys = [k for k in preferred if k in raw_fallback] + [k for k in raw_fallback if k not in preferred]

    rows = df.to_dict("records")
    print(render_title(len(rows), include_train_sh=include_train_sh))
    print(render_header(cols))
    print(render_separator(total_width))
    for i, row in enumerate(rows):
        print(
            render_row(
                row,
                i,
                use_time_window=use_time_window,
                fallback_keys=fallback_keys,
                cols=cols,
                metric_policy=policy,
                include_train_sh=include_train_sh,
                include_rrr=include_rrr,
                include_portion=include_portion,
                include_nmb_c=include_nmb_c,
            )
        )
    print(render_separator(total_width))
    print(render_legend(include_train_sh=include_train_sh))
    print()
