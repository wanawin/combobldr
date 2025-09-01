#!/usr/bin/env python3
"""
Builds positional_matrices_<STATE>_<DRAW>.json for the profile app.

- Reads a text file of draws (each line can contain 5 digits with or without separators).
  Examples that all work on one line:
    "Tue Aug 26 ... 1-7-4-8-8"
    "17488"
    "1 7 4 8 8"
- Detects chronological direction (newest-first or oldest-first) and fixes to oldest→newest.
- Computes 10x10 positional transition matrices:
    P(next Pi = y | seed Pi = x), as PERCENTAGES (rows sum ≈ 100).
- Saves JSON as: positional_matrices_<STATE>_<DRAW>.json   (e.g., positional_matrices_DC_mid.json)
- Optional: limit to the most recent N draws with --recent N.

Usage:
  python build_profile.py --input dc5_history.txt --state DC --draw mid
"""

import argparse
import json
import re
from typing import List, Optional, Tuple

DRAW_RE = re.compile(r'(\d)[^\d]*?(\d)[^\d]*?(\d)[^\d]*?(\d)[^\d]*?(\d)')

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create learned profile JSON for the Pick-5 profiler app.")
    ap.add_argument("--input", "-i", required=True, help="Path to text file with past draws.")
    ap.add_argument("--state", "-s", required=True, choices=["OH","DC","FL","GA","PA","LA","VA","DE"],
                    help="State code (OH, DC, FL, GA, PA, LA, VA, DE).")
    ap.add_argument("--draw", "-d", required=True, choices=["mid","eve"],
                    help="Draw: mid or eve (lowercase).")
    ap.add_argument("--recent", type=int, default=0,
                    help="If >0, only use the most recent N draws after order is corrected.")
    return ap.parse_args()

def extract_draw_from_line(line: str) -> Optional[List[int]]:
    """
    Returns [d1..d5] if the line contains 5 digits in order (with any separators), else None.
    """
    m = DRAW_RE.search(line)
    if not m:
        return None
    return [int(g) for g in m.groups()]

def load_draws(path: str) -> List[List[int]]:
    out: List[List[int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            digs = extract_draw_from_line(raw)
            if digs is not None:
                out.append(digs)
    if len(out) < 2:
        raise ValueError("Need at least 2 parsed draws in the file.")
    return out

def coverage(draws: List[List[int]]) -> int:
    """
    Rough score: how many seed-digit bins (per position) are non-empty.
    Higher means the direction is more likely oldest→newest for our purposes.
    """
    seen = 0
    for pos in range(5):
        bins = [0]*10
        for d in draws[:-1]:
            bins[d[pos]] += 1
        seen += sum(1 for b in bins if b > 0)
    return seen

def ensure_oldest_to_newest(draws: List[List[int]]) -> Tuple[List[List[int]], str]:
    """
    Decide whether to reverse by comparing coverage in both directions.
    """
    cov_a = coverage(draws)
    cov_b = coverage(list(reversed(draws)))
    if cov_b > cov_a:
        return list(reversed(draws)), "reversed input (newest→oldest detected)"
    return draws, "as provided (assumed oldest→newest)"

def build_transition_matrices(draws: List[List[int]]) -> dict:
    """
    Build counts then convert to percentage matrices per position (10x10 each).
    Returns dict {'P1': [[...],[...],...], ..., 'P5': [[...],[...],...]} in percentages.
    """
    counts = {pos: [[0]*10 for _ in range(10)] for pos in range(5)}
    for i in range(len(draws)-1):
        seed = draws[i]
        nxt  = draws[i+1]
        for pos in range(5):
            x = seed[pos]; y = nxt[pos]
            counts[pos][x][y] += 1

    mats_pct = {}
    for pos in range(5):
        mat = counts[pos]
        pct = []
        for row in mat:
            s = sum(row)
            if s == 0:
                pct.append([0.0]*10)
            else:
                pct.append([round(100.0 * c / s, 6) for c in row])
        mats_pct[f"P{pos+1}"] = pct
    return mats_pct

def write_json(path: str, mats: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mats, f, indent=2)

def main():
    args = parse_args()
    draws = load_draws(args.input)
    draws, note = ensure_oldest_to_newest(draws)

    if args.recent and args.recent > 1:
        draws = draws[-args.recent:]
        recent_note = f"Using most recent {args.recent} draws."
    else:
        recent_note = "Using all parsed draws."

    mats = build_transition_matrices(draws)
    out_name = f"positional_matrices_{args.state}_{args.draw}.json"
    write_json(out_name, mats)

    # Console summary
    print(f"[ok] Parsed {len(draws)} draws ({note}). {recent_note}")
    print(f"[ok] Wrote {out_name}")
    for pos in range(1,6):
        sums = [round(sum(r), 3) for r in mats[f'P{pos}']]
        print(f"  P{pos} row sums ~100? -> {sums}")

    print("\nDone. Use this profile in your app by selecting "
          f"{args.state} {args.draw.upper()} in the sidebar.")

if __name__ == "__main__":
    main()
