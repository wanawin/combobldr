# app.py
# Pick-5 Winner Profiler — Profile Picker + Must-NOT-Include + Best/Second-Best-Per-Box tie rule
# Now with minimum low/high/even/odd (default 2 each, adjustable).

from __future__ import annotations
import json
import os
from itertools import permutations
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st

st.set_page_config(page_title="Pick-5 Profiler (Tie Rule + Filters)", layout="wide")
st.title("Pick-5 Winner Profiler")
st.caption("Profile-picker + min low/high/even/odd filters + must-not-include + tie rule.")

BEST_LOOKBACK = 7
LOW_MAX = 4  # low digits 0..4
EPS = 1e-15

STATES = ["OH", "DC", "FL", "GA", "PA", "LA", "VA", "DE"]
DRAWS  = ["mid", "eve"]

# ---------- Loader ----------
def load_positional_matrices_for(state: str, draw: str):
    fname = f"positional_matrices_{state}_{draw}.json"
    search = [fname]
    if state == "DC" and draw == "mid":
        search.insert(0, "positional_matrices.json")
    for path in search:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                mats = json.load(f)
            return mats, path
    raise FileNotFoundError(f"No profile found for {state} {draw.upper()}.")

# ---------- Helpers ----------
def parse_seeds(text: str) -> List[List[int]]:
    seeds = []
    for ln in text.strip().splitlines():
        s = ln.strip()
        if s.isdigit() and len(s) == 5:
            seeds.append([int(c) for c in s])
    return seeds

def avg_positional_preds(seeds: List[List[int]], mats_pct: Dict[str, List[List[float]]]) -> Dict[int, np.ndarray]:
    preds: Dict[int, np.ndarray] = {}
    for pos in range(1, 6):
        mat = np.array(mats_pct[f"P{pos}"], dtype=float) / 100.0
        acc = np.zeros(10, dtype=float)
        for s in seeds:
            acc += mat[s[pos-1]]
        acc /= max(1, len(seeds))
        preds[pos] = acc
    return preds

def straight_score(perm: Tuple[int,...], preds: Dict[int, np.ndarray]) -> float:
    score = 1.0
    for i, d in enumerate(perm, start=1):
        score *= preds[i][d]
    return score

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def count_low_high_even_odd(s: str, low_max: int = LOW_MAX):
    digs = [int(c) for c in s]
    lows = sum(1 for d in digs if d <= low_max)
    highs = 5 - lows
    evens = sum(1 for d in digs if d % 2 == 0)
    odds = 5 - evens
    return lows, highs, evens, odds

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Profile")
    state = st.selectbox("State", STATES, index=STATES.index("DC"))
    draw  = st.selectbox("Draw", DRAWS, index=0)
    uploaded = st.file_uploader("Or upload profile JSON", type=["json"])

    st.subheader("Generation breadth")
    K_per_pos = st.slider("Top-K digits per position", 3, 7, 5, 1)
    max_results = st.slider("Max results to show", 10, 200, 50, 10)

    st.subheader("Minimum counts (default 2)")
    min_low  = st.number_input("Minimum lows (≤4)", 0, 5, 2, 1)
    min_high = st.number_input("Minimum highs (≥5)", 0, 5, 2, 1)
    min_even = st.number_input("Minimum evens", 0, 5, 2, 1)
    min_odd  = st.number_input("Minimum odds", 0, 5, 2, 1)

    st.subheader("Final filter")
    forbid_str = st.text_input("Must NOT include these digits")
    forbid_digits = {int(x) for x in forbid_str.replace(",", " ").split() if x.isdigit()}

# ---------- Load profile ----------
if uploaded:
    mats_pct = json.load(uploaded)
    src_info = "(from upload)"
else:
    mats_pct, used_path = load_positional_matrices_for(state, draw)
    src_info = f"(loaded {used_path})"
st.success(f"Profile selected: {state} {draw.upper()} {src_info}")

# ---------- Seeds ----------
seed_text = st.text_area("Enter last 7 seeds (oldest first, one per line, 5 digits each)")
go = st.button("Analyze")
if not go:
    st.stop()

seeds_all = parse_seeds(seed_text)
if len(seeds_all) < BEST_LOOKBACK:
    st.warning(f"You provided {len(seeds_all)} seed(s). Using last {len(seeds_all)}.")
seeds = seeds_all[-BEST_LOOKBACK:]

# ---------- Predictions ----------
preds = avg_positional_preds(seeds, mats_pct)

# ---------- Candidate generation ----------
top_idx = [list(np.argsort(preds[pos])[::-1][:K_per_pos]) for pos in range(1, 6)]
boxes_seen = set()
kept: List[Tuple[str, float]] = []

for d1 in top_idx[0]:
    for d2 in top_idx[1]:
        for d3 in top_idx[2]:
            for d4 in top_idx[3]:
                for d5 in top_idx[4]:
                    box = tuple(sorted([d1,d2,d3,d4,d5]))
                    if box in boxes_seen: continue
                    boxes_seen.add(box)

                    best_score, best_perms = -1.0, []
                    for perm in set(permutations(box)):
                        sc = straight_score(perm, preds)
                        if sc > best_score + EPS:
                            best_score, best_perms = sc, [perm]
                        elif abs(sc - best_score) <= EPS:
                            best_perms.append(perm)

                    if best_score <= 0.0 + EPS:
                        s_min = "".join(map(str, min(best_perms)))
                        kept.append((s_min, best_score))
                        continue

                    pos_vals = [set() for _ in range(5)]
                    for perm in best_perms:
                        for i,d in enumerate(perm):
                            pos_vals[i].add(d)
                    multi_positions = [i for i,s in enumerate(pos_vals) if len(s)>1]

                    if len(multi_positions)==1 and len(pos_vals[multi_positions[0]])==2:
                        variants = {}
                        for perm in best_perms:
                            key = perm[multi_positions[0]]
                            if key not in variants:
                                variants[key] = perm
                            if len(variants)==2: break
                        for perm in variants.values():
                            s = "".join(map(str, perm))
                            kept.append((s, best_score))
                    else:
                        s_min = "".join(map(str, min(best_perms)))
                        kept.append((s_min, best_score))

# ---------- Filter results ----------
# Sort by score desc then lex
kept.sort(key=lambda x: (-x[1], x[0]))
if len(kept) > max_results:
    kept = kept[:max_results]

filtered = []
for s, sc in kept:
    lows, highs, evens, odds = count_low_high_even_odd(s)
    if lows < min_low or highs < min_high or evens < min_even or odds < min_odd:
        continue
    if forbid_digits and any(int(ch) in forbid_digits for ch in s):
        continue
    filtered.append((s, sc))

# ---------- Display ----------
st.markdown("### Final Straights")
if filtered:
    for combo, p in filtered:
        st.write(f"**{combo}** — weight {pct(p)}")
else:
    st.warning("No candidates survived the filters.")
