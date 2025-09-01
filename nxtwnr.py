# app.py
# Pick-5 Winner Profiler — Profile Picker + "Must NOT include" + Best/2nd-Best-Per-Box tie rule
#
# Workflow:
# 1) Pick State + Draw (loads positional_matrices_<STATE>_<mid|eve>.json; DC mid also tries positional_matrices.json)
# 2) Paste last 7 seeds (oldest->newest). App averages conditional rows to get per-position probabilities.
# 3) Build candidates from the top-K digits per position. Group by BOX (sorted digits).
# 4) For each BOX, evaluate ALL permutations with the position probabilities:
#       - keep the single best straight, OR
#       - keep two straights if exactly one position has a 2-way tie among the best perms.
# 5) Apply "Must NOT include these digits" to the final straight list.
# 6) Show analytics + final list (ranked by score).
#
# Requires: streamlit, numpy

from __future__ import annotations
import json
import os
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Pick-5 Profiler (Tie Rule + Exclusions)", layout="wide")
st.title("Pick-5 Winner Profiler")
st.caption("Profile-picker + 'must-not-include' filter + best/second-best per box tie rule.")

BEST_LOOKBACK = 7
LOW_MAX = 4  # low digits 0..4

STATES = ["OH", "DC", "FL", "GA", "PA", "LA", "VA", "DE"]
DRAWS  = ["mid", "eve"]  # lowercase

EPS = 1e-15

# ---------- File loader (fixed & robust) ----------
def load_positional_matrices_for(state: str, draw: str):
    """
    Load learned positional matrices for State + Draw.
    Tries: 'positional_matrices_<STATE>_<draw>.json'
    Also tries legacy 'positional_matrices.json' for DC mid.
    Returns (mats_dict, path_str).
    """
    fname = f"positional_matrices_{state}_{draw}.json"
    search = [fname]
    if state == "DC" and draw == "mid":
        search.insert(0, "positional_matrices.json")

    for path in search:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                mats = json.load(f)
            for key in ("P1", "P2", "P3", "P4", "P5"):
                if key not in mats:
                    raise ValueError(f"{path} is missing key '{key}'.")
                for row in mats[key]:
                    s = sum(row)
                    if not (99.0 <= s <= 101.0):  # percentages with tiny drift
                        raise ValueError(f"{path}: a row in {key} sums to {s:.3f}, expected ~100.")
            return mats, path

    tried = ", ".join(search)
    raise FileNotFoundError(
        "No profile file found for "
        f"{state} {draw.upper()}.\n"
        f"Tried: {tried}.\n"
        "Create it with your builder and save as "
        "'positional_matrices_<STATE>_<mid|eve>.json' (e.g., positional_matrices_FL_eve.json)."
    )

# ---------- Helpers ----------
def parse_seeds(text: str) -> List[List[int]]:
    seeds = []
    for ln in text.strip().splitlines():
        s = ln.strip()
        if s.isdigit() and len(s) == 5:
            seeds.append([int(c) for c in s])
        elif s:
            raise ValueError(f"Invalid seed line '{s}'. Use exactly 5 digits, no dashes.")
    return seeds

def avg_positional_preds(seeds: List[List[int]], mats_pct: Dict[str, List[List[float]]]) -> Dict[int, np.ndarray]:
    """
    Average conditional rows for last N seeds to get per-position probabilities.
    mats_pct are percentages; convert to probabilities first.
    """
    preds: Dict[int, np.ndarray] = {}
    for pos in range(1, 6):
        mat = np.array(mats_pct[f"P{pos}"], dtype=float) / 100.0  # 10x10
        acc = np.zeros(10, dtype=float)
        for s in seeds:
            acc += mat[s[pos-1]]
        acc /= max(1, len(seeds))
        preds[pos] = acc
    return preds

def digit_inclusion_probs(preds: Dict[int, np.ndarray]) -> np.ndarray:
    not_in = np.ones(10, dtype=float)
    for pos in range(1, 6):
        not_in *= (1.0 - preds[pos])  # prob digit d NOT used at this position
    return 1.0 - not_in

def poisson_binomial_probs(p_list: List[float]) -> np.ndarray:
    n = len(p_list)
    dp = np.zeros(n+1, dtype=float)
    dp[0] = 1.0
    for p in p_list:
        ndp = np.zeros(n+1, dtype=float)
        for k in range(n+1):
            ndp[k] += dp[k] * (1.0 - p)
            if k+1 <= n:
                ndp[k+1] += dp[k] * p
        dp = ndp
    return dp

def sum_distribution_from_positions(preds: Dict[int, np.ndarray]) -> np.ndarray:
    pmf = np.array([1.0])
    for pos in range(1, 6):
        v = preds[pos]
        pos_vec = np.zeros(46, dtype=float)
        for d in range(10):
            pos_vec[d] = v[d]
        pmf = np.convolve(pmf, pos_vec)[:46]
    return pmf

def straight_score(perm: Tuple[int, int, int, int, int], preds: Dict[int, np.ndarray]) -> float:
    score = 1.0
    for i, d in enumerate(perm, start=1):
        score *= preds[i][d]
    return score

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# ---------- Sidebar: profile + options ----------
with st.sidebar:
    st.subheader("Profile")
    state = st.selectbox("State", STATES, index=STATES.index("DC"))
    draw  = st.selectbox("Draw", DRAWS, index=0)  # default mid
    st.caption("Loads positional_matrices_<STATE>_<mid|eve>.json (DC Mid also tries positional_matrices.json).")
    uploaded = st.file_uploader("Or upload a profile JSON (override for this session)", type=["json"])

    st.markdown("---")
    st.subheader("Generation breadth")
    K_per_pos = st.slider("Top-K digits per position (to form candidate boxes)", 3, 7, 5, 1)
    max_results = st.slider("Max results to show", 10, 200, 50, 10)

    st.markdown("---")
    st.subheader("Final filter")
    forbid_str = st.text_input(
        "Must NOT include these digits",
        help="Comma/space-separated digits. Any final straight containing any of them will be dropped."
    )
    forbid_digits = {int(x) for x in forbid_str.replace(",", " ").split() if x.isdigit()}

# Load profile
mats_pct = None
src_info = ""
if uploaded is not None:
    try:
        mats_pct = json.load(uploaded)
        for key in ("P1","P2","P3","P4","P5"):
            if key not in mats_pct:
                st.error("Uploaded JSON missing key: " + key)
                st.stop()
        src_info = "(from upload)"
    except Exception as e:
        st.error(f"Could not parse uploaded JSON: {e}")
        st.stop()
else:
    try:
        mats_pct, used_path = load_positional_matrices_for(state, draw)
        src_info = f"(loaded: {used_path})"
    except Exception as e:
        st.error(str(e))
        st.stop()

st.success(f"Profile selected: **{state} {draw.upper()}** {src_info}")

# ---------- Seeds input ----------
st.subheader("Enter the last 7 seeds (oldest first, most recent last)")
seed_text = st.text_area(
    "One per line, 5 digits each (e.g., 42150)",
    height=150,
    placeholder="e.g.\n42150\n36788\n72556\n76244\n10936\n62511\n42003"
)
go = st.button("Analyze")

if not go:
    st.stop()

# Parse seeds
try:
    seeds_all = parse_seeds(seed_text)
except ValueError as e:
    st.error(str(e))
    st.stop()

if len(seeds_all) < BEST_LOOKBACK:
    st.warning(f"You provided {len(seeds_all)} seed(s). For best results, paste **{BEST_LOOKBACK}**.")
seeds = seeds_all[-BEST_LOOKBACK:]  # last up to 7

# ---------- Compute predictions ----------
preds = avg_positional_preds(seeds, mats_pct)
incl = digit_inclusion_probs(preds)

# Analytics: parity & low/high & sum
p_odd = [float(np.sum(preds[pos][1::2])) for pos in range(1,6)]
p_low = [float(np.sum(preds[pos][:LOW_MAX+1])) for pos in range(1,6)]
odd_dist = poisson_binomial_probs(p_odd)
low_dist = poisson_binomial_probs(p_low)
SUM_BANDS = [(0,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45)]
sum_pmf = sum_distribution_from_positions(preds)
sum_band_pct = [(f"{lo}-{hi}", float(np.sum(sum_pmf[lo:hi+1]))) for (lo,hi) in SUM_BANDS]

# ---------- Build candidate boxes from top-K per position, then apply tie rule ----------
# 1) For each position, take indices of top-K digits by probability
top_idx = [list(np.argsort(preds[pos])[::-1][:K_per_pos]) for pos in range(1, 6)]

# 2) Generate candidate straights by K-grid and collect UNIQUE boxes
#    We'll evaluate each box exactly once (all permutations) with the tie rule.
boxes_seen = set()
kept_per_box: List[Tuple[str, float]] = []

# quick product loop without importing itertools.product explicitly
for d1 in top_idx[0]:
    for d2 in top_idx[1]:
        for d3 in top_idx[2]:
            for d4 in top_idx[3]:
                for d5 in top_idx[4]:
                    box = tuple(sorted([d1,d2,d3,d4,d5]))
                    if box in boxes_seen:
                        continue
                    boxes_seen.add(box)

                    # Evaluate all unique permutations of this box
                    best_score = -1.0
                    best_perms = []
                    for perm in set(permutations(box)):
                        sc = straight_score(perm, preds)
                        if sc > best_score + EPS:
                            best_score = sc
                            best_perms = [perm]
                        elif abs(sc - best_score) <= EPS:
                            best_perms.append(perm)

                    # Choose 1 or 2 straights per tie rule
                    if best_score <= 0.0 + EPS:
                        # zero-score: keep a canonical minimal straight
                        s_min = "".join(map(str, min(best_perms)))
                        kept_per_box.append((s_min, best_score))
                        continue

                    # Determine positional tie structure among best perms
                    pos_vals = [set() for _ in range(5)]
                    for perm in best_perms:
                        for i, d in enumerate(perm):
                            pos_vals[i].add(d)
                    multi_positions = [i for i, s in enumerate(pos_vals) if len(s) > 1]

                    if len(multi_positions) == 1 and len(pos_vals[multi_positions[0]]) == 2:
                        # Keep exactly two variants (any order stable)
                        idx = multi_positions[0]
                        variants = {}
                        for perm in best_perms:
                            key = perm[idx]
                            if key not in variants:
                                variants[key] = perm
                            if len(variants) == 2:
                                break
                        for perm in variants.values():
                            kept_per_box.append(("".join(map(str, perm)), best_score))
                    else:
                        # Keep one canonical best
                        s_min = "".join(map(str, min(best_perms)))
                        kept_per_box.append((s_min, best_score))

# 3) Sort overall by score desc, then lexicographically; trim to max_results first
kept_per_box.sort(key=lambda x: (-x[1], x[0]))
if len(kept_per_box) > max_results:
    kept_per_box = kept_per_box[:max_results]

# 4) Apply "must-not-include digits" on the final straight list
if forbid_digits:
    filtered = [(s, sc) for (s, sc) in kept_per_box if all(int(ch) not in forbid_digits for ch in s)]
else:
    filtered = kept_per_box

# ---------- Display ----------
c1, c2 = st.columns([2,1])
with c1:
    st.markdown("### Place-Value Digit Probabilities (next winner)")
    tabs = st.tabs([f"P{pos}" for pos in range(1,6)])
    for pos, tab in zip(range(1,6), tabs):
        with tab:
            order = np.argsort(preds[pos])[::-1]
            st.write(", ".join([f"{d}: {pct(preds[pos][d])}" for d in order[:10]]))
with c2:
    st.markdown("### Digit Inclusion (any position)")
    order = np.argsort(incl)[::-1]
    st.write(", ".join([f"{d}: {pct(incl[d])}" for d in order]))

st.markdown("---")
c3, c4, c5 = st.columns(3)
with c3:
    st.markdown("### Parity (odd-count)")
    st.write(", ".join([f"{k} odds: {pct(odd_dist[k])}" for k in range(6)]))
with c4:
    st.markdown(f"### Low/High (Low ≤ {LOW_MAX})")
    st.write(", ".join([f"{k} lows: {pct(low_dist[k])}" for k in range(6)]))
with c5:
    st.markdown("### Sum ranges")
    st.write(", ".join([f"{band}: {pct(p)}" for band, p in sum_band_pct]))

st.markdown("---")
st.markdown("### Final Straights (after tie rule & 'must-not-include' filter)")
if filtered:
    for combo, p in filtered:
        st.write(f"**{combo}** — weight {pct(p)}")
else:
    st.warning("No candidates remained. Try increasing Top-K, clearing the 'must-not-include' digits, or check seeds/profile.")

st.caption(
    "Notes: Built from top-K per position, then grouped by BOX and evaluated across all permutations. "
    "Per box: kept one best straight, or two if exactly one position had a 2-way tie. "
    "Lastly applied the digit exclusion."
)
