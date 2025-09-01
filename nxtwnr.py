# app.py
# DC-5 Next Winner Profiler — Uses learned positional profiles
# Requires: positional_matrices.json (learned from your 2-year history)
# UI: Paste the last 7 seeds (most-recent last). Displays probabilities & top 20 combos.

from __future__ import annotations
import json
import numpy as np
import streamlit as st

# -------------------- Config / Learned constants --------------------
BEST_LOOKBACK = 7            # from your history: best window size
LOW_MAX = 4                  # low digits are 0..4; high are 5..9
SUM_BANDS = [(0,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45)]

st.set_page_config(page_title="DC-5 Winner Profiler (Profile-Based)", layout="wide")
st.title("DC-5 Winner Profiler (Profile-Based)")
st.caption(
    "Uses learned positional profiles from your 2-year history. "
    "Paste the last 7 seeds (oldest first, most recent last). "
    "It does **not** recalculate history — it applies your learned profiles to your seeds."
)

# -------------------- Load learned positional matrices --------------------
# Expected format: dict {'P1': [[row per seed digit 0..9 -> 10 probabilities (%)...], ...], ..., 'P5': ...}
# Each row sums to 100 (percent).
@st.cache_data(show_spinner=False)
def load_positional_matrices(path: str = "positional_matrices.json") -> dict[str, list[list[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        mats = json.load(f)
    # basic sanity: row sums
    for key in ("P1","P2","P3","P4","P5"):
        if key not in mats:
            raise ValueError(f"Missing key {key} in positional_matrices.json")
        for row in mats[key]:
            s = sum(row)
            if not (99.9 <= s <= 100.1):
                raise ValueError(f"Row in {key} does not sum to 100: {row} (sum={s})")
    return mats

try:
    POS_MATS_PCT = load_positional_matrices()
except Exception as e:
    st.error(f"Could not load 'positional_matrices.json': {e}")
    st.stop()

# -------------------- Helpers --------------------
def parse_seeds(text: str) -> list[list[int]]:
    seeds = []
    for ln in text.strip().splitlines():
        s = ln.strip()
        if len(s) == 0:
            continue
        if not (s.isdigit() and len(s) == 5):
            raise ValueError(f"Invalid line '{s}'. Each seed must be exactly 5 digits (e.g., 42150).")
        seeds.append([int(c) for c in s])
    return seeds

def avg_positional_preds(seeds: list[list[int]]) -> dict[int, np.ndarray]:
    """
    For each position 1..5, average the learned row P(next Pi | seed Pi) across the last BEST_LOOKBACK seeds.
    Returns probabilities (0..1) arrays of length 10 for each position.
    """
    preds: dict[int, np.ndarray] = {}
    for pos in range(1, 6):
        mat = np.array(POS_MATS_PCT[f"P{pos}"], dtype=float) / 100.0  # 10x10
        acc = np.zeros(10, dtype=float)
        for s in seeds:
            acc += mat[s[pos-1]]  # row for the seed digit at this position
        acc /= max(1, len(seeds))
        preds[pos] = acc
    return preds

def digit_inclusion_probs(preds: dict[int, np.ndarray]) -> np.ndarray:
    """
    P(d appears anywhere) = 1 - product over positions of (1 - P(Pi=d))
    """
    not_in = np.ones(10, dtype=float)
    for pos in range(1, 6):
        not_in *= (1.0 - preds[pos])  # elementwise: prob digit d not used at this position
    return 1.0 - not_in

def poisson_binomial_probs(p_list: list[float]) -> np.ndarray:
    """
    Poisson-binomial distribution for number of successes across independent but non-identical Bernoulli trials.
    Dynamic programming. Returns array of length (n+1).
    """
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

def sum_distribution_from_positions(preds: dict[int, np.ndarray]) -> np.ndarray:
    """
    Exact convolution of 5 independent position digit PMFs (0..9) -> sum 0..45.
    """
    pmf = np.array([1.0])  # start with degenerate 0
    for pos in range(1, 6):
        v = preds[pos]
        # make a length-46 vector for this position with mass at indices 0..9
        pos_vec = np.zeros(46, dtype=float)
        for d in range(10):
            pos_vec[d] = v[d]
        pmf = np.convolve(pmf, pos_vec)[:46]
    return pmf  # indexed by sum 0..45

def topN_combos(preds: dict[int, np.ndarray], K_per_pos: int = 5, N: int = 20) -> list[tuple[str, float]]:
    """
    Build top-N 5-digit combos by product of per-position probabilities.
    """
    idxs = [list(np.argsort(preds[pos])[::-1][:K_per_pos]) for pos in range(1, 6)]
    cand: list[tuple[str, float]] = []
    for d1 in idxs[0]:
        for d2 in idxs[1]:
            for d3 in idxs[2]:
                for d4 in idxs[3]:
                    for d5 in idxs[4]:
                        p = preds[1][d1]*preds[2][d2]*preds[3][d3]*preds[4][d4]*preds[5][d5]
                        cand.append(("".join(map(str, [d1,d2,d3,d4,d5])), p))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[:N]

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# -------------------- UI --------------------
st.subheader("Enter the last 7 seeds (most recent last)")
seed_text = st.text_area("One per line, 5 digits each", height=180, placeholder="e.g.\n42150\n36788\n72556\n76244\n10936\n62511\n42003")

go = st.button("Analyze")

if not go:
    st.stop()

try:
    seeds_all = parse_seeds(seed_text)
except ValueError as e:
    st.error(str(e))
    st.stop()

if len(seeds_all) < BEST_LOOKBACK:
    st.warning(f"You provided {len(seeds_all)} seed(s). For best accuracy, paste **{BEST_LOOKBACK}** seeds (most recent last). Using what you gave.")
seeds = seeds_all[-BEST_LOOKBACK:]  # use last up to 7

# -------------------- Core Calculations --------------------
preds = avg_positional_preds(seeds)                   # per-position pmfs (0..9)
incl = digit_inclusion_probs(preds)                   # P(digit appears anywhere)
p_odd_pos = [float(np.sum(preds[pos][1::2])) for pos in range(1,6)]
p_low_pos = [float(np.sum(preds[pos][:LOW_MAX+1])) for pos in range(1,6)]

odd_dist = poisson_binomial_probs(p_odd_pos)          # k=0..5 odds
low_dist = poisson_binomial_probs(p_low_pos)          # k=0..5 lows

sum_pmf = sum_distribution_from_positions(preds)      # sum 0..45
sum_band_pct = []
for lo, hi in SUM_BANDS:
    s = float(np.sum(sum_pmf[lo:hi+1]))
    sum_band_pct.append((f"{lo}-{hi}", s))

top20 = topN_combos(preds, K_per_pos=5, N=20)

# -------------------- Display --------------------
c1, c2 = st.columns([2,1])
with c1:
    st.markdown("### Place-Value Digit Probabilities (next winner)")
    tabs = st.tabs([f"P{pos}" for pos in range(1,6)])
    for pos, tab in zip(range(1,6), tabs):
        with tab:
            order = np.argsort(preds[pos])[::-1]
            st.write("Top digits by probability:")
            st.write(", ".join([f"{d}: {pct(preds[pos][d])}" for d in order[:10]]))

with c2:
    st.markdown("### Digit Inclusion (any position)")
    order = np.argsort(incl)[::-1]
    st.write(", ".join([f"{d}: {pct(incl[d])}" for d in order]))

st.markdown("---")
c3, c4, c5 = st.columns(3)
with c3:
    st.markdown("### Parity (odd-count) in next winner")
    st.write(", ".join([f"{k} odds: {pct(odd_dist[k])}" for k in range(6)]))
with c4:
    st.markdown(f"### Low/High (Low ≤ {LOW_MAX})")
    st.write(", ".join([f"{k} lows: {pct(low_dist[k])}" for k in range(6)]))
with c5:
    st.markdown("### Sum ranges")
    st.write(", ".join([f"{band}: {pct(p)}" for band, p in sum_band_pct]))

st.markdown("---")
st.markdown("### Top 20 full 5-digit candidates (by product of per-position probabilities)")
for combo, p in top20:
    st.write(f"**{combo}** — weight {pct(p)}")

# -------------------- Explanations (contextual) --------------------
st.markdown("### Why these recommendations?")
# Recent absence ("due") across your last 7 seeds:
recent_digits = [d for s in seeds for d in s]
absent = [d for d in range(10) if d not in recent_digits]
present = [d for d in range(10) if d in recent_digits]

top_pos_notes = []
for pos in range(1,6):
    o = np.argsort(preds[pos])[::-1][:3]
    top_pos_notes.append(f"P{pos} is leaning to {o[0]} ({pct(preds[pos][o[0]])}), "
                         f"then {o[1]} ({pct(preds[pos][o[1]])}), {o[2]} ({pct(preds[pos][o[2]])}).")

st.write("• Using **7 seeds** because that window had the best positional accuracy in your history.")
st.write("• **Place-value evidence:** " + " ".join(top_pos_notes))
st.write("• **Digit inclusion odds** reflect how the 5 positions combine; a digit high in multiple positions becomes a strong overall include.")
if absent:
    # show inclusion chances for absent digits
    msg = ", ".join([f"{d}:{pct(incl[d])}" for d in absent])
    st.write(f"• **Due context:** digits **not seen** in your last 7 seeds → {msg}.")
else:
    st.write("• **Due context:** every digit appears at least once in your last 7 seeds.")

st.info("Tip: Use the place-value probabilities first, then check parity/low-high/sum bands to shape final selections.")
