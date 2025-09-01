# app.py
# Pick-5 Winner Profiler (Profile Picker) — now with a "Copy/Paste Positional Profile" block
# So you can paste p1..p5 directly into the straight-maker app (JSON or shorthand), plus a JSON download.

from __future__ import annotations
import json
import os
import io
import numpy as np
import streamlit as st

st.set_page_config(page_title="Pick-5 Winner Profiler (Profile Picker)", layout="wide")
st.title("Pick-5 Winner Profiler (Profile-Based)")
st.caption("Choose a learned profile (State + Mid/Eve), paste your last 7 seeds (oldest first, most recent last).")

BEST_LOOKBACK = 7
LOW_MAX = 4  # low digits 0..4

STATES = ["OH", "DC", "FL", "GA", "PA", "LA", "VA", "DE"]
DRAWS  = ["mid", "eve"]  # lower-case

# ---------- File loader ----------
def load_positional_matrices_for(state: str, draw: str):
    """
    Load the learned positional matrices for the selected State + Draw.
    - Tries 'positional_matrices_<STATE>_<draw>.json'
    - For DC+mid, also tries legacy 'positional_matrices.json' for backward compatibility
    Returns (mats, path) where mats is the dict {'P1':..., 'P5':...}.
    """
    fname = f"positional_matrices_{state}_{draw}.json"
    search = [fname]
    if state == "DC" and draw == "mid":
        search.insert(0, "positional_matrices.json")

    for path in search:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                mats = json.load(f)
            # sanity check
            for key in ("P1", "P2", "P3", "P4", "P5"):
                if key not in mats:
                    raise ValueError(f"{path} is missing key '{key}'.")
                for row in mats[key]:
                    s = sum(row)
                    if not (99.0 <= s <= 101.0):  # allow tiny float drift
                        raise ValueError(f"{path}: a row in {key} sums to {s:.3f}, expected ~100.")
            return mats, path

    tried = ", ".join(search)
    raise FileNotFoundError(
        "No profile file found for "
        f"{state} {draw.upper()}.\n"
        f"Tried: {tried}.\n"
        "Create it with your build_profile.py and save using the filename pattern "
        "'positional_matrices_<STATE>_<mid|eve>.json' (e.g., positional_matrices_FL_eve.json)."
    )

# ---------- Helpers ----------
def parse_seeds(text: str) -> list[list[int]]:
    seeds = []
    for ln in text.strip().splitlines():
        s = ln.strip()
        if s.isdigit() and len(s) == 5:
            seeds.append([int(c) for c in s])
        elif s:
            raise ValueError(f"Invalid seed line '{s}'. Use exactly 5 digits, no dashes.")
    return seeds

def avg_positional_preds(seeds: list[list[int]], mats_pct: dict[str, list[list[float]]]) -> dict[int, np.ndarray]:
    """
    Average per-position distributions across the last BEST_LOOKBACK seeds
    using the learned conditional rows (percentages -> probabilities).
    """
    preds: dict[int, np.ndarray] = {}
    for pos in range(1, 6):
        mat = np.array(mats_pct[f"P{pos}"], dtype=float) / 100.0  # 10x10
        acc = np.zeros(10, dtype=float)
        for s in seeds:
            acc += mat[s[pos-1]]
        acc /= max(1, len(seeds))
        preds[pos] = acc
    return preds

def digit_inclusion_probs(preds: dict[int, np.ndarray]) -> np.ndarray:
    not_in = np.ones(10, dtype=float)
    for pos in range(1, 6):
        not_in *= (1.0 - preds[pos])  # prob digit d NOT used at this position
    return 1.0 - not_in

def poisson_binomial_probs(p_list: list[float]) -> np.ndarray:
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
    pmf = np.array([1.0])
    for pos in range(1, 6):
        v = preds[pos]
        pos_vec = np.zeros(46, dtype=float)
        for d in range(10):
            pos_vec[d] = v[d]
        pmf = np.convolve(pmf, pos_vec)[:46]
    return pmf

def topN_combos(preds: dict[int, np.ndarray], K_per_pos: int = 5, N: int = 20) -> list[tuple[str, float]]:
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

def preds_to_json(preds: dict[int, np.ndarray]) -> str:
    """Build JSON with p1..p5, digits 0..9 as percentage values (0–100)."""
    out = {}
    for pos in range(1, 6):
        row = {str(d): float(preds[pos][d] * 100.0) for d in range(10)}
        out[f"p{pos}"] = row
    return json.dumps(out, indent=2)

def preds_to_shorthand(preds: dict[int, np.ndarray]) -> str:
    """
    Build shorthand like:
    p1: 4:28.57, 7:28.57, 0:28.57, 2:14.29, ...
    Ordered by descending prob; includes all digits 0..9.
    """
    lines = []
    for pos in range(1, 6):
        order = np.argsort(preds[pos])[::-1]
        parts = [f"{d}:{preds[pos][d]*100:.2f}" for d in order]
        lines.append(f"p{pos}: " + ", ".join(parts))
    return "; ".join(lines)

# ---------- Sidebar: profile picker ----------
with st.sidebar:
    st.subheader("Profile")
    state = st.selectbox("State", STATES, index=STATES.index("DC"))
    draw  = st.selectbox("Draw", DRAWS, index=0)  # default mid
    st.caption("Profiles are JSON files named: positional_matrices_<STATE>_<mid|eve>.json")
    uploaded = st.file_uploader("Or load a profile JSON (override this session)", type=["json"])

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
    "One per line, 5 digits each",
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
seeds = seeds_all[-BEST_LOOKBACK:]  # use last up to 7

# ---------- Compute predictions ----------
preds = avg_positional_preds(seeds, mats_pct)
incl = digit_inclusion_probs(preds)

p_odd = [float(np.sum(preds[pos][1::2])) for pos in range(1,6)]
p_low = [float(np.sum(preds[pos][:LOW_MAX+1])) for pos in range(1,6)]
odd_dist = poisson_binomial_probs(p_odd)
low_dist = poisson_binomial_probs(p_low)

SUM_BANDS = [(0,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45)]
sum_pmf = sum_distribution_from_positions(preds)
sum_band_pct = [(f"{lo}-{hi}", float(np.sum(sum_pmf[lo:hi+1]))) for (lo,hi) in SUM_BANDS]

top20 = topN_combos(preds, K_per_pos=5, N=20)

# ---------- Display: analytics ----------
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
st.markdown("### Top 20 full 5-digit candidates (by product of per-position probabilities)")
for combo, p in top20:
    st.write(f"**{combo}** — weight {pct(p)}")

# ---------- NEW: Copy/Paste Positional Profile (for Straight Maker) ----------
st.markdown("---")
st.markdown("## Copy/Paste Positional Profile (for Straight Maker)")
st.caption("Use either format below in your straight-maker app’s 'Positional stats' field.")

# JSON format (digits 0..9 as percents, per p1..p5)
json_blob = preds_to_json(preds)
st.markdown("**JSON format**")
st.code(json_blob, language="json")

# Shorthand format (sorted by descending prob; includes all digits)
short_blob = preds_to_shorthand(preds)
st.markdown("**Shorthand format**")
st.code(short_blob)

# Download JSON
buf = io.StringIO()
buf.write(json_blob)
st.download_button(
    label="Download p1..p5 profile (.json)",
    data=buf.getvalue(),
    file_name=f"positional_profile_{state}_{draw}.json",
    mime="application/json"
)

st.markdown("---")
st.markdown("**Notes**")
st.write("- This app uses the picked **State/Draw** profile file and does not re-learn weekly.")
st.write("- To refresh a profile, rebuild its JSON with `build_profile.py` and save it as "
         "`positional_matrices_<STATE>_<mid|eve>.json` in this folder "
         "(e.g., `positional_matrices_FL_eve.json`).")
st.write("- DC Mid also works with your legacy `positional_matrices.json` filename.")
