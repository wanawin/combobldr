# app.py
# DC-5 Box Generator + Best Straight Picker (with tie notes)
from __future__ import annotations
import json
from itertools import combinations_with_replacement, permutations
from collections import Counter
import streamlit as st

st.set_page_config(page_title="DC-5: Constrained Boxes -> Best Straights", layout="wide")
st.title("DC-5: Constrained Boxes → Best Straight(s)")

# -------------------- Helpers --------------------
LOW_MAX_DEFAULT = 4  # low digits are 0..4, high are 5..9 by default

def parse_mandatory_digits(s: str) -> list[int]:
    out = []
    for token in s.replace(",", " ").split():
        if token.isdigit():
            d = int(token)
            if 0 <= d <= 9:
                out.append(d)
    # dedupe while preserving order
    seen = set()
    res = []
    for d in out:
        if d not in seen:
            res.append(d)
            seen.add(d)
    return res

def longest_consecutive_run_length(digits_sorted_unique: list[int]) -> int:
    if not digits_sorted_unique:
        return 0
    run = 1
    best = 1
    for i in range(1, len(digits_sorted_unique)):
        if digits_sorted_unique[i] == digits_sorted_unique[i-1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best

def violates_patterns(counts: Counter, allow_quints, allow_quads, allow_triples, allow_double_doubles):
    vals = list(counts.values())
    if not allow_quints and any(v == 5 for v in vals):
        return True
    if not allow_quads and any(v == 4 for v in vals):
        return True
    if not allow_triples and any(v == 3 for v in vals):
        return True
    # double double = two distinct pairs (2,2,1 pattern or more pairs)
    pairs = sum(1 for v in vals if v == 2)
    if not allow_double_doubles and pairs >= 2:
        return True
    return False

def parse_positional_stats(text: str) -> dict[int, list[float]]:
    """
    Accepts JSON or a 'p1:/p2:/...' shorthand.
    Returns dict {1: [p(d=0)..p(d=9)], ..., 5: [...]}, probabilities in [0,1].
    - If values sum ≈100, treat as percentages; if ≈1, treat as probabilities.
    - Missing digits default to 0.
    Raises ValueError if malformed.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        out = {}
        for k in ("p1","p2","p3","p4","p5"):
            if k not in obj:
                raise ValueError(f"Missing key '{k}' in JSON.")
            row = [float(obj[k].get(str(d), 0.0)) for d in range(10)]
            s = sum(row)
            if 99.5 <= s <= 100.5:
                row = [x/100.0 for x in row]
            elif 0.99 <= s <= 1.01:
                pass
            else:
                if s > 0:
                    row = [x/s for x in row]
            out[int(k[1])] = row
        return out
    except json.JSONDecodeError:
        # Try "p1: 4:28.57,7:28.57,..." shorthand (semicolon-separated records)
        out = {}
        lines = [ln.strip() for ln in text.split(";") if ln.strip()]
        if not lines:
            raise ValueError("Could not parse positional stats.")
        for rec in lines:
            if ":" not in rec:
                continue
            head, tail = rec.split(":", 1)
            head = head.strip().lower()
            if not head.startswith("p") or len(head) != 2 or head[1] not in "12345":
                continue
            pos = int(head[1])
            row = [0.0]*10
            for chunk in tail.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if ":" not in chunk:
                    parts = chunk.split()
                    if len(parts) != 2:
                        continue
                    d_str, v_str = parts
                else:
                    d_str, v_str = [x.strip() for x in chunk.split(":", 1)]
                if d_str.isdigit():
                    d = int(d_str)
                    if 0 <= d <= 9:
                        try:
                            v = float(v_str)
                        except:
                            v = 0.0
                        row[d] = v
            s = sum(row)
            if 99.5 <= s <= 100.5:
                row = [x/100.0 for x in row]
            elif 0.99 <= s <= 1.01:
                pass
            else:
                if s > 0:
                    row = [x/s for x in row]
            out[pos] = row
        if len(out) != 5:
            raise ValueError("Provide p1..p5.")
        return out

def straight_score(straight: tuple[int, int, int, int, int], pos_probs: dict[int, list[float]]) -> float:
    score = 1.0
    for i, d in enumerate(straight, start=1):
        score *= pos_probs[i][d]
    return score

# -------------------- Sidebar: constraints --------------------
st.sidebar.header("Constraints")

sum_min, sum_max = st.sidebar.slider("Sum range", 0, 45, (0, 45))
low_max = st.sidebar.number_input("Low max digit (low ≤ this value)", 0, 9, LOW_MAX_DEFAULT, 1)

mand_str = st.sidebar.text_input(
    "Mandatory digits (OR logic: at least one must appear)",
    help="Comma/space-separated digits, e.g. 7, 0, 2"
)
mand_digits = parse_mandatory_digits(mand_str)

col1, col2 = st.sidebar.columns(2)
even_exact = col1.number_input("# Even (leave -1 to ignore)", -1, 5, -1, 1)
odd_exact  = col2.number_input("# Odd (leave -1 to ignore)",  -1, 5, -1, 1)

col3, col4 = st.sidebar.columns(2)
low_exact  = col3.number_input("# Low (0..low_max) (leave -1 to ignore)", -1, 5, -1, 1)
high_exact = col4.number_input("# High (leave -1 to ignore)", -1, 5, -1, 1)

st.sidebar.markdown("**Pattern allowances** (check to allow; uncheck to filter out):")
allow_quints = st.sidebar.checkbox("Allow quints (aaaaa)", value=False)
allow_quads  = st.sidebar.checkbox("Allow quads  (aaaab)", value=False)
allow_triples= st.sidebar.checkbox("Allow triples (aaabc)", value=True)
allow_dd     = st.sidebar.checkbox("Allow double doubles (aabbc)", value=True)
allow_runs4p = st.sidebar.checkbox("Allow runs ≥4 (e.g., 1-2-3-4)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Positional stats (optional, to pick best straight)**")
st.sidebar.caption("Paste JSON with keys p1..p5 mapping digits 0..9 to % or prob. "
                   "Example (JSON): {\"p1\":{\"4\":28.57,\"7\":28.57,\"0\":28.57,\"2\":14.29}, ...} "
                   "Or shorthand: p1: 4:28.57,7:28.57,0:28.57,2:14.29; p2: ...")
pos_stats_text = st.sidebar.text_area("Positional stats", height=160, value="")

go = st.sidebar.button("Generate")

# -------------------- Core generation --------------------
if go:
    # Parse positional stats (optional)
    pos_probs = {}
    if pos_stats_text.strip():
        try:
            pos_probs = parse_positional_stats(pos_stats_text)
        except Exception as e:
            st.warning(f"Couldn't parse positional stats — proceeding without scoring straights.\nDetails: {e}")

    # Enumerate all 5-digit "boxes" (orderless multisets of digits 0..9)
    total = 0
    kept = []

    for comb in combinations_with_replacement(range(10), 5):
        total += 1
        s = sum(comb)
        if not (sum_min <= s <= sum_max):
            continue

        counts = Counter(comb)
        # even/odd checks
        evens = sum(1 for d in comb if d % 2 == 0)
        odds  = 5 - evens
        if even_exact >= 0 and evens != even_exact:
            continue
        if odd_exact  >= 0 and odds  != odd_exact:
            continue

        # low/high checks
        lows  = sum(1 for d in comb if d <= low_max)
        highs = 5 - lows
        if low_exact  >= 0 and lows  != low_exact:
            continue
        if high_exact >= 0 and highs != high_exact:
            continue

        # --------- MANDATORY DIGITS = OR logic ---------
        # If any mandatory digits are provided, require at least one to be present.
        if mand_digits and not any(d in counts for d in mand_digits):
            continue

        # pattern constraints
        if violates_patterns(counts, allow_quints, allow_quads, allow_triples, allow_dd):
            continue

        # runs constraint
        if not allow_runs4p:
            uniq_sorted = sorted(set(comb))
            # no runs of length >= 4
            run = 1
            bad = False
            for i in range(1, len(uniq_sorted)):
                if uniq_sorted[i] == uniq_sorted[i-1] + 1:
                    run += 1
                    if run >= 4:
                        bad = True
                        break
                else:
                    run = 1
            if bad:
                continue

        kept.append(comb)

    st.success(f"Found {len(kept)} box combos (out of {total} total).")

    # If positional stats exist, find best straight(s) for each box
    if pos_probs:
        tie_note = []
        best_straights = []
        for box in kept:
            best_score = -1.0
            best_perms = set()
            # Avoid duplicate perms when there's a pair
            for perm in set(permutations(box)):
                score = 1.0
                for i, d in enumerate(perm, start=1):
                    score *= pos_probs.get(i, [0.0]*10)[d]
                if score > best_score + 1e-15:
                    best_score = score
                    best_perms = {perm}
                elif abs(score - best_score) <= 1e-15:
                    best_perms.add(perm)
            # record
            for perm in sorted(best_perms):
                best_straights.append(("".join(map(str, perm)), best_score, box))
            if len(best_perms) > 1:
                tie_note.append(("".join(map(str, box)), len(best_perms)))

        # Sort overall by score desc
        best_straights.sort(key=lambda x: x[1], reverse=True)

        st.markdown("### Best Straight(s) per Box (scored by positional stats)")
        st.caption("If multiple straights tie for a box, all are shown and noted below.")
        # copy/paste list
        lines = []
        for s, sc, b in best_straights:
            lines.append(s)
        st.code("\n".join(lines))

        if tie_note:
            st.info("Ties detected for these boxes (box → # of best straights):\n" +
                    ", ".join([f"{box}:{n}" for box, n in tie_note]))
    else:
        st.markdown("### Boxes (no positional stats provided)")
        st.caption("Paste positional stats in the sidebar to score and order straights.")
        # display boxes as strings
        st.code("\n".join("".join(map(str, b)) for b in kept))
else:
    st.info("Set your constraints and click **Generate**.")
