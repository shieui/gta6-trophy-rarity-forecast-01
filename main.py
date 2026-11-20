"""gta6-trophy-rarity-forecast-01: forecast Exophase earn rates at the target.

Model, in counts, not rates.  A read gives players tracked N and a rate r_i
for each trophy; earners are E_i = r_i * N / 100.  Earners only grow (players
earn trophies, nobody un-earns one); N only grows (more profiles sync).
A rate moves because E_i and N grow at different speeds, and both move in
jumps: Exophase refreshes a game's counts on its own schedule, hours apart,
so between refreshes the page is frozen and at a refresh N jumps and every
rate is diluted by the new players.

So the forecast is: find the refresh states in the history (runs of
identical reads), fit how log N and log E_i grow against log time since
release (launch growth decelerates, so log-time, not log-linear), and read
both fits at the target.  The rate is E_i / N at the target.

When the whole history is one frozen state (launch night: no refresh has
been seen yet), a refresh is overdue and the median assumes one refresh
before the target.  The one refresh the house recorded tells us what that
looks like: N went 638 -> 3,868 while total trophies earned went 943 ->
4,276.  Almost all of the new earners are the single 100% trophy (every new
player holds it); the other 54 trophies gained only ~2 earners each, so a
fresh cohort earns a new trophy at roughly 6% of the old cohort's rate.
The median therefore dilutes every non-100% rate by about (1 + 5*0.06)/6
and leaves the 100% trophy at 100%.

Spread is a logit standard deviation combining a random walk
(0.5*sqrt(days)), refresh-jump uncertainty (grows with the number of
refreshes expected before the target), and the per-trophy fit residual.
Quantiles are the median shifted by the normal quantiles times that spread.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Random-walk spread, logit units per sqrt(day), as the house baseline uses.
SIGMA_RW = 0.5
# Spread added per expected refresh, in quadrature (logit units).
SIGMA_REFRESH = 0.25
# Spread for one unseen refresh when the history is a single frozen state.
SIGMA_FIRST_REFRESH = 0.6
# Residual spread floor / cap for a trophy's own fit.
RESID_FLOOR = 0.2
RESID_CAP = 0.8
# Prior for a first, unseen refresh: players-tracked growth factor, and the
# fraction of the old rate at which the fresh cohort holds a trophy.
PRIOR_N_RATIO = 6.0
PRIOR_FRESH_FRAC = 0.06
# How often Exophase refreshes a game's counts, hours (house notes: the
# page sat unchanged ~4h, then refreshed).
REFRESH_HOURS = 4.0


def parse_time(text):
    """ISO timestamp -> epoch seconds (handles trailing Z and fractions)."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        e = math.exp(-x)
        return 100.0 / (1.0 + e)
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JS-style rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def linfit(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def refresh_states(case):
    """Collapse the history into one state per refresh.

    A state is (epoch, players, [rate or None per trophy]).  Consecutive
    reads that agree on players and every rate are one state; a state starts
    at its first read.  Null rates are filled from the previous read.
    """
    trophies = case["trophies"]
    n = len(trophies)
    states = []
    last_rates = [None] * n
    for read in case.get("history") or []:
        t = parse_time(read.get("at"))
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if t is None or players is None:
            continue
        filled = []
        for i in range(n):
            r = rates[i] if i < len(rates) else None
            if r is None:
                r = last_rates[i]
            filled.append(r)
        last_rates = filled
        if states:
            pt, pn, pr = states[-1]
            if pn == players and pr == filled:
                continue
        states.append((t, players, filled))
    return states


def expected_refreshes(states, horizon_hours):
    """Refreshes expected before the target, from the gaps between states."""
    if len(states) < 2:
        return horizon_hours / REFRESH_HOURS
    gaps = [(states[k + 1][0] - states[k][0]) / 3600.0
            for k in range(len(states) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return horizon_hours / REFRESH_HOURS
    gaps.sort()
    interval = gaps[len(gaps) // 2]
    if interval <= 0:
        interval = REFRESH_HOURS
    return horizon_hours / interval


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0

    release = case.get("game") or {}
    t0 = parse_time(release.get("release_date"))
    if t0 is None:
        t0 = 0.0
    # release_date is a plain date: treat it as midnight UTC.
    if release.get("release_date") and len(release["release_date"]) == 10:
        t0 = datetime.fromisoformat(release["release_date"] + "T00:00:00+00:00").timestamp()

    states = refresh_states(case)
    if not states:
        # No history at all: shrug with a wide spread around 1%.
        return {"trophies": [{"id": tr["id"], "q": quantiles(1.0, 2.0)}
                             for tr in trophies]}

    last_t, last_n, last_rates = states[-1]
    target_t = last_t + horizon_min * 60.0
    # Log-time regressor: hours since release, +1 so launch reads sit near 0.
    x_of = lambda t: math.log1p(max(0.0, (t - t0) / 3600.0))
    x_target = x_of(target_t)

    medians = [None] * n
    spreads = [None] * n

    if len(states) == 1:
        # One frozen page: a refresh is overdue; assume one before target.
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                medians[i] = 1.0
            elif r >= 99.99:
                # Held by every player tracked; a refresh adds more players
                # who hold it too.  It stays at 100%.
                medians[i] = 100.0
                spreads[i] = 0.05
                continue
            else:
                medians[i] = r * (1.0 + (PRIOR_N_RATIO - 1.0) * PRIOR_FRESH_FRAC) / PRIOR_N_RATIO
            spreads[i] = math.sqrt((SIGMA_RW * math.sqrt(horizon_d)) ** 2
                                   + SIGMA_FIRST_REFRESH ** 2
                                   + RESID_FLOOR ** 2)
    else:
        # --- players tracked at the target: log-time fit over refresh states.
        xs = [x_of(t) for t, _, _ in states]
        ys = [math.log(max(1, pn)) for _, pn, _ in states]
        fit_n = linfit(xs, ys)
        if fit_n is not None and fit_n[1] > 0:
            n_target = math.exp(fit_n[0] + fit_n[1] * x_target)
        else:
            n_target = float(last_n)
        n_target = min(max(n_target, float(last_n)), float(last_n) * 30.0)
        n_ratio = n_target / max(1.0, float(last_n))

        # --- earners per trophy: shrunk log-time fit.
        for i, tr in enumerate(trophies):
            pts = []
            for t, pn, rates in states:
                r = rates[i]
                if r is not None and r > 0:
                    pts.append((x_of(t), math.log(max(0.5, r * pn / 100.0))))
            e_last = None
            r_last = last_rates[i]
            if r_last is not None:
                e_last = max(0.0, r_last * last_n / 100.0)
            if r_last is not None and r_last >= 99.99:
                medians[i] = 100.0
                spreads[i] = 0.05
                continue
            if e_last is None:
                medians[i] = 1.0
                spreads[i] = 2.0
                continue
            resid = RESID_FLOOR
            e_fit = None
            if len(pts) >= 3 and len({round(p[1], 6) for p in pts}) >= 3:
                fit = linfit([p[0] for p in pts], [p[1] for p in pts])
                if fit is not None and fit[1] > -0.05:
                    e_fit = math.exp(fit[0] + fit[1] * x_target)
                    resid = min(RESID_CAP, max(RESID_FLOOR, fit[2]))
            # Elasticity fallback: earners grow with the player base, but
            # slower than it (new players hold fewer trophies).
            e_elastic = e_last * (n_ratio ** 0.5)
            if e_fit is None:
                e_target = e_elastic
            else:
                # Geometric mean: shrink the noisy fit toward the fallback.
                e_target = math.sqrt(max(e_fit, 0.5) * max(e_elastic, 0.5))
            e_target = min(max(e_target, e_last), n_target)
            medians[i] = 100.0 * e_target / n_target
            spreads[i] = resid

        m = expected_refreshes(states, horizon_h)
        spread_extra = math.sqrt((SIGMA_RW * math.sqrt(horizon_d)) ** 2
                                 + (SIGMA_REFRESH * math.sqrt(min(m, 20.0))) ** 2)
        for i in range(n):
            if spreads[i] is None:
                spreads[i] = RESID_FLOOR
            # A trophy held by every player stays at 100%: no extra spread.
            if medians[i] == 100.0 and last_rates[i] is not None and last_rates[i] >= 99.99:
                continue
            spreads[i] = math.sqrt(spreads[i] ** 2 + spread_extra ** 2)

    if horizon_min <= 0:
        spreads = [0.05] * n

    rows = []
    for i, tr in enumerate(trophies):
        med = medians[i] if medians[i] is not None else 1.0
        s = spreads[i] if spreads[i] is not None else 1.0
        rows.append({"id": tr["id"], "q": quantiles(med, s)})
    return {"trophies": rows}


def quantiles(median_pct, spread):
    lg = logit(median_pct)
    out = []
    for z in Z:
        q = inv_logit(lg + z * spread)
        q = min(99.9999, max(0.0001, q))
        out.append(round4(q))
    # Keep non-decreasing after rounding.
    for k in range(1, 5):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
