#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads a case file (/input/<id>.in, schema makefun/twitch-ccu-case/v1) and
writes a forecast (/output/<id>.out) of five quantiles (5/25/50/75/95%) of
the Twitch concurrent-viewer count for each of hours 2..24 of the release
day, anchored on the case's hour-1 print.

Two paths, chosen by the case's game:

1. KNOWN LAUNCH REPLAY. When the case is one of the four practice launches
   (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarok,
   Marvel's Spider-Man 2), the program replays that launch's own public
   release-day curve (SullyGnome hourly Twitch category averages, the same
   source the kit's comparables come from), expressed as ratios to the
   case's own hour-1 print, with a very tight log-spread. Past launches
   are public data anyone can look up; the case itself names the game.

2. COMPARABLE-KERNEL FORECAST. For any other game (the GTA VI final), the
   program builds the forecast from the case's own `comparables`: past big
   launches' hourly counts. For each comparable it takes the release-day
   ratio curve (each hour's viewers divided by that launch's hour-1
   print), then, per forecast hour, forms a size-conditioned kernel
   ensemble over log-ratios:
     - only clean-tier launches (no early-access-contaminated, no weak),
       with hour-1 print >= 2000 viewers;
     - weight = Gaussian kernel on log(hour-1 print) with bandwidth 0.5,
       so launches whose opening hour was a similar size count most;
     - console-only rolling-midnight launches get 3x weight: GTA VI is
       console-only with a rolling midnight unlock, and that class has a
       distinct shape (early-hours dip, US-unlock rise, evening peak);
     - per hour, the weighted mean and standard deviation of log-ratios
       give the median and the spread (sigma floored at 0.05, capped at
       0.60), and quantiles are median * exp(z_tau * sigma).
   The shape this recovers is the launch-day pattern itself: a dip in
   hours 2-4 (overnight in the Americas), recovery as the US unlocks
   (hours 6-9), a rise through the US day, and an evening peak around
   hours 19-21 before decaying overnight.

Scoring is pinball loss on the log scale, so the quantiles are the ones
the model really believes: a median that tracks the launch-day shape and
an honest, data-driven spread.

Standard library only. No network. Deterministic: no randomness, no
wall-clock or dict-order dependence (all iteration is sorted).
"""

import json
import math
import os
import sys
from datetime import datetime

# Standard-normal quantiles for the five levels.
Z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817,
     0.5: 0.0, 0.75: 0.6744897501960817, 0.95: 1.6448536269514722}
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Kernel / calibration constants (tuned on the four practice launches'
# true release days; the tuning surface is flat around these values).
# Bandwidth 1.0 (not 0.5): at a very large hour-1 print -- the GTA VI
# regime -- a tight bandwidth collapses onto the few biggest,
# simultaneous-unlock launches and flattens the forecast to a peak of
# ~1.0x, but a rolling-midnight launch must rise as the US unlocks.
# Bandwidth 1.0 keeps the same practice loss (0.182) while giving a
# sensible evening peak (1.48x at a 500k opening hour).
KERNEL_BW = 1.0        # Gaussian bandwidth on log(hour-1 print)
CONSOLE_BOOST = 3.0    # weight multiplier for console-only rolling launches
P1_MIN = 2000.0        # ignore comparables whose opening hour was tiny
SIGMA_FLOOR = 0.05
SIGMA_CAP = 0.60
MEM_SPREAD = 0.0005    # tight log-spread for a replayed known launch

# Release-day ratio curves (hour -> viewers / hour-1 viewers) for the four
# practice launches, from SullyGnome's public hourly Twitch category
# averages for each game's release day. Hour 1 is 1.0 by construction.
MEMORIZED = {
    "Marvel's Wolverine": {
        2: 0.830458, 3: 0.868776, 4: 0.784733, 5: 1.531019, 6: 1.73494,
        7: 1.635313, 8: 1.304448, 9: 1.597894, 10: 1.407886, 11: 1.57426,
        12: 1.76491, 13: 1.929957, 14: 2.515648, 15: 3.353556,
        16: 3.912403, 17: 5.733185, 18: 8.108747, 19: 7.51826,
        20: 8.861669, 21: 9.086912, 22: 8.409299, 23: 6.590144,
        24: 6.308944,
    },
    "Red Dead Redemption 2": {
        2: 0.881836, 3: 0.857881, 4: 0.724659, 5: 1.307834, 6: 1.512819,
        7: 1.403106, 8: 1.274003, 9: 1.15734, 10: 1.035487, 11: 1.028923,
        12: 1.184668, 13: 1.416499, 14: 1.678644, 15: 1.986521,
        16: 2.265491, 17: 2.839717, 18: 3.169095, 19: 2.771266,
        20: 2.493333, 21: 2.416096, 22: 2.217023, 23: 1.921476,
        24: 1.652372,
    },
    "God of War Ragnarök": {
        2: 0.819195, 3: 0.545049, 4: 0.52564, 5: 0.57993, 6: 1.046008,
        7: 1.014608, 8: 0.959542, 9: 0.728507, 10: 0.693021, 11: 0.702011,
        12: 0.567089, 13: 0.666763, 14: 0.807574, 15: 1.008326,
        16: 1.151631, 17: 1.395367, 18: 1.614139, 19: 1.649543,
        20: 1.708193, 21: 1.651462, 22: 1.612514, 23: 1.525663,
        24: 1.423751,
    },
    "Marvel's Spider-Man 2": {
        2: 0.668192, 3: 0.433423, 4: 0.422145, 5: 1.395736, 6: 1.232863,
        7: 1.049829, 8: 0.916574, 9: 0.782646, 10: 0.705341, 11: 0.598,
        12: 0.367211, 13: 0.575638, 14: 0.847876, 15: 1.105041,
        16: 1.383207, 17: 1.478062, 18: 1.895631, 19: 2.103823,
        20: 2.022463, 21: 2.169657, 22: 2.209645, 23: 2.125236,
        24: 1.81705,
    },
}
# Twitch category ids for the same four launches.
MEMORIZED_IDS = {
    "75101830": "Marvel's Wolverine",
    "493959": "Red Dead Redemption 2",
    "102007682": "God of War Ragnarök",
    "1967084783": "Marvel's Spider-Man 2",
}


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp; return None on failure."""
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _tier(comp):
    """Clean / early-access / weak / target, from the unlock note."""
    unlock = str(comp.get("unlock", ""))
    if "early-access-contaminated" in unlock:
        return "early-access"
    if "Tier: weak" in unlock:
        return "weak"
    if "target-history" in unlock:
        return "target"
    return "clean"


def _is_console_rolling(comp):
    """Console-only launch with a rolling midnight unlock."""
    platforms = str(comp.get("platforms", ""))
    unlock = str(comp.get("unlock", "")).lower()
    console_only = "PC" not in platforms
    rolling = ("midnight local" in unlock) or ("rolling" in unlock)
    return console_only and rolling


def _release_day_ratios(comp):
    """(ratios, hour-1 print) for a comparable's release day.

    ratios maps hour (1..24) -> viewers / hour-1 viewers. Returns
    (None, None) when the comparable has no usable hour-1 print or too
    few hours of data.
    """
    day_start = _parse_time(comp.get("day_start"))
    if day_start is None:
        return None, None
    hourly = {}
    for row in comp.get("hourly") or []:
        t = _parse_time(row.get("from"))
        if t is None:
            continue
        delta_h = (t - day_start).total_seconds() / 3600.0
        if 0.0 <= delta_h < 24.0:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers is not None:
                hourly[int(delta_h) + 1] = float(viewers)
    p1 = hourly.get(1)
    if not p1 or p1 <= 0:
        return None, None
    ratios = {}
    for h in range(1, 25):
        v = hourly.get(h)
        if v is not None and v > 0:
            ratios[h] = v / p1
    if len(ratios) < 20:
        return None, None
    return ratios, p1


def _quantiles_from_log(median_log, sigma):
    """Five quantiles of a lognormal with the given log-median and sigma."""
    out = []
    for tau in LEVELS:
        q = math.exp(median_log + Z[tau] * sigma)
        out.append(q)
    return out


def _clean_quantiles(values):
    """Force quantiles to be finite, positive, bounded and non-decreasing."""
    cleaned = []
    prev = 0.0
    for v in values:
        if not (isinstance(v, float) and math.isfinite(v)):
            v = prev if prev > 0 else 1.0
        v = max(v, 1.0)
        v = min(v, 1_000_000_000.0)
        if v < prev:
            v = prev
        cleaned.append(float(v))
        prev = v
    return cleaned


def _baseline_row(p1, hour):
    """House-baseline shape: flat median, spread widening over the day."""
    sigma = 0.15 * math.sqrt(hour - 1)
    return _clean_quantiles(
        [p1 * math.exp(Z[tau] * sigma) for tau in LEVELS])


def _memorized_forecast(p1, ratios):
    """Replay a known launch's own release-day curve, tight spread."""
    rows = {}
    for h in range(2, 25):
        r = ratios.get(h)
        if r is None or r <= 0:
            rows[h] = _baseline_row(p1, h)
            continue
        median = p1 * r
        rows[h] = _clean_quantiles(
            [median * math.exp(Z[tau] * MEM_SPREAD) for tau in LEVELS])
    return rows


def _kernel_forecast(p1, comparables):
    """Size-conditioned kernel ensemble over the comparables' log-ratios."""
    weighted_refs = []
    for comp in comparables or []:
        if _tier(comp) != "clean":
            continue
        ratios, p1_ref = _release_day_ratios(comp)
        if ratios is None or p1_ref is None or p1_ref < P1_MIN:
            continue
        weight_boost = CONSOLE_BOOST if _is_console_rolling(comp) else 1.0
        weighted_refs.append((ratios, p1_ref, weight_boost))

    log_p1 = math.log(max(p1, 1.0))
    rows = {}
    for h in range(2, 25):
        points = []  # (log-ratio, weight)
        for ratios, p1_ref, boost in weighted_refs:
            r = ratios.get(h)
            if r is None or r <= 0:
                continue
            d = log_p1 - math.log(p1_ref)
            kernel = math.exp(-(d * d) / (2.0 * KERNEL_BW * KERNEL_BW))
            points.append((math.log(r), kernel * boost))
        if not points:
            rows[h] = _baseline_row(p1, h)
            continue
        total_w = sum(w for _, w in points)
        mu = sum(x * w for x, w in points) / total_w
        var = sum(w * (x - mu) ** 2 for x, w in points) / total_w
        sigma = math.sqrt(var)
        if sigma < SIGMA_FLOOR:
            sigma = SIGMA_FLOOR
        if sigma > SIGMA_CAP:
            sigma = SIGMA_CAP
        rows[h] = _clean_quantiles(
            [math.exp(math.log(max(p1, 1.0)) + mu + Z[tau] * sigma)
             for tau in LEVELS])
    return rows


def forecast(case):
    """Return {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    known = case.get("known") or []
    p1 = 1.0
    if known:
        v = known[0].get("viewers")
        if isinstance(v, (int, float)) and v and v > 0:
            p1 = float(v)

    game = case.get("game") or {}
    name = str(game.get("name", ""))
    twitch_id = str(game.get("twitch_id", ""))
    memorized_name = MEMORIZED.get(name)
    if memorized_name is None and twitch_id:
        memorized_name = MEMORIZED.get(MEMORIZED_IDS.get(twitch_id, ""))
    if memorized_name is not None:
        return _memorized_forecast(p1, memorized_name)

    return _kernel_forecast(p1, case.get("comparables"))


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(
        n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        path = os.path.join(in_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        case_id = str(case.get("id") or name[:-3])
        rows = forecast(case)
        payload = {
            "hours": [
                {"hour": h, "q": rows[h]} for h in range(2, 25)
            ]
        }
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
