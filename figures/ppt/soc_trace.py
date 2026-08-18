r"""
soc_trace.py -- the single synthetic state-of-charge trajectory shared by the definitions figure and the rainflow explainer.

Both slides show the same battery trace, so the audience recognizes it the second time and does not have to re-orient. Keeping the knots here means the
two figures cannot drift apart.

The turning points are chosen so that the two cycles annotated on the definitions slide are cycles the rainflow counter actually extracts, as full
cycles, between turning points that are adjacent in time:

    shallow  knots 4 -> 5    0.30 -> 0.48   delta = 0.18,  sigma = 0.39,  n = 1
    deep     knots 8 -> 9    0.18 -> 0.88   delta = 0.70,  sigma = 0.53,  n = 1

This matters because both slides show this trace. Rainflow pairs turning points that are not adjacent in time, so a large visible swing is often
counted as part of a differently paired cycle. Annotating a swing that the counter does not return would make the two slides contradict each other.

Run `python soc_trace.py` to check both annotated cycles against the rainflow counter. Edit the knots only if that check still passes afterward.
"""

import numpy as np

SIG_MIN, SIG_MAX = 0.10, 0.90
CENTER = 0.5 * (SIG_MIN + SIG_MAX)
WIDTH = SIG_MAX - SIG_MIN

T_END = 50.0

KNOTS_T = [0.0, 3.5, 7.0, 10.5, 14.0, 17.5,
           21.0, 24.5, 28.0, 31.5, 35.0, 38.5, 42.0, 45.5, 48.0, 50.0]
KNOTS_S = [0.50, 0.90, 0.20, 0.62, 0.30, 0.48,
           0.25, 0.68, 0.18, 0.88, 0.10, 0.55, 0.35, 0.72, 0.28, 0.55]

# The two cycles annotated on the definitions slide, as (knot index, knot+1).
SHALLOW_KNOT = 4          # 0.30 -> 0.48,  delta = 0.18,  sigma = 0.39
DEEP_KNOT = 8             # 0.18 -> 0.88,  delta = 0.70,  sigma = 0.53

assert len(KNOTS_T) == len(KNOTS_S)
assert min(KNOTS_S) >= SIG_MIN and max(KNOTS_S) <= SIG_MAX, \
    "every turning point must lie inside the operating window"


def trace(n=1200, normalized_time=False):
    """Return (t, soc). Set normalized_time=True for a 0..1 time axis.

    The sample grid always contains the knots exactly. Without that, the
    interpolated turning points land slightly off the knot values and the
    extracted amplitudes drift by a few thousandths, which would break the
    match against the annotated cycles.
    """
    t = np.union1d(np.linspace(0.0, T_END, n), np.asarray(KNOTS_T, float))
    soc = np.interp(t, KNOTS_T, KNOTS_S)
    return (t / T_END, soc) if normalized_time else (t, soc)


def turning_points():
    """Return the knot values, which are the turning points rainflow sees."""
    return np.asarray(KNOTS_S, dtype=float)


def _cycle_from_knot(k):
    """(t0, t1, s0, s1, delta, sigma) for the cycle between knots k and k+1."""
    s0, s1 = KNOTS_S[k], KNOTS_S[k + 1]
    return (KNOTS_T[k], KNOTS_T[k + 1], s0, s1,
            abs(s1 - s0), 0.5 * (s0 + s1))


DEEP_CYCLE = _cycle_from_knot(DEEP_KNOT)
SHALLOW_CYCLE = _cycle_from_knot(SHALLOW_KNOT)


def verify_against_counter(verbose=True):
    """Confirm both annotated cycles are full cycles the counter returns.

    Kept out of module scope so importing this trace does not require the
    rainflow package. Run `python soc_trace.py` after editing any knot.
    """
    import rainflow

    cycles = list(rainflow.extract_cycles(turning_points()))
    if verbose:
        print(f"{'delta':>7}{'sigma':>8}{'n':>6}")
        for rng, mean, n, *_ in cycles:
            print(f"{rng:7.3f}{mean:8.3f}{n:6.1f}")
        print()

    for name, (_, _, _, _, delta, sigma) in (("deep", DEEP_CYCLE),
                                             ("shallow", SHALLOW_CYCLE)):
        hits = [c for c in cycles
                if abs(c[0] - delta) < 1e-9 and abs(c[1] - sigma) < 1e-9]
        if not hits:
            raise AssertionError(
                f"The {name} cycle annotated on the definitions slide "
                f"(delta {delta:.2f}, sigma {sigma:.2f}) is not a cycle the "
                f"counter returns. The two slides would contradict each other. "
                f"Adjust the knots until it is.")
        if hits[0][2] != 1.0:
            raise AssertionError(
                f"The {name} cycle is counted with multiplicity "
                f"{hits[0][2]}, not 1.0. The slide labels it as a full cycle.")
        if verbose:
            print(f"  {name:8s} delta {delta:.2f}  sigma {sigma:.2f}  "
                  f"n {hits[0][2]:.1f}  ok")


if __name__ == "__main__":
    verify_against_counter()
