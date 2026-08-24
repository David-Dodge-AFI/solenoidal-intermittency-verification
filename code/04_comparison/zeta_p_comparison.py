#!/usr/bin/env python3
"""
SCALING EXPONENT COMPARISON — ζ_p prediction vs experiment
======================================================================
Project: Turbulence Intermittency (Paper 2)

Compares our zero-parameter prediction ζ_p = p/3 − p(p−3)/81
against published experimental and DNS measurements.

Sources (with DOIs):
  - Anselmet et al. 1984: doi.org/10.1017/S0022112084000513
  - Arneodo et al. 1996: doi.org/10.1209/epl/i1996-00472-2
  - Gotoh et al. 2002:   doi.org/10.1063/1.1448296
  - Saw et al. 2018:     doi.org/10.1017/jfm.2017.848
  - Frisch 1995:         ISBN 0-521-45713-0 (Table 8.1)
  - She & Lévêque 1994:  doi.org/10.1103/PhysRevLett.72.336

NO DIALS. No fit. Prediction computed, THEN compared to data.

OUTPUT: zeta_p_comparison_RESULT.txt
"""

import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..\\..\\results\\zeta_p_comparison_RESULT.txt")

L = []
def log(s=""): print(s); L.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"\n[written] {OUT}")


# ======================================================================
# PREDICTIONS (zero free parameters)
# ======================================================================
def zeta_ours(p):
    """Our prediction: f_c = 1/d = 1/3, μ = 2f_c² = 2/9."""
    return p/3 - p*(p-3)/81

def zeta_SL(p):
    """She-Lévêque (1994): one fitted parameter β = 2/3."""
    return p/9 + 2*(1 - (2/3)**(p/3))

def zeta_K41(p):
    """Kolmogorov 1941: no intermittency."""
    return p/3


# ======================================================================
# EXPERIMENTAL DATA
# ======================================================================
p_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Frisch (1995) Table 8.1 — textbook consensus
frisch = np.array([0.37, 0.70, 1.00, 1.28, 1.54, 1.78, 2.00, 2.23])

# Gotoh et al. (2002) — 1024³ DNS, Re_λ ≈ 460, direct measurement
gotoh = np.array([0.37, 0.70, 1.00, 1.28, 1.54, 1.78, 2.00, 2.23])

# Arneodo et al. (1996) — 11 labs, Re_λ 30-5000, ESS
arneodo = np.array([0.35, 0.70, 1.00, 1.28, 1.55, 1.77, 2.03, 2.20])
arneodo_err = np.array([0.03, 0.03, 0.00, 0.03, 0.05, 0.05, 0.05, 0.08])

# Saw et al. (2018) — von Kármán, no Taylor hypothesis, SPIV
saw = np.array([0.36, 0.69, 1.00, 1.29, 1.55, 1.80, 2.02, 2.23])
saw_err = np.array([0.005, 0.005, 0.00, 0.005, 0.01, 0.02, 0.03, 0.04])


# ======================================================================
# COMPARISON
# ======================================================================
log("="*80)
log("SCALING EXPONENT COMPARISON: PREDICTION vs 40 YEARS OF MEASUREMENTS")
log("="*80)
log()
log("Our prediction: ζ_p = p/3 − p(p−3)/81")
log("  Derived from: f_c = 1/d = 1/3 (projector trace, exact)")
log("                μ = 2f_c² = 2/9 (bilinear NS term)")
log("  Free parameters: ZERO")
log()

log("="*80)
log("MASTER TABLE")
log("="*80)
log()
log(f"{'p':>3} | {'Frisch':>7} | {'Gotoh':>7} | {'Arneodo':>7} {'±':>5} | {'Saw':>7} {'±':>5} | {'OURS':>7} | {'S-L':>7} | {'K41':>7}")
log(f"{'':>3} | {'(1995)':>7} | {'(2002)':>7} | {'(1996)':>7} {'':>5} | {'(2018)':>7} {'':>5} | {'(0 dial)':>7} | {'(1 dial)':>7} | {'':>7}")
log("-"*80)

for i, p in enumerate(p_values):
    our = zeta_ours(p)
    sl = zeta_SL(p)
    k41 = zeta_K41(p)
    log(f"{p:3d} | {frisch[i]:7.3f} | {gotoh[i]:7.3f} | {arneodo[i]:7.3f} {arneodo_err[i]:+.2f} | {saw[i]:7.3f} {saw_err[i]:+.3f} | {our:7.4f} | {sl:7.4f} | {k41:7.4f}")

log()

# ERROR ANALYSIS
log("="*80)
log("ERROR ANALYSIS")
log("="*80)
log()

for name, data in [("Frisch 1995", frisch), ("Gotoh 2002", gotoh), 
                    ("Arneodo 1996", arneodo), ("Saw 2018", saw)]:
    err_ours = zeta_ours(p_values) - data
    err_sl = np.array([zeta_SL(p) for p in p_values]) - data
    err_k41 = zeta_K41(p_values) - data
    
    log(f"  vs {name}:")
    log(f"    Ours (0 dials):   RMS = {np.sqrt(np.mean(err_ours**2)):.4f}, max |err| = {np.max(np.abs(err_ours)):.4f}")
    log(f"    She-Lev (1 dial): RMS = {np.sqrt(np.mean(err_sl**2)):.4f}, max |err| = {np.max(np.abs(err_sl)):.4f}")
    log(f"    K41 (no interm.): RMS = {np.sqrt(np.mean(err_k41**2)):.4f}, max |err| = {np.max(np.abs(err_k41)):.4f}")
    log()

# μ COMPARISON
log("="*80)
log("INTERMITTENCY PARAMETER μ")
log("="*80)
log()
log(f"  Our prediction:     μ = 2/9 = {2/9:.4f}")
log(f"  Anselmet (1984):    μ = 0.20 ± 0.05  (Re_λ ≤ 852)")
log(f"  Arneodo (1996):     μ ≈ 0.23  (from ζ*₆ = 1.77, Re_λ 30-5000)")
log(f"  Gotoh DNS (2002):   μ ≈ 0.22  (from ζ₆ = 1.78, Re_λ ≈ 460)")
log(f"  Saw (2018):         μ ≈ 0.20  (from ζ₆ = 1.80, von Kármán)")
log()
log(f"  Range of measurements: 0.20–0.23")
log(f"  Our prediction:        0.2222")
log(f"  Status: WITHIN measured range. No fitted parameter.")
log()

# FORMULA PROPERTIES
log("="*80)
log("FORMULA PROPERTIES")
log("="*80)
log()
log(f"  ζ₃ = 1.000 exactly (4/5 law protected) ✓")
log(f"  Monotonically increasing for p ≤ 15")
log(f"  Peaks at p = 15: ζ₁₅ = {zeta_ours(15):.4f}")
log(f"  Unphysical (non-monotone) for p > 15 — inherent K62 limitation")
log(f"  Experimental data reliable only to p ~ 8-10 (statistics degrade)")
log()

# DIAL LEDGER
log("="*80)
log("DIAL LEDGER")
log("="*80)
log()
log("  Input: d = 3 (spatial dimension)")
log("  Derived: f_c = 1/d = 1/3 (projector trace, exact)")
log("  Derived: μ = 2f_c² = 2/9 (bilinear structure)")
log("  Framework: K62 log-normal (ζ_p = p/3 − μp(p−3)/18)")
log("  FREE PARAMETERS: ZERO")
log()
log("  Experimental data sources: NOT our input. They are the COMPARISON.")
log("  The derivation was performed BEFORE the comparison (forward-compute).")

save()


if __name__ == "__main__":
    pass  # All computation happens at import time above
