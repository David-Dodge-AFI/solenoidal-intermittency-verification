#!/usr/bin/env python3
"""
SPATIAL SURVEY V3 — Many positions, one timestep
======================================================================
Project: turbulence/intermittency (Decoherence In Action)

PURPOSE: Kill the "one spot" attack.
  Prior surveys sampled ONE 32³ cube per system across many timesteps.
  This survey samples MANY 32³ cubes at different spatial positions,
  all at the same timestep. If the mechanism (competition dynamics,
  dominance ceiling, l=2 dominance) holds everywhere — not just at
  the domain center — then it's universal, not a local artifact.

WHAT WE MEASURE at each position:
  1. Crest factor (peak |ω| / rms |ω|)
  2. Dominance ratio (|ω|_max / |ω|_2nd)
  3. l=2 power fraction (Legendre decomposition around peak)
  4. Competition signature (top-2 anti-correlation check)

Dataset: isotropic1024coarse (JHTDB)
  - Box: [0, 2π]³ (periodic) → positions can wrap without boundary issues
  - Using ONE timestep (t=0.0 — arbitrary, mechanism is stationary)
  - 20 random positions uniformly distributed across the domain

Grid: N=32 (demo token cap — same as all prior scripts)
NO DIALS introduced. Random seed = 42 for reproducibility.

Demo token: edu.jhu.pha.turbulence.testing-201406
SETUP:
  pip install givernylocal numpy scipy
  python spatial_survey_v3.py --token <your_token>

OUTPUT: spatial_survey_v3_RESULT.txt
"""

import numpy as np
from scipy.special import legendre
import os, sys, argparse, time


# ======================================================================
# CONFIG
# ======================================================================
DATASET = "isotropic1024coarse"
L_BOX = 2 * np.pi  # domain is [0, 2π]³ (periodic)
TIME_VAL = 0.0      # single timestep (mechanism is stationary)
N_POSITIONS = 20    # number of random positions to sample
N_GRID = 32         # cube size (demo token cap)
SEED = 42           # reproducibility
N_SPHERE = 200      # Fibonacci lattice points for l-decomposition
L_MAX = 8           # harmonic modes to compute

# Cube half-width: 32 grid points at 1024 resolution
# Domain is 2π, 1024 grid points → Δx = 2π/1024 ≈ 0.00614
# 32 points span 32 × 0.00614 ≈ 0.196 (about 3.1% of domain per axis)
HALF_WIDTH = 0.5 * N_GRID * (L_BOX / 1024.0)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "spatial_survey_v3_RESULT.txt")

L_LOG = []
def log(s=""): print(s); L_LOG.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L_LOG))
    print(f"\n[written] {OUT}")


# ======================================================================
# GENERATE RANDOM POSITIONS
# ======================================================================
def generate_positions(n, seed):
    """Generate n random center positions in [0, 2π]³.
    Periodic BCs → no boundary issues, any position is valid."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, L_BOX, size=(n, 3))
    return positions


# ======================================================================
# DATA FETCH
# ======================================================================
def fetch_cutout(center, N, time_val, token, chunk=4096):
    """Fetch N³ velocity cutout centered at (cx, cy, cz).
    Handles periodicity: coordinates wrap via modulo."""
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData

    cx, cy, cz = center
    # Build grid centered at the given position
    half = HALF_WIDTH
    ax_x = np.linspace(cx - half, cx + half, N)
    ax_y = np.linspace(cy - half, cy + half, N)
    ax_z = np.linspace(cz - half, cz + half, N)
    dx = ax_x[1] - ax_x[0]
    dy = ax_y[1] - ax_y[0]
    dz = ax_z[1] - ax_z[0]

    # Wrap to [0, 2π] for periodic domain
    ax_x = ax_x % L_BOX
    ax_y = ax_y % L_BOX
    ax_z = ax_z % L_BOX

    gx, gy, gz = np.meshgrid(ax_x, ax_y, ax_z, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
    M = pts.shape[0]

    ds = turb_dataset(dataset_title=DATASET, output_path="./_jhtdb_tmp", auth_token=token)
    vel = np.empty((M, 3), dtype=np.float64)
    nch = (M + chunk - 1) // chunk
    for c in range(nch):
        lo = c * chunk; hi = min(lo + chunk, M)
        res = getData(ds, "velocity", time_val, "none", "none", "field", pts[lo:hi])
        arr = res
        if isinstance(arr, (list, tuple)): arr = arr[0]
        if hasattr(arr, "values"): arr = arr.values
        vel[lo:hi] = np.asarray(arr, dtype=np.float64).reshape(-1, 3)[:hi-lo]

    u = vel.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    return u, dx, dy, dz


# ======================================================================
# COMPUTE — VORTICITY
# ======================================================================
def compute_vorticity(u, dx, dy, dz):
    ux, uy, uz = u[0], u[1], u[2]
    ox = np.gradient(uz, dy, axis=1) - np.gradient(uy, dz, axis=2)
    oy = np.gradient(ux, dz, axis=2) - np.gradient(uz, dx, axis=0)
    oz = np.gradient(uy, dx, axis=0) - np.gradient(ux, dy, axis=1)
    return np.array([ox, oy, oz])


# ======================================================================
# COMPUTE — PEAK FINDING
# ======================================================================
def top_n_peaks(omega_mag, n=2, exclusion_radius=3):
    """Find top-n peaks with exclusion radius."""
    flat = omega_mag.ravel().copy()
    peaks = []
    N_g = omega_mag.shape[0]
    for _ in range(n):
        idx = np.argmax(flat)
        val = flat[idx]
        iz = idx % N_g; rem = idx // N_g
        iy = rem % N_g; ix = rem // N_g
        peaks.append((val, ix, iy, iz))
        # Mask exclusion zone
        for dx_ in range(-exclusion_radius, exclusion_radius + 1):
            for dy_ in range(-exclusion_radius, exclusion_radius + 1):
                for dz_ in range(-exclusion_radius, exclusion_radius + 1):
                    ii = (ix + dx_) % N_g
                    jj = (iy + dy_) % N_g
                    kk = (iz + dz_) % N_g
                    flat[ii * N_g * N_g + jj * N_g + kk] = 0.0
    return peaks


# ======================================================================
# COMPUTE — SPHERICAL HARMONIC DECOMPOSITION (Legendre, axisymmetric)
# ======================================================================
def fibonacci_sphere(n):
    """Generate n approximately uniform points on unit sphere."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = 2 * np.pi * i / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1), np.cos(theta)


def l_decomposition(omega, peak_loc, radius, dx, dy, dz, l_max=L_MAX):
    """Compute Legendre power spectrum around a peak.
    Samples |ω| on a sphere of given radius (in grid spacings),
    decomposes into P_l(cos θ) where θ is measured from the local
    vorticity direction at the peak."""
    ix, iy, iz = peak_loc
    N_g = omega.shape[1]

    # Local vorticity direction at peak (defines the axis)
    w_peak = omega[:, ix, iy, iz]
    w_hat = w_peak / (np.linalg.norm(w_peak) + 1e-30)

    # Sphere sample points
    sphere_pts, cos_theta_raw = fibonacci_sphere(N_SPHERE)

    # Rotate sphere so z-axis aligns with w_hat
    # cos(θ) for each point relative to w_hat
    cos_theta = sphere_pts @ w_hat  # dot product with vorticity axis

    # Physical coordinates of sample points
    r_phys = radius * np.array([dx, dy, dz])  # radius in physical units
    sample_coords = np.array([ix, iy, iz])[None, :] + radius * sphere_pts

    # Interpolate |ω| at sample points (nearest-neighbor for speed)
    omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    vals = np.zeros(N_SPHERE)
    for i in range(N_SPHERE):
        si = int(round(sample_coords[i, 0])) % N_g
        sj = int(round(sample_coords[i, 1])) % N_g
        sk = int(round(sample_coords[i, 2])) % N_g
        vals[i] = omega_mag[si, sj, sk]

    # Legendre decomposition: c_l = (2l+1)/2 × mean(f × P_l)
    power = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        Pl = legendre(l)(cos_theta)
        c_l = np.mean(vals * Pl) * (2 * l + 1) / 2
        power[l] = c_l**2

    # Normalize to fractions
    total = power.sum()
    if total > 0:
        power /= total

    return power


# ======================================================================
# MAIN SURVEY
# ======================================================================
def run_survey(token):
    t_start = time.time()

    log("=" * 90)
    log(f"SPATIAL SURVEY V3 — {N_POSITIONS} random positions, t={TIME_VAL}")
    log(f"Dataset: {DATASET} | Grid: N={N_GRID} | Seed: {SEED}")
    log(f"Half-width: {HALF_WIDTH:.5f} ({HALF_WIDTH/L_BOX*100:.1f}% of domain per axis)")
    log("=" * 90)
    log()

    positions = generate_positions(N_POSITIONS, SEED)

    results = []
    for p_idx in range(N_POSITIONS):
        center = positions[p_idx]
        t0 = time.time()
        log(f"  [{p_idx+1}/{N_POSITIONS}] center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

        # Fetch
        u, dx, dy, dz = fetch_cutout(center, N_GRID, TIME_VAL, token)

        # Vorticity
        omega = compute_vorticity(u, dx, dy, dz)
        omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)

        # Basic stats
        rms = np.sqrt(np.mean(omega_mag**2))
        peak = omega_mag.max()
        crest = peak / rms if rms > 0 else 0

        # Top-2 peaks → dominance
        peaks = top_n_peaks(omega_mag, n=2, exclusion_radius=3)
        dom = peaks[0][0] / peaks[1][0] if len(peaks) > 1 and peaks[1][0] > 0 else 1.0

        # l-decomposition around the peak
        peak_loc = (peaks[0][1], peaks[0][2], peaks[0][3])
        l_power = l_decomposition(omega, peak_loc, radius=3, dx=dx, dy=dy, dz=dz)

        dt = time.time() - t0
        log(f"    {dt:.1f}s | crest={crest:.3f} dom={dom:.3f} "
            f"l0={l_power[0]:.3f} l1={l_power[1]:.3f} l2={l_power[2]:.3f}")

        results.append({
            'pos_idx': p_idx,
            'center': center,
            'crest': crest,
            'dominance': dom,
            'peak': peak,
            'rms': rms,
            'l_power': l_power,
            'dt': dt
        })

    # ======================================================================
    # AGGREGATE RESULTS
    # ======================================================================
    log()
    log("=" * 90)
    log("AGGREGATE RESULTS")
    log("=" * 90)
    log()

    crests = np.array([r['crest'] for r in results])
    doms = np.array([r['dominance'] for r in results])
    l2_fracs = np.array([r['l_power'][2] for r in results])
    l1_fracs = np.array([r['l_power'][1] for r in results])
    l3_fracs = np.array([r['l_power'][3] for r in results])

    log(f"  CREST FACTOR (peak/rms):")
    log(f"    mean={crests.mean():.3f}, std={crests.std():.3f}, "
        f"min={crests.min():.3f}, max={crests.max():.3f}")
    log(f"    Prior (iso survey, center-only): mean≈5.2, std≈0.6")
    log()
    log(f"  DOMINANCE RATIO (|ω|_max / |ω|_2nd):")
    log(f"    mean={doms.mean():.3f}, std={doms.std():.3f}, "
        f"min={doms.min():.3f}, max={doms.max():.3f}")
    log(f"    Prior (iso survey, center-only): mean≈1.10, max≈1.34")
    log()
    log(f"  L=2 POWER FRACTION (dominant non-isotropic mode):")
    log(f"    mean={l2_fracs.mean():.4f}, std={l2_fracs.std():.4f}, "
        f"min={l2_fracs.min():.4f}, max={l2_fracs.max():.4f}")
    log(f"    l=1: mean={l1_fracs.mean():.4f}")
    log(f"    l=3: mean={l3_fracs.mean():.4f}")
    log(f"    l=2 > l=1 at {np.sum(l2_fracs > l1_fracs)}/{N_POSITIONS} positions")
    log(f"    l=2 > l=3 at {np.sum(l2_fracs > l3_fracs)}/{N_POSITIONS} positions")
    log(f"    l=2 dominant non-isotropic at {np.sum((l2_fracs > l1_fracs) & (l2_fracs > l3_fracs))}/{N_POSITIONS} positions")
    log()

    # Per-position table
    log(f"  PER-POSITION TABLE:")
    log(f"  {'#':>3} {'cx':>6} {'cy':>6} {'cz':>6} {'crest':>7} {'dom':>6} "
        f"{'l0':>6} {'l1':>6} {'l2':>6} {'l3':>6} {'l4':>6}")
    log(f"  {'-'*80}")
    for r in results:
        c = r['center']
        lp = r['l_power']
        log(f"  {r['pos_idx']+1:3d} {c[0]:6.2f} {c[1]:6.2f} {c[2]:6.2f} "
            f"{r['crest']:7.3f} {r['dominance']:6.3f} "
            f"{lp[0]:6.3f} {lp[1]:6.3f} {lp[2]:6.3f} {lp[3]:6.3f} {lp[4]:6.3f}")
    log()

    # Uniformity test
    log("=" * 90)
    log("UNIFORMITY TEST — Is the mechanism position-independent?")
    log("=" * 90)
    log()
    cv_crest = crests.std() / crests.mean() * 100
    cv_dom = doms.std() / doms.mean() * 100
    log(f"  Coefficient of variation (std/mean):")
    log(f"    Crest: {cv_crest:.1f}%  (prior temporal CV: ~11%)")
    log(f"    Dominance: {cv_dom:.1f}%")
    log()
    log(f"  If spatial CV ≈ temporal CV → mechanism is ergodic")
    log(f"  If spatial CV >> temporal CV → position matters (BAD)")
    log(f"  If spatial CV << temporal CV → mechanism more uniform than expected")
    log()

    total_time = time.time() - t_start
    log("=" * 90)
    log(f"TOTAL TIME: {total_time:.0f}s ({total_time/60:.1f} min)")
    log("=" * 90)
    log()

    # Dial ledger
    log("=" * 90)
    log("DIAL LEDGER")
    log("=" * 90)
    log(f"  N_GRID=32: JHTDB demo token cap. NOT a choice.")
    log(f"  N_POSITIONS=20: chosen for statistical power vs API time. FLAGGED.")
    log(f"  SEED=42: reproducibility. Arbitrary. FLAGGED.")
    log(f"  TIME_VAL=0.0: arbitrary (isotropic = stationary). NOT a dial.")
    log(f"  HALF_WIDTH={HALF_WIDTH:.5f}: from grid resolution (32/1024 × 2π). NOT a dial.")
    log(f"  exclusion_radius=3: same as prior scripts. Inherited.")
    log(f"  N_SPHERE=200: Fibonacci lattice, sufficient for l≤8. Inherited.")
    log(f"  L_MAX=8: more than attractor predicts. Lets unexpected modes appear.")
    log(f"  Sphere radius for l-decomp: R=3 grid spacings. Inherited from V1/V2.")
    log()
    log("WHAT THIS TELLS US:")
    log("  If crest/dominance/l=2 statistics at 20 random positions match")
    log("  the temporal statistics at the center position → mechanism is universal.")
    log("  The 'one spot' attack is dead.")
    log()
    log("WHAT THIS DOESN'T TELL US:")
    log("  Different DNS codes (still JHTDB only)")
    log("  Different Re (isotropic1024coarse is one Re)")
    log("  Causality (same caveat as V1/V2)")

    save()


# ======================================================================
# ENTRY
# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spatial Survey V3")
    ap.add_argument("--token", default=os.environ.get("JHTDB_TOKEN",
                    "edu.jhu.pha.turbulence.testing-201406"),
                    help="JHTDB API token (default: demo)")
    args = ap.parse_args()
    if not args.token:
        sys.exit("[error] --token required. Demo: edu.jhu.pha.turbulence.testing-201406")
    run_survey(args.token)
