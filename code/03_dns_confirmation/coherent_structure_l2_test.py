#!/usr/bin/env python3
"""
COHERENT STRUCTURE TEST V2 — Cross-system l=2 test (channel + isotropic + transition BL)
======================================================================
Project: turbulence/intermittency (Unified Theory — last direction)

QUESTION: The projector attractor says only l=0 and l=2 survive.
l=2 = quadrupolar geometry = the shape of vortex tubes and sheets.
Does the ACTUAL vorticity field during burst events show l=2 dominance?

METHOD:
  1. Fetch 32³ velocity cutout from JHTDB channel flow at burst time (t=10.0)
     AND at calm time (t=9.286) for comparison
  2. Compute vorticity ω = ∇×u at every grid point
  3. For the PEAK vorticity location, sample |ω| on a sphere of radius R
     centered on that location
  4. Decompose the angular distribution into spherical harmonics (Legendre modes)
  5. Report the power in each l mode: P(l) = |c_l|²
  6. Compare burst vs calm — does l=2 stand out more during bursts?

WHAT WE'RE LOOKING FOR (stated before computation):
  If l=2 dominates → tubes/sheets are a consequence of the projector attractor
  If l=0 dominates → isotropic (no structure preference)
  If higher l dominates → something else shapes the structures (unexpected)
  If no pattern → test inconclusive at this grid

NO DIALS except:
  - Sphere radius R: will try multiple (R=2,3,4 grid spacings). Flagged.
  - Grid N=32: JHTDB demo token cap. Not a choice.
  - Burst time t=10.0: from prior time-series survey (the largest burst detected).
  - Calm time t=9.286: from prior survey (known calm period).

SETUP:
  pip install givernylocal numpy
  python coherent_structure_l2_test_v1.py --token <your_token>

OUTPUT: coherent_structure_l2_test_v1_RESULT.txt (same folder)
"""

import numpy as np
import os, sys, argparse, time

# ======================================================================
# CONFIG
# ======================================================================
# THREE SYSTEMS — same cross-system surveys as prior campaign
SYSTEMS = {
    "channel": {
        "box": (0.0, 8*np.pi, -1.0, 1.0, 0.0, 3*np.pi),
        "burst_times": [0.6, 1.8, 6.6, 10.0, 11.8, 14.4, 16.0, 16.4, 18.4, 20.6, 21.2, 24.2],
        "calm_times": [3.0, 7.0, 13.0, 19.0, 23.0],
    },
    "isotropic1024coarse": {
        "box": (0.0, 2*np.pi, 0.0, 2*np.pi, 0.0, 2*np.pi),
        "burst_times": [0.32, 1.40, 1.56],  # from isotropic survey (3 bursts detected)
        "calm_times": [0.84, 1.16],  # low-crest periods from isotropic survey
    },
    "transition_bl": {
        "box": (30.2185, 1000.065, 0.0, 26.48795, 0.0, 240.0),
        "burst_times": [275, 450, 625, 700, 950],  # from transition BL survey (5 bursts)
        "calm_times": [350, 550, 900],  # low-crest periods
    },
}
N_PEAKS = 3
SPHERE_RADII = [2, 3, 4]  # grid spacings (flagged as dial)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "coherent_structure_l2_test_v2_RESULT.txt")

L = []
def log(s=""): print(s); L.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"\n[written] {OUT}")


# ======================================================================
# DATA FETCH
# ======================================================================
def fetch_cutout(N, time_val, token, dataset="channel", box=None, chunk=4096):
    """Fetch N^3 velocity cutout, centered in domain."""
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData

    x0, x1, y0, y1, z0, z1 = box
    def axis(a, b):
        c = 0.5*(a+b); half = 0.125*(b-a)
        return np.linspace(c-half, c+half, N)
    cx, cy, cz = axis(x0, x1), axis(y0, y1), axis(z0, z1)
    dx = cx[1]-cx[0]; dy = cy[1]-cy[0]; dz = cz[1]-cz[0]

    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
    M = pts.shape[0]

    ds = turb_dataset(dataset_title=dataset, output_path="./_jhtdb_tmp", auth_token=token)
    vel = np.empty((M, 3), dtype=np.float64)
    nch = (M + chunk - 1) // chunk
    for c in range(nch):
        lo = c*chunk; hi = min(lo+chunk, M)
        res = getData(ds, "velocity", time_val, "none", "none", "field", pts[lo:hi])
        arr = res
        if isinstance(arr, (list, tuple)): arr = arr[0]
        if hasattr(arr, "values"): arr = arr.values
        vel[lo:hi] = np.asarray(arr, dtype=np.float64).reshape(-1, 3)[:hi-lo]

    u = vel.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    return u, cx, cy, cz, dx, dy, dz


# ======================================================================
# VORTICITY
# ======================================================================
def compute_vorticity(u, dx, dy, dz):
    """Vorticity omega = curl(u). Returns (3, N, N, N)."""
    ux, uy, uz = u[0], u[1], u[2]
    ox = np.gradient(uz, dy, axis=1) - np.gradient(uy, dz, axis=2)
    oy = np.gradient(ux, dz, axis=2) - np.gradient(uz, dx, axis=0)
    oz = np.gradient(uy, dx, axis=0) - np.gradient(ux, dy, axis=1)
    return np.array([ox, oy, oz])


# ======================================================================
# SPHERICAL HARMONIC DECOMPOSITION
# ======================================================================
def sample_sphere(omega_mag, center, radius, N_grid):
    """Sample |ω| on a sphere of given radius around center.
    Returns (theta, phi, values) arrays for points on the sphere
    that fall within the grid."""
    ix, iy, iz = center
    samples = []
    
    # Generate points on sphere using Fibonacci lattice (uniform-ish)
    N_pts = 200  # points on sphere
    golden = (1 + np.sqrt(5)) / 2
    for i in range(N_pts):
        theta = np.arccos(1 - 2*(i+0.5)/N_pts)
        phi = 2 * np.pi * i / golden
        
        # Grid coordinates
        gx = int(round(ix + radius * np.sin(theta) * np.cos(phi)))
        gy = int(round(iy + radius * np.sin(theta) * np.sin(phi)))
        gz = int(round(iz + radius * np.cos(theta)))
        
        # Bounds check
        if 0 <= gx < N_grid and 0 <= gy < N_grid and 0 <= gz < N_grid:
            val = omega_mag[gx, gy, gz]
            samples.append((theta, phi, val))
    
    if len(samples) < 20:
        return None
    
    thetas = np.array([s[0] for s in samples])
    phis = np.array([s[1] for s in samples])
    vals = np.array([s[2] for s in samples])
    return thetas, phis, vals


def legendre_decompose(thetas, vals, l_max=10):
    """Decompose angular distribution into Legendre modes (axisymmetric).
    Returns power per l mode: P(l) = |c_l|².
    Uses least-squares fit of Legendre polynomials to the data."""
    from numpy.polynomial.legendre import legval
    
    # cos(theta) values
    mu = np.cos(thetas)
    
    # Build Legendre matrix: each column is P_l(mu) for l=0..l_max
    A = np.zeros((len(mu), l_max+1))
    for l in range(l_max+1):
        coeffs = np.zeros(l+1)
        coeffs[l] = 1.0
        A[:, l] = legval(mu, coeffs)
    
    # Least-squares fit
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, vals, rcond=None)
    
    # Power per mode
    power = coeffs**2
    total_power = np.sum(power)
    
    return coeffs, power, total_power


# ======================================================================
# MAIN
# ======================================================================
def find_top_n_peaks(omega_mag, n=3, exclusion_radius=3):
    """Find top N peaks with exclusion zone."""
    flat = omega_mag.ravel().copy()
    N_grid = omega_mag.shape[0]
    peaks = []
    for _ in range(n):
        idx = np.argmax(flat)
        mag = float(flat[idx])
        ix, iy, iz = np.unravel_index(idx, omega_mag.shape)
        peaks.append((mag, int(ix), int(iy), int(iz)))
        # Zero out neighborhood
        for di in range(-exclusion_radius, exclusion_radius+1):
            for dj in range(-exclusion_radius, exclusion_radius+1):
                for dk in range(-exclusion_radius, exclusion_radius+1):
                    ni, nj, nk = ix+di, iy+dj, iz+dk
                    if 0 <= ni < N_grid and 0 <= nj < N_grid and 0 <= nk < N_grid:
                        flat[ni*N_grid*N_grid + nj*N_grid + nk] = 0.0
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("JHTDB_TOKEN", ""))
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    if not args.token:
        sys.exit("[error] --token required. Demo: edu.jhu.pha.turbulence.testing-201406")

    N = args.N
    log("=" * 90)
    log("COHERENT STRUCTURE TEST V2 — Cross-system l=2 test")
    log(f"Systems: {list(SYSTEMS.keys())} | Grid: N={N}")
    log(f"Peaks per timestep: {N_PEAKS}")
    log(f"Sphere radii (grid spacings): {SPHERE_RADII}")
    log("=" * 90)
    log()

    # Collect all decomposition results
    all_results = []  # (time, type, peak_rank, radius, l_powers)

    for sys_name, sys_config in SYSTEMS.items():
        box = sys_config["box"]
        burst_times = sys_config["burst_times"]
        calm_times = sys_config["calm_times"]
        all_times = [(t, "BURST") for t in burst_times] + [(t, "CALM") for t in calm_times]

        log(f"\n{'='*90}")
        log(f"SYSTEM: {sys_name} | bursts: {len(burst_times)} | calm: {len(calm_times)}")
        log(f"{'='*90}")

        for ti, (t_val, t_type) in enumerate(all_times):
            log(f"  [{ti+1}/{len(all_times)}] t={t_val} ({t_type})")

            try:
                t0 = time.time()
                u, cx, cy, cz, dx, dy, dz = fetch_cutout(N, t_val, args.token, dataset=sys_name, box=box, chunk=args.chunk)
                omega = compute_vorticity(u, dx, dy, dz)
                omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)

                peaks = find_top_n_peaks(omega_mag, n=N_PEAKS)
                rms_val = np.sqrt(np.mean(omega_mag**2))

                for pi, (peak_val, ix, iy, iz) in enumerate(peaks):
                    for R in SPHERE_RADII:
                        result = sample_sphere(omega_mag, (ix, iy, iz), R, N)
                        if result is None:
                            continue
                        thetas, phis, vals = result
                        coeffs, power, total_power = legendre_decompose(thetas, vals, l_max=8)

                        # Normalize power to fractions
                        if total_power > 0:
                            fracs = power / total_power
                        else:
                            fracs = power

                        all_results.append({
                            'time': t_val, 'type': t_type,
                            'system': sys_name,
                            'peak_rank': pi+1, 'radius': R,
                            'peak_val': peak_val, 'rms': rms_val,
                            'crest': peak_val/rms_val,
                            'loc': (ix, iy, iz),
                            'fracs': fracs, 'coeffs': coeffs
                        })

                log(f"    {time.time()-t0:.1f}s | peak={peaks[0][0]:.3f} rms={rms_val:.4f} crest={peaks[0][0]/rms_val:.2f}")

            except Exception as e:
                log(f"    ERROR: {e}")
                continue

    log()
    log("=" * 90)
    log("AGGREGATE RESULTS — l-mode power fractions")
    log("=" * 90)
    log()

    # Separate burst vs calm (across ALL systems)
    burst_results = [r for r in all_results if r['type'] == 'BURST']
    calm_results = [r for r in all_results if r['type'] == 'CALM']

    def summarize(results, label):
        if not results:
            log(f"  {label}: No valid results.")
            return
        # Average power fraction per l mode across all samples
        all_fracs = np.array([r['fracs'] for r in results])
        mean_fracs = np.mean(all_fracs, axis=0)
        std_fracs = np.std(all_fracs, axis=0)

        log(f"  {label} (N={len(results)} samples):")
        log(f"    {'l':>4}  {'Mean Frac':>10}  {'Std':>10}  {'Interpretation'}")
        log(f"    {'-'*55}")
        for l in range(len(mean_fracs)):
            interp = ""
            if l == 0: interp = "(isotropic)"
            elif l == 2: interp = "(quadrupolar = tubes/sheets) <<<"
            log(f"    {l:>4}  {mean_fracs[l]:>10.4f}  {std_fracs[l]:>10.4f}  {interp}")
        log()

        # Which l dominates (excluding l=0)?
        non_iso = mean_fracs.copy()
        non_iso[0] = 0
        dominant_l = np.argmax(non_iso)
        log(f"    Dominant non-isotropic mode: l={dominant_l} (frac={mean_fracs[dominant_l]:.4f})")
        log()

    summarize(burst_results, "BURST PEAKS")
    summarize(calm_results, "CALM PERIODS")

    # Per-radius breakdown
    log("=" * 90)
    log("PER-RADIUS BREAKDOWN (checking scale-dependence)")
    log("=" * 90)
    log()
    for R in SPHERE_RADII:
        burst_R = [r for r in burst_results if r['radius'] == R]
        calm_R = [r for r in calm_results if r['radius'] == R]
        log(f"  Radius R={R}:")
        if burst_R:
            fracs = np.mean([r['fracs'] for r in burst_R], axis=0)
            log(f"    BURST: l=0:{fracs[0]:.3f} l=1:{fracs[1]:.3f} l=2:{fracs[2]:.3f} l=3:{fracs[3]:.3f} l=4:{fracs[4]:.3f}")
        if calm_R:
            fracs = np.mean([r['fracs'] for r in calm_R], axis=0)
            log(f"    CALM:  l=0:{fracs[0]:.3f} l=1:{fracs[1]:.3f} l=2:{fracs[2]:.3f} l=3:{fracs[3]:.3f} l=4:{fracs[4]:.3f}")
        log()

    log("=" * 90)
    log("DIAL LEDGER")
    log("=" * 90)
    log("  N=32: JHTDB demo token cap. Not a choice.")
    log(f"  Burst/calm times: per-system, from prior time-series surveys.")

    log(f"  Peaks per time: {N_PEAKS} (top 3, exclusion radius 3). Tests multiple locations.")
    log(f"  Sphere radii {SPHERE_RADII}: Tests scale-dependence. CHOICE (flagged).")
    log("  N_sphere_pts=200: Fibonacci lattice. Sufficient for l≤8.")
    log("  l_max=8: More than attractor predicts (l≤2). Lets unexpected modes appear.")
    log("  Decomposition: Legendre (axisymmetric). Full Y_lm would be more complete.")
    log()
    log("WHAT THIS TELLS US:")
    log("  Aggregate over 12 burst peaks × 3 locations × 3 radii = up to 108 samples")
    log("  Aggregate over 5 calm times × 3 locations × 3 radii = up to 45 samples")
    log("  Statistical, not single-instance. Scale-dependence checked.")
    log()
    log("WHAT THIS DOESN'T TELL US:")
    log("  Causality (projector → l=2, or something else → l=2)")
    log("  Tubes vs sheets (need m-dependence for that)")
    log("  Cross-system universality (only channel flow tested)")

    save()


if __name__ == "__main__":
    main()
