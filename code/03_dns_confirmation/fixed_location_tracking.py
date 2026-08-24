#!/usr/bin/env python3
"""
INTERMITTENCY FIXED LOCATION TRACKING V1 — Parallel life stories
======================================================================
Project: turbulence/intermittency (Decoherence In Action)

V3 showed the burst and the competition, but only through the lens of
"whoever is king at each instant." This script tracks FIXED GRID
LOCATIONS through the full time window — so we see every competitor's
arc simultaneously, including when they're NOT the king.

TRACKED LOCATIONS (from V3 top-3 data):
  A: (6, 31, 17)  — the big burst king (t=9.98-10.04)
  B: (28, 0, 16)  — previous king (t=9.90-9.92)
  C: (13, 31, 27) — mid-window king (t=9.95-9.97)
  D: (0, 0, 12)   — successor king (t=10.05-10.07)
  E: (29, 0, 16)  — late competitor (appears in #2/#3 late)

AT EACH LOCATION, EACH TIMESTEP:
  1. |omega| magnitude (the life arc of that structure)
  2. Strain eigenvalues (what's stretching/compressing it)
  3. Effective stretching rate (is it being fed or starved?)
  4. Local coherence (how organized is it?)

TIME WINDOW: t=9.90 to 10.10, step 0.01 (same as V3 — 21 timesteps)

This shows us:
  - Does the king's growth suppress the others? (energy theft)
  - Does the king's death feed the successor? (redistribution)
  - Are they independent or coupled?

DATA SOURCE: JHTDB channel flow
GRID: N=32 (demo token cap)
NO DIALS. No fit. Raw numbers.

SETUP:
  pip install givernylocal numpy
  python intermittency_fixed_location_tracking_v1.py --token <your_token>

OUTPUT: intermittency_fixed_location_tracking_v1_RESULT.txt (same folder)
"""

import numpy as np
import os, sys, argparse, time

# ======================================================================
# CONFIG
# ======================================================================
DATASET = "channel"
BOX = (0.0, 8*np.pi, -1.0, 1.0, 0.0, 3*np.pi)

# Same time window as V3
TIMES = [round(9.90 + i*0.01, 2) for i in range(21)]

# Fixed locations to track (from V3 top-3 analysis)
LOCATIONS = {
    "A": (6, 31, 17),   # big burst king
    "B": (28, 0, 16),   # previous king
    "C": (13, 31, 27),  # mid-window king
    "D": (0, 0, 12),    # successor king
    "E": (29, 0, 16),   # late competitor
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "intermittency_fixed_location_tracking_v1_RESULT.txt")

L = []
def log(s=""): print(s); L.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"\n[written] {OUT}")


# ======================================================================
# DATA FETCH
# ======================================================================
def fetch_cutout(N, time_val, token, chunk=4096):
    """Fetch N^3 velocity cutout from channel flow, centered in domain."""
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData

    x0, x1, y0, y1, z0, z1 = BOX
    def axis(a, b):
        c = 0.5*(a+b); half = 0.125*(b-a)
        return np.linspace(c-half, c+half, N)
    cx, cy, cz = axis(x0, x1), axis(y0, y1), axis(z0, z1)
    dx = cx[1]-cx[0]; dy = cy[1]-cy[0]; dz = cz[1]-cz[0]

    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
    M = pts.shape[0]

    ds = turb_dataset(dataset_title=DATASET, output_path="./_jhtdb_tmp", auth_token=token)
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
# COMPUTE
# ======================================================================
def compute_vorticity(u, dx, dy, dz):
    """Vorticity omega = curl(u). Returns (3, N, N, N)."""
    ux, uy, uz = u[0], u[1], u[2]
    ox = np.gradient(uz, dy, axis=1) - np.gradient(uy, dz, axis=2)
    oy = np.gradient(ux, dz, axis=2) - np.gradient(uz, dx, axis=0)
    oz = np.gradient(uy, dx, axis=0) - np.gradient(ux, dy, axis=1)
    return np.array([ox, oy, oz])


def compute_strain_at_point(u, dx, dy, dz, ix, iy, iz):
    """Compute strain tensor S_ij at a single grid point."""
    S = np.zeros((3, 3))
    spacings = [dx, dy, dz]
    for i in range(3):
        for j in range(3):
            grad = np.gradient(u[i], spacings[j], axis=j)
            S[i, j] = grad[ix, iy, iz]
    S = 0.5 * (S + S.T)
    return S


def local_coherence(omega, ix, iy, iz, radius=2):
    """Mean |cos(angle)| between omega at center and all neighbors within radius."""
    N = omega.shape[1]
    omega_c = omega[:, ix, iy, iz]
    mag_c = np.linalg.norm(omega_c)
    if mag_c < 1e-30:
        return float("nan")
    omega_hat_c = omega_c / mag_c

    cos_angles = []
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            for dk in range(-radius, radius+1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni, nj, nk = ix+di, iy+dj, iz+dk
                if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                    omega_n = omega[:, ni, nj, nk]
                    mag_n = np.linalg.norm(omega_n)
                    if mag_n > 1e-30:
                        cos_a = abs(np.dot(omega_hat_c, omega_n / mag_n))
                        cos_angles.append(cos_a)

    if len(cos_angles) == 0:
        return float("nan")
    return float(np.mean(cos_angles))


def analyze_at_location(u, omega, dx, dy, dz, ix, iy, iz):
    """Analyze a specific grid location. Returns dict."""
    # Vorticity magnitude at this point
    omega_vec = omega[:, ix, iy, iz]
    omega_mag = float(np.linalg.norm(omega_vec))

    # Strain at this point
    S = compute_strain_at_point(u, dx, dy, dz, ix, iy, iz)
    evals, evecs = np.linalg.eigh(S)
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]

    # Effective stretching
    omega_S_omega = float(omega_vec @ S @ omega_vec)
    eff_stretch = omega_S_omega / omega_mag**2 if omega_mag > 1e-30 else 0.0

    # Strain magnitude
    strain_mag = float(np.sqrt(np.sum(S**2)))

    # Local coherence
    coherence = local_coherence(omega, ix, iy, iz, radius=2)

    # Trace (incompressibility check)
    trace = float(np.trace(S))

    return dict(
        omega_mag=omega_mag,
        lambda_1=float(evals[0]),
        lambda_2=float(evals[1]),
        lambda_3=float(evals[2]),
        eff_stretch=eff_stretch,
        strain_mag=strain_mag,
        local_coherence=coherence,
        trace=trace,
    )


# ======================================================================
# MAIN
# ======================================================================
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
    log("INTERMITTENCY FIXED LOCATION TRACKING V1 — Parallel life stories")
    log(f"Dataset: JHTDB channel | Grid: N={N} | Timesteps: {len(TIMES)}")
    log("=" * 90)
    log()
    log("TRACKED LOCATIONS:")
    for label, loc in LOCATIONS.items():
        log(f"  {label}: grid ({loc[0]:>2},{loc[1]:>2},{loc[2]:>2})")
    log()
    log(f"Time points: {TIMES}")
    log()

    # Storage: results[label][time_idx] = dict
    results = {label: [] for label in LOCATIONS}
    rms_list = []

    t_start = time.time()

    for ti, t_val in enumerate(TIMES):
        log(f"--- t = {t_val:.2f} ---")
        t0 = time.time()
        try:
            u, cx, cy, cz, dx, dy, dz = fetch_cutout(N, t_val, args.token, args.chunk)
            omega = compute_vorticity(u, dx, dy, dz)
            omega_mag_field = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            rms = float(np.sqrt(np.mean(omega_mag_field**2)))
            rms_list.append(rms)

            line_parts = [f"  {time.time()-t0:.1f}s | rms={rms:.4f} |"]
            for label, loc in LOCATIONS.items():
                ix, iy, iz = loc
                r = analyze_at_location(u, omega, dx, dy, dz, ix, iy, iz)
                r["time"] = t_val
                r["fetch_ok"] = True
                results[label].append(r)
                line_parts.append(f" {label}={r['omega_mag']:.2f}")
            log(" ".join(line_parts))

        except Exception as e:
            log(f"  ERROR: {str(e)[:100]}")
            rms_list.append(float("nan"))
            for label in LOCATIONS:
                results[label].append(dict(time=t_val, fetch_ok=False))
        log()

    total_time = time.time() - t_start
    log(f"Total fetch+compute time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log()

    # ==================================================================
    # TABLE: |omega| at each location over time
    # ==================================================================
    log("=" * 90)
    log("TABLE 1: |omega| at each fixed location over time")
    log("=" * 90)
    log()
    header = f"{'t':>5} {'rms':>6}  {'A(6,31,17)':>10} {'B(28,0,16)':>10} {'C(13,31,27)':>11} {'D(0,0,12)':>10} {'E(29,0,16)':>10}"
    log(header)
    log("-" * len(header))
    for ti, t_val in enumerate(TIMES):
        parts = [f"{t_val:>5.2f} {rms_list[ti]:>6.4f}"]
        for label in ["A", "B", "C", "D", "E"]:
            r = results[label][ti]
            if r.get("fetch_ok", False):
                parts.append(f"{r['omega_mag']:>10.3f}")
            else:
                parts.append(f"{'ERROR':>10}")
        log("  ".join(parts))
    log()

    # ==================================================================
    # TABLE 2: Effective stretching at each location over time
    # ==================================================================
    log("=" * 90)
    log("TABLE 2: Effective stretching rate at each fixed location over time")
    log("=" * 90)
    log()
    header2 = f"{'t':>5}  {'A(6,31,17)':>10} {'B(28,0,16)':>10} {'C(13,31,27)':>11} {'D(0,0,12)':>10} {'E(29,0,16)':>10}"
    log(header2)
    log("-" * len(header2))
    for ti, t_val in enumerate(TIMES):
        parts = [f"{t_val:>5.2f}"]
        for label in ["A", "B", "C", "D", "E"]:
            r = results[label][ti]
            if r.get("fetch_ok", False):
                parts.append(f"{r['eff_stretch']:>+10.4f}")
            else:
                parts.append(f"{'ERROR':>10}")
        log("  ".join(parts))
    log()

    # ==================================================================
    # TABLE 3: Local coherence at each location over time
    # ==================================================================
    log("=" * 90)
    log("TABLE 3: Local coherence at each fixed location over time")
    log("=" * 90)
    log()
    header3 = f"{'t':>5}  {'A(6,31,17)':>10} {'B(28,0,16)':>10} {'C(13,31,27)':>11} {'D(0,0,12)':>10} {'E(29,0,16)':>10}"
    log(header3)
    log("-" * len(header3))
    for ti, t_val in enumerate(TIMES):
        parts = [f"{t_val:>5.2f}"]
        for label in ["A", "B", "C", "D", "E"]:
            r = results[label][ti]
            if r.get("fetch_ok", False):
                parts.append(f"{r['local_coherence']:>10.4f}")
            else:
                parts.append(f"{'ERROR':>10}")
        log("  ".join(parts))
    log()

    # ==================================================================
    # TABLE 4: lambda_3 (compression) at each location over time
    # ==================================================================
    log("=" * 90)
    log("TABLE 4: lambda_3 (compression) at each fixed location over time")
    log("=" * 90)
    log()
    header4 = f"{'t':>5}  {'A(6,31,17)':>10} {'B(28,0,16)':>10} {'C(13,31,27)':>11} {'D(0,0,12)':>10} {'E(29,0,16)':>10}"
    log(header4)
    log("-" * len(header4))
    for ti, t_val in enumerate(TIMES):
        parts = [f"{t_val:>5.2f}"]
        for label in ["A", "B", "C", "D", "E"]:
            r = results[label][ti]
            if r.get("fetch_ok", False):
                parts.append(f"{r['lambda_3']:>+10.4f}")
            else:
                parts.append(f"{'ERROR':>10}")
        log("  ".join(parts))
    log()

    # ==================================================================
    # DIAL LEDGER
    # ==================================================================
    log("=" * 90)
    log("DIAL LEDGER")
    log("=" * 90)
    log("  N=32 : JHTDB demo token cap (4096 pts/call). Not a choice.")
    log("  Cutout center: domain midpoint (0.125 half-width). Geometric, not tuned.")
    log("  Coherence radius=2 : smallest meaningful neighborhood (5^3-1=124 pts).")
    log("  Tracked locations: chosen from V3 top-3 appearances. Not tuned —")
    log("    they are the structures that showed up in the competition.")
    log("  Time window [9.90, 10.10] step 0.01 : same as V3.")
    log("  No free parameters. No fits. No thresholds.")
    log()

    save()


if __name__ == "__main__":
    main()
