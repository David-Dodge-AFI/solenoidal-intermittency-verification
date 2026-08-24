#!/usr/bin/env python3
"""
INTERMITTENCY BURST SNAPSHOT V1 — Decoherence at burst vs calm
======================================================================
Project: turbulence/intermittency (Decoherence In Action)

QUESTION: At the burst peak (channel t=10.0, crest=12.3), what does the
local decoherence geometry look like compared to calm (t=9.286, crest=6.3)?

MEASURES at the grid point of max |omega| for each timestep:
  1. Strain eigenvalues (lambda_1 >= lambda_2 >= lambda_3)
     - Sorted descending. Incompressibility: lambda_1+lambda_2+lambda_3 = 0.
  2. Vorticity-strain alignment: cos(theta) = omega_hat . e_i for all 3 eigenvectors
     - Classical result: vorticity preferentially aligns with INTERMEDIATE eigenvector (e_2)
     - Our mechanism: alignment with e_1 (stretching direction) is what FEEDS the burst;
       decoherence acts to BREAK that alignment.
  3. Local strain-vorticity angle statistics in a neighborhood around max-|omega|

DATA SOURCE: JHTDB channel flow (external DNS, Re_tau=1000)
GRID: N=32 cutout (demo token 4096-point cap)
NO DIALS. Raw numbers only.

SETUP:
  pip install givernylocal numpy
  python intermittency_burst_snapshot_v1.py --token <your_token>
  (demo token: edu.jhu.pha.turbulence.testing-201406)

OUTPUT: intermittency_burst_snapshot_v1_RESULT.txt (same folder)
"""

import numpy as np
import os, sys, argparse, time

# ======================================================================
# JHTDB CONFIG — channel flow
# ======================================================================
DATASET = "channel"
BOX = (0.0, 8*np.pi, -1.0, 1.0, 0.0, 3*np.pi)  # (x0, x1, y0, y1, z0, z1)

# Target timesteps
T_BURST = 10.0    # crest = 12.3 (burst peak)
T_CALM  = 9.286   # crest = 6.3  (one sample before burst — calm baseline)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "intermittency_burst_snapshot_v1_RESULT.txt")

L = []
def log(s=""): print(s); L.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"\n[written] {OUT}")


# ======================================================================
# DATA FETCH — adapted from jhtdb_crossystem_envelope.py
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

    # u shape: (3, N, N, N)
    u = vel.reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    return u, cx, cy, cz, dx, dy, dz


# ======================================================================
# COMPUTE — vorticity, strain, alignment
# ======================================================================
def compute_vorticity(u, dx, dy, dz):
    """Compute vorticity omega = curl(u) via central differences. Returns (3, N, N, N)."""
    ux, uy, uz = u[0], u[1], u[2]
    # omega_x = du_z/dy - du_y/dz
    ox = np.gradient(uz, dy, axis=1) - np.gradient(uy, dz, axis=2)
    # omega_y = du_x/dz - du_z/dx
    oy = np.gradient(ux, dz, axis=2) - np.gradient(uz, dx, axis=0)
    # omega_z = du_y/dx - du_x/dy
    oz = np.gradient(uy, dx, axis=0) - np.gradient(ux, dy, axis=1)
    return np.array([ox, oy, oz])


def compute_strain_tensor(u, dx, dy, dz):
    """Compute strain rate tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i).
    Returns (N, N, N, 3, 3)."""
    N = u.shape[1]
    # Velocity gradient tensor A_ij = du_i / dx_j
    # u shape: (3, N, N, N); axes 0=component, 1=x, 2=y, 3=z
    A = np.zeros((N, N, N, 3, 3))
    spacings = [dx, dy, dz]
    for i in range(3):  # velocity component
        for j in range(3):  # spatial direction
            A[:, :, :, i, j] = np.gradient(u[i], spacings[j], axis=j)
    # S = 0.5 * (A + A^T)
    S = 0.5 * (A + A.transpose(0, 1, 2, 4, 3))
    return S


def analyze_point(omega_vec, S_mat):
    """At a single grid point, compute strain eigenvalues + vorticity-strain alignment.

    Args:
        omega_vec: (3,) vorticity vector at this point
        S_mat: (3,3) strain rate tensor at this point

    Returns dict with:
        eigenvalues: (3,) sorted descending (lambda_1 >= lambda_2 >= lambda_3)
        eigenvectors: (3,3) columns are eigenvectors, sorted same as eigenvalues
        omega_mag: |omega|
        cos_with_e1: omega_hat . e_1 (stretching direction)
        cos_with_e2: omega_hat . e_2 (intermediate — classical alignment)
        cos_with_e3: omega_hat . e_3 (compression direction)
        strain_mag: |S| = sqrt(S_ij S_ij)
        omega_S_omega: omega . S . omega (vortex stretching term)
    """
    # Eigendecomposition of symmetric S
    evals, evecs = np.linalg.eigh(S_mat)
    # eigh returns ascending; flip to descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    omega_mag = np.linalg.norm(omega_vec)
    omega_hat = omega_vec / omega_mag if omega_mag > 1e-30 else np.zeros(3)

    cos_e1 = abs(float(np.dot(omega_hat, evecs[:, 0])))
    cos_e2 = abs(float(np.dot(omega_hat, evecs[:, 1])))
    cos_e3 = abs(float(np.dot(omega_hat, evecs[:, 2])))

    strain_mag = np.sqrt(np.sum(S_mat**2))

    # Vortex stretching: omega_i S_ij omega_j
    omega_S_omega = float(omega_vec @ S_mat @ omega_vec)

    return dict(
        eigenvalues=evals,
        eigenvectors=evecs,
        omega_mag=float(omega_mag),
        cos_with_e1=cos_e1,
        cos_with_e2=cos_e2,
        cos_with_e3=cos_e3,
        strain_mag=float(strain_mag),
        omega_S_omega=omega_S_omega,
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
    log("=" * 70)
    log("INTERMITTENCY BURST SNAPSHOT V1")
    log("Decoherence geometry: burst (t=10.0) vs calm (t=9.286)")
    log("Dataset: JHTDB channel flow | Grid: N=%d" % N)
    log("=" * 70)
    log()

    results = {}
    for label, t_val in [("CALM (t=9.286)", T_CALM), ("BURST (t=10.0)", T_BURST)]:
        log("-" * 70)
        log(f"FETCHING: {label}")
        log("-" * 70)
        t0 = time.time()
        u, cx, cy, cz, dx, dy, dz = fetch_cutout(N, t_val, args.token, args.chunk)
        fetch_time = time.time() - t0
        log(f"  Fetch time: {fetch_time:.1f}s")

        # Compute vorticity
        omega = compute_vorticity(u, dx, dy, dz)
        omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)

        # Find max-|omega| location
        idx_flat = np.argmax(omega_mag)
        ix, iy, iz = np.unravel_index(idx_flat, omega_mag.shape)
        peak_omega = float(omega_mag[ix, iy, iz])
        rms_omega = float(np.sqrt(np.mean(omega_mag**2)))
        crest = peak_omega / rms_omega if rms_omega > 1e-30 else float("nan")

        log(f"  |omega|_max = {peak_omega:.4f} at grid ({ix},{iy},{iz})")
        log(f"  |omega|_rms = {rms_omega:.4f}")
        log(f"  Crest factor = {crest:.3f}")
        log(f"  Physical location: x={cx[ix]:.4f}, y={cy[iy]:.4f}, z={cz[iz]:.4f}")
        log()

        # Compute strain tensor
        S = compute_strain_tensor(u, dx, dy, dz)

        # Analyze at max-omega point
        omega_at_peak = omega[:, ix, iy, iz]
        S_at_peak = S[ix, iy, iz, :, :]

        r = analyze_point(omega_at_peak, S_at_peak)
        results[label] = r
        results[label]["crest"] = crest
        results[label]["peak_omega"] = peak_omega
        results[label]["rms_omega"] = rms_omega

        log(f"  STRAIN EIGENVALUES (sorted descending):")
        log(f"    lambda_1 = {r['eigenvalues'][0]:+.6f}  (stretching)")
        log(f"    lambda_2 = {r['eigenvalues'][1]:+.6f}  (intermediate)")
        log(f"    lambda_3 = {r['eigenvalues'][2]:+.6f}  (compression)")
        log(f"    sum = {sum(r['eigenvalues']):+.6f}  (should be ~0, incompressibility)")
        log()
        log(f"  VORTICITY-STRAIN ALIGNMENT (|cos(theta)|):")
        log(f"    |omega_hat . e_1| = {r['cos_with_e1']:.4f}  (stretching dir)")
        log(f"    |omega_hat . e_2| = {r['cos_with_e2']:.4f}  (intermediate dir)")
        log(f"    |omega_hat . e_3| = {r['cos_with_e3']:.4f}  (compression dir)")
        log()
        log(f"  STRETCHING TERM:")
        log(f"    omega . S . omega = {r['omega_S_omega']:+.6f}")
        log(f"    |S| = {r['strain_mag']:.6f}")
        log(f"    omega . S . omega / |omega|^2 = {r['omega_S_omega']/peak_omega**2:+.6f}"
            f"  (effective stretching rate)")
        log()

    # ================================================================
    # SIDE-BY-SIDE COMPARISON
    # ================================================================
    log("=" * 70)
    log("SIDE-BY-SIDE COMPARISON")
    log("=" * 70)
    log()

    calm = results["CALM (t=9.286)"]
    burst = results["BURST (t=10.0)"]

    log(f"{'Quantity':<40} {'CALM (t=9.286)':>15} {'BURST (t=10.0)':>15}")
    log("-" * 70)
    log(f"{'Crest factor':<40} {calm['crest']:>15.3f} {burst['crest']:>15.3f}")
    log(f"{'|omega|_max':<40} {calm['peak_omega']:>15.4f} {burst['peak_omega']:>15.4f}")
    log(f"{'|omega|_rms':<40} {calm['rms_omega']:>15.4f} {burst['rms_omega']:>15.4f}")
    log(f"{'|S| at max-omega':<40} {calm['strain_mag']:>15.4f} {burst['strain_mag']:>15.4f}")
    log(f"{'lambda_1 (stretching)':<40} {calm['eigenvalues'][0]:>+15.6f} {burst['eigenvalues'][0]:>+15.6f}")
    log(f"{'lambda_2 (intermediate)':<40} {calm['eigenvalues'][1]:>+15.6f} {burst['eigenvalues'][1]:>+15.6f}")
    log(f"{'lambda_3 (compression)':<40} {calm['eigenvalues'][2]:>+15.6f} {burst['eigenvalues'][2]:>+15.6f}")
    log(f"{'|cos| with e_1 (stretching)':<40} {calm['cos_with_e1']:>15.4f} {burst['cos_with_e1']:>15.4f}")
    log(f"{'|cos| with e_2 (intermediate)':<40} {calm['cos_with_e2']:>15.4f} {burst['cos_with_e2']:>15.4f}")
    log(f"{'|cos| with e_3 (compression)':<40} {calm['cos_with_e3']:>15.4f} {burst['cos_with_e3']:>15.4f}")
    log(f"{'omega.S.omega (stretching term)':<40} {calm['omega_S_omega']:>+15.6f} {burst['omega_S_omega']:>+15.6f}")
    log(f"{'omega.S.omega / |omega|^2 (eff. rate)':<40} "
        f"{calm['omega_S_omega']/calm['peak_omega']**2:>+15.6f} "
        f"{burst['omega_S_omega']/burst['peak_omega']**2:>+15.6f}")
    log()

    # ================================================================
    # RAW INTERPRETATION GUIDE (not interpretation — what each number MEANS)
    # ================================================================
    log("=" * 70)
    log("WHAT TO LOOK FOR (not interpretation — what the numbers mean)")
    log("=" * 70)
    log()
    log("  OUR MECHANISM PREDICTS:")
    log("    At BURST: vorticity aligns with stretching direction (e_1) ->")
    log("      cos_with_e1 HIGH, omega.S.omega LARGE AND POSITIVE.")
    log("      This is the alignment that FEEDS the burst.")
    log()
    log("    At CALM: vorticity more aligned with intermediate (e_2) ->")
    log("      cos_with_e2 HIGH (the classical isotropic result).")
    log("      Decoherence has already broken the stretching alignment.")
    log()
    log("    THE DECOHERENCE SIGNAL:")
    log("      If burst shows HIGH e_1 alignment and calm shows it BROKEN ->")
    log("      that IS the mechanism working: alignment built (burst rose),")
    log("      then decoherence broke it (burst died).")
    log()
    log("    ALTERNATIVE (mechanism NOT supported):")
    log("      If BOTH show similar alignment patterns, the burst is not")
    log("      alignment-driven and our mechanism doesn't explain it.")
    log()
    log("  NO VERDICT ISSUED. Numbers speak for themselves.")
    log()

    log("=" * 70)
    log("DIAL LEDGER")
    log("=" * 70)
    log("  N=32 : set by JHTDB demo token cap (4096 pts/call). Not a choice.")
    log("  Cutout center: domain midpoint (0.125 half-width). Geometric, not tuned.")
    log("  No other free parameters.")
    log()

    save()


if __name__ == "__main__":
    main()
