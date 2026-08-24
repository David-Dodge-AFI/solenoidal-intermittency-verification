#!/usr/bin/env python3
"""
INTERMITTENCY INJECTION TEST V1 — Wall (near forcing) vs Center (away)
======================================================================
Project: turbulence/intermittency (Decoherence In Action)

THE TEST: In channel flow, the walls continuously generate vorticity
(no-slip boundary). This is our "rock in the river" — a fixed, continuous
injection source. The center of the channel is "downstream" — away from
the forcing.

If the mechanism works under continuous injection:
  - Near-wall should show the SAME D/R≈1, just with different absolute magnitudes
  - Or it might show something different — let the data tell us

METHOD:
  Two cutouts at EACH timestep, same x/z center, different y:
    WALL cutout: y centered near the wall (y=0.75 to 1.0 in channel coords [-1,1])
    CENTER cutout: y centered at channel midplane (y=-0.125 to 0.125)

  Same measurements as the ratio surveys: crest, dominance, top-2 peaks.
  Same time sampling as channel ratio survey: dt=0.5, t=0 to 25 (51 steps).
  Compare the statistics between the two positions.

DATA SOURCE: JHTDB channel flow
  Box: x=[0, 8π], y=[-1, 1], z=[0, 3π]
  Wall at y=+1 and y=-1
GRID: N=32 per cutout (demo token cap)
NO DIALS. No fit. Raw numbers.

SETUP:
  pip install givernylocal numpy
  python intermittency_injection_wall_vs_center_v1.py --token <your_token>

OUTPUT: intermittency_injection_wall_vs_center_v1_RESULT.txt

NOTE: 51 timesteps × 2 cutouts × ~15s = ~25 minutes.
"""

import numpy as np
import os, sys, argparse, time

# ======================================================================
# CONFIG
# ======================================================================
DATASET = "channel"

# Channel box: x=[0, 8π], y=[-1, 1], z=[0, 3π]
# Two sampling positions (same x/z center, different y):
#   WALL: near y=+1 wall (y from 0.75 to 1.0)
#   CENTER: channel midplane (y from -0.125 to 0.125)

# x and z: same for both (center of domain, 0.125 half-width)
X_RANGE = (0.0, 8*np.pi)
Z_RANGE = (0.0, 3*np.pi)

# y ranges for the two cutouts
Y_WALL = (0.75, 1.0)       # near the y=+1 wall
Y_CENTER = (-0.125, 0.125)  # channel midplane

TIMES = [round(i*0.5, 1) for i in range(51)]  # 0.0, 0.5, 1.0, ..., 25.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "intermittency_injection_wall_vs_center_v1_RESULT.txt")

L = []
def log(s=""): print(s); L.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"\n[written] {OUT}")


# ======================================================================
# DATA FETCH
# ======================================================================
def fetch_cutout_region(N, time_val, y_range, token, chunk=4096):
    """Fetch N^3 velocity cutout with specified y range."""
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData

    # x: center of domain, 0.125 half-width
    x0, x1 = X_RANGE
    xc = 0.5*(x0+x1); xhalf = 0.125*(x1-x0)
    cx = np.linspace(xc-xhalf, xc+xhalf, N)

    # y: specified range
    cy = np.linspace(y_range[0], y_range[1], N)

    # z: center of domain, 0.125 half-width
    z0, z1 = Z_RANGE
    zc = 0.5*(z0+z1); zhalf = 0.125*(z1-z0)
    cz = np.linspace(zc-zhalf, zc+zhalf, N)

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
    ux, uy, uz = u[0], u[1], u[2]
    ox = np.gradient(uz, dy, axis=1) - np.gradient(uy, dz, axis=2)
    oy = np.gradient(ux, dz, axis=2) - np.gradient(uz, dx, axis=0)
    oz = np.gradient(uy, dx, axis=0) - np.gradient(ux, dy, axis=1)
    return np.array([ox, oy, oz])


def top_n_peaks(omega_mag, n=2):
    flat = omega_mag.ravel().copy()
    peaks = []
    N_grid = omega_mag.shape[0]
    for _ in range(n):
        idx = np.argmax(flat)
        mag = float(flat[idx])
        ix, iy, iz = np.unravel_index(idx, omega_mag.shape)
        peaks.append((mag, int(ix), int(iy), int(iz)))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    ni, nj, nk = ix+di, iy+dj, iz+dk
                    if 0 <= ni < N_grid and 0 <= nj < N_grid and 0 <= nk < N_grid:
                        flat[ni*N_grid*N_grid + nj*N_grid + nk] = 0.0
    return peaks


def analyze_cutout(u, dx, dy, dz):
    omega = compute_vorticity(u, dx, dy, dz)
    omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    peak_omega = float(np.max(omega_mag))
    rms_omega = float(np.sqrt(np.mean(omega_mag**2)))
    crest = peak_omega / rms_omega if rms_omega > 1e-30 else float("nan")
    top2 = top_n_peaks(omega_mag, n=2)
    dominance = top2[0][0] / top2[1][0] if top2[1][0] > 1e-30 else float("nan")
    return dict(
        crest=crest, peak_omega=peak_omega, rms_omega=rms_omega,
        dominance=dominance,
        peak1_mag=top2[0][0], peak2_mag=top2[1][0],
    )


# ======================================================================
# BURST DETECTION
# ======================================================================
def find_bursts(times, crests, threshold_sigma=1.0):
    crests = np.array(crests)
    mean_c = np.mean(crests); std_c = np.std(crests)
    threshold = mean_c + threshold_sigma * std_c
    bursts = []
    for i in range(1, len(crests) - 1):
        if crests[i] > crests[i-1] and crests[i] > crests[i+1]:
            if crests[i] > threshold:
                rise_start = i
                for j in range(i-1, -1, -1):
                    if j == 0 or crests[j] <= crests[j-1]:
                        rise_start = j; break
                decay_end = i
                for j in range(i+1, len(crests)):
                    if j == len(crests)-1 or crests[j] <= crests[j+1]:
                        decay_end = j; break
                rise_time = times[i] - times[rise_start]
                decay_time = times[decay_end] - times[i]
                bursts.append(dict(
                    time=times[i], crest=float(crests[i]),
                    rise_time=rise_time, decay_time=decay_time,
                ))
    return bursts, mean_c, std_c, threshold


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
    log("INTERMITTENCY INJECTION TEST V1 — Wall (continuous forcing) vs Center (away)")
    log(f"Dataset: JHTDB channel | Grid: N={N} per cutout | dt=0.5 | Steps: {len(TIMES)}")
    log("=" * 90)
    log()
    log(f"WALL cutout:   y = [{Y_WALL[0]}, {Y_WALL[1]}] (near y=+1 wall)")
    log(f"CENTER cutout: y = [{Y_CENTER[0]}, {Y_CENTER[1]}] (channel midplane)")
    log(f"Same x/z center for both (domain midpoint, 0.125 half-width)")
    log()

    wall_results = []
    center_results = []
    t_start = time.time()

    for ti, t_val in enumerate(TIMES):
        t0 = time.time()
        try:
            # Wall cutout
            u_w, cx, cy_w, cz, dx_w, dy_w, dz_w = fetch_cutout_region(
                N, t_val, Y_WALL, args.token, args.chunk)
            rw = analyze_cutout(u_w, dx_w, dy_w, dz_w)
            rw["time"] = t_val; rw["fetch_ok"] = True
            wall_results.append(rw)

            # Center cutout
            u_c, cx, cy_c, cz, dx_c, dy_c, dz_c = fetch_cutout_region(
                N, t_val, Y_CENTER, args.token, args.chunk)
            rc = analyze_cutout(u_c, dx_c, dy_c, dz_c)
            rc["time"] = t_val; rc["fetch_ok"] = True
            center_results.append(rc)

            if ti % 10 == 0:
                log(f"  [{ti+1}/{len(TIMES)}] t={t_val:.1f} "
                    f"WALL: crest={rw['crest']:.2f} dom={rw['dominance']:.3f} rms={rw['rms_omega']:.3f} | "
                    f"CTR: crest={rc['crest']:.2f} dom={rc['dominance']:.3f} rms={rc['rms_omega']:.3f} "
                    f"({time.time()-t0:.1f}s)")
        except Exception as e:
            log(f"  [{ti+1}/{len(TIMES)}] t={t_val:.1f} ERROR: {str(e)[:80]}")
            wall_results.append(dict(time=t_val, fetch_ok=False))
            center_results.append(dict(time=t_val, fetch_ok=False))

    total_time = time.time() - t_start
    log()
    log(f"Total fetch+compute time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log()

    # ==================================================================
    # TIME-SERIES TABLE
    # ==================================================================
    ok_w = [r for r in wall_results if r.get("fetch_ok", False)]
    ok_c = [r for r in center_results if r.get("fetch_ok", False)]

    log("=" * 90)
    log("TIME-SERIES: WALL vs CENTER")
    log("=" * 90)
    log()
    log(f"{'t':>5} {'W_crest':>8} {'W_rms':>6} {'W_dom':>6} | {'C_crest':>8} {'C_rms':>6} {'C_dom':>6}")
    log("-" * 60)
    for i in range(min(len(ok_w), len(ok_c))):
        rw = ok_w[i]; rc = ok_c[i]
        log(f"{rw['time']:>5.1f} {rw['crest']:>8.3f} {rw['rms_omega']:>6.3f} "
            f"{rw['dominance']:>6.3f} | {rc['crest']:>8.3f} {rc['rms_omega']:>6.3f} "
            f"{rc['dominance']:>6.3f}")
    log()

    # ==================================================================
    # STATISTICS COMPARISON
    # ==================================================================
    log("=" * 90)
    log("STATISTICS: WALL vs CENTER")
    log("=" * 90)
    log()

    w_crests = [r["crest"] for r in ok_w]
    c_crests = [r["crest"] for r in ok_c]
    w_doms = [r["dominance"] for r in ok_w]
    c_doms = [r["dominance"] for r in ok_c]
    w_rms = [r["rms_omega"] for r in ok_w]
    c_rms = [r["rms_omega"] for r in ok_c]

    log(f"  {'':>20} {'WALL':>12} {'CENTER':>12}")
    log(f"  {'-'*45}")
    log(f"  {'RMS |omega| mean':>20} {np.mean(w_rms):>12.4f} {np.mean(c_rms):>12.4f}")
    log(f"  {'Crest mean':>20} {np.mean(w_crests):>12.3f} {np.mean(c_crests):>12.3f}")
    log(f"  {'Crest std':>20} {np.std(w_crests):>12.3f} {np.std(c_crests):>12.3f}")
    log(f"  {'Crest max':>20} {max(w_crests):>12.3f} {max(c_crests):>12.3f}")
    log(f"  {'Crest min':>20} {min(w_crests):>12.3f} {min(c_crests):>12.3f}")
    log(f"  {'Dominance mean':>20} {np.mean(w_doms):>12.3f} {np.mean(c_doms):>12.3f}")
    log(f"  {'Dominance max':>20} {max(w_doms):>12.3f} {max(c_doms):>12.3f}")
    log(f"  {'Crest σ/mean':>20} {np.std(w_crests)/np.mean(w_crests):>12.3f} "
        f"{np.std(c_crests)/np.mean(c_crests):>12.3f}")
    log()

    # ==================================================================
    # BURST ANALYSIS — BOTH REGIONS
    # ==================================================================
    times_ok = [r["time"] for r in ok_w]

    log("=" * 90)
    log("BURST ANALYSIS — WALL")
    log("=" * 90)
    log()
    w_bursts, w_mean, w_std, w_thresh = find_bursts(times_ok, w_crests)
    log(f"  Crest: mean={w_mean:.3f} std={w_std:.3f} threshold={w_thresh:.3f}")
    log(f"  Bursts detected: {len(w_bursts)}")
    if w_bursts:
        w_dr = [b["decay_time"]/b["rise_time"] for b in w_bursts if b["rise_time"] > 0]
        if w_dr:
            log(f"  Decay/Rise: mean={np.mean(w_dr):.3f} range=[{min(w_dr):.3f}, {max(w_dr):.3f}]")
    log()

    log("=" * 90)
    log("BURST ANALYSIS — CENTER")
    log("=" * 90)
    log()
    c_bursts, c_mean, c_std, c_thresh = find_bursts(times_ok, c_crests)
    log(f"  Crest: mean={c_mean:.3f} std={c_std:.3f} threshold={c_thresh:.3f}")
    log(f"  Bursts detected: {len(c_bursts)}")
    if c_bursts:
        c_dr = [b["decay_time"]/b["rise_time"] for b in c_bursts if b["rise_time"] > 0]
        if c_dr:
            log(f"  Decay/Rise: mean={np.mean(c_dr):.3f} range=[{min(c_dr):.3f}, {max(c_dr):.3f}]")
    log()

    # ==================================================================
    # CROSS-REGION COMPARISON
    # ==================================================================
    log("=" * 90)
    log("CROSS-REGION: Does D/R≈1 hold near the forcing source?")
    log("=" * 90)
    log()
    log("  Previous results (channel midpoint, full survey):")
    log("    Decay/Rise: mean=1.135")
    log()
    log("  This test:")
    if w_bursts and w_dr:
        log(f"    WALL  Decay/Rise: mean={np.mean(w_dr):.3f}")
    else:
        log(f"    WALL: insufficient bursts ({len(w_bursts)})")
    if c_bursts and c_dr:
        log(f"    CENTER Decay/Rise: mean={np.mean(c_dr):.3f}")
    else:
        log(f"    CENTER: insufficient bursts ({len(c_bursts)})")
    log()

    # ==================================================================
    log("=" * 90)
    log("DIAL LEDGER")
    log("=" * 90)
    log("  N=32 per cutout : JHTDB demo token cap. Not a choice.")
    log("  x/z: domain midpoint, 0.125 half-width. Geometric.")
    log("  WALL y=[0.75, 1.0]: near the y=+1 wall (forcing source).")
    log("    NOTE: not AT the wall (y=1.0 is the wall itself — velocity=0 there).")
    log("    This samples the viscous/buffer layer where wall-generated vorticity lives.")
    log("  CENTER y=[-0.125, 0.125]: channel midplane (away from forcing).")
    log("  dt=0.5 : 51 steps, full t=0–25. Matches channel survey resolution.")
    log("  BURST THRESHOLD = mean + 1σ per region (same as all surveys).")
    log("  No fits. No tuning.")
    log()

    save()


if __name__ == "__main__":
    main()
