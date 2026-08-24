#!/usr/bin/env python3
"""
INTERMITTENCY BURST RATIO SURVEY — TRANSITION BOUNDARY LAYER
======================================================================
Project: turbulence/intermittency (Decoherence In Action)

Same ratio survey on transition_bl dataset.
Boundary layer transitioning from laminar to turbulent.

Dataset: transition_bl (JHTDB)
  - Boundary layer over a flat plate
  - Active transition zone (laminar → turbulent)
  - Box: [30.2185, 1000.065] x [0, 26.488] x [0, 240.0]
  - tspan: [0, 1175.0]

Time sampling: dt=25.0 (48 timesteps across t=0 to 1175)
Grid: N=32 (demo token cap)
NO DIALS. No fit. Raw numbers.

SETUP:
  pip install givernylocal numpy
  python intermittency_burst_ratio_survey_transition_bl_v1.py --token <your_token>

OUTPUT: intermittency_burst_ratio_survey_transition_bl_v1_RESULT.txt
"""

import numpy as np
import os, sys, argparse, time

# ======================================================================
# CONFIG
# ======================================================================
DATASET = "transition_bl"
BOX = (30.2185, 1000.065, 0.0, 26.48795, 0.0, 240.0)

# Full transition_bl dataset, dt=25
TIMES = [round(i*25.0, 1) for i in range(48)]  # 0, 25, 50, ..., 1175

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "intermittency_burst_ratio_survey_transition_bl_v1_RESULT.txt")

L = []
def log(s=""): print(s); L.append(str(s))
def save():
    with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(L))
    print(f"\n[written] {OUT}")


# ======================================================================
# DATA FETCH
# ======================================================================
def fetch_cutout(N, time_val, token, chunk=4096):
    """Fetch N^3 velocity cutout, centered in domain."""
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


def analyze_timestep_light(u, dx, dy, dz):
    omega = compute_vorticity(u, dx, dy, dz)
    omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
    peak_omega = float(np.max(omega_mag))
    rms_omega = float(np.sqrt(np.mean(omega_mag**2)))
    crest = peak_omega / rms_omega if rms_omega > 1e-30 else float("nan")
    top2 = top_n_peaks(omega_mag, n=2)
    dominance = top2[0][0] / top2[1][0] if top2[1][0] > 1e-30 else float("nan")
    return dict(
        crest=crest, peak_omega=peak_omega, rms_omega=rms_omega,
        peak1_mag=top2[0][0], peak1_loc=(top2[0][1], top2[0][2], top2[0][3]),
        peak2_mag=top2[1][0], peak2_loc=(top2[1][1], top2[1][2], top2[1][3]),
        dominance=dominance,
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
                    time=times[i], index=i, crest=float(crests[i]),
                    rise_time=rise_time, decay_time=decay_time,
                    rise_amplitude=float(crests[i] - crests[rise_start]),
                    decay_amplitude=float(crests[i] - crests[decay_end]),
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
    log("INTERMITTENCY BURST RATIO SURVEY — TRANSITION BOUNDARY LAYER")
    log(f"Dataset: {DATASET} | Grid: N={N} | dt=25 | Steps: {len(TIMES)}")
    log("=" * 90)
    log()

    all_results = []
    t_start = time.time()

    for ti, t_val in enumerate(TIMES):
        t0 = time.time()
        try:
            u, cx, cy, cz, dx, dy, dz = fetch_cutout(N, t_val, args.token, args.chunk)
            r = analyze_timestep_light(u, dx, dy, dz)
            r["time"] = t_val; r["fetch_ok"] = True
            all_results.append(r)
            if ti % 10 == 0:
                log(f"  [{ti+1}/{len(TIMES)}] t={t_val:.0f} crest={r['crest']:.3f} "
                    f"dom={r['dominance']:.3f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            log(f"  [{ti+1}/{len(TIMES)}] t={t_val:.0f} ERROR: {str(e)[:80]}")
            all_results.append(dict(time=t_val, fetch_ok=False))

    total_time = time.time() - t_start
    log()
    log(f"Total fetch+compute time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log()

    # ==================================================================
    ok = [r for r in all_results if r.get("fetch_ok", False)]
    times_ok = [r["time"] for r in ok]
    crests_ok = [r["crest"] for r in ok]
    doms_ok = [r["dominance"] for r in ok]

    log("=" * 90)
    log("TIME-SERIES")
    log("=" * 90)
    log()
    log(f"{'t':>6} {'crest':>7} {'|w|max':>8} {'rms':>7} {'dom':>5}")
    log("-" * 42)
    for r in ok:
        log(f"{r['time']:>6.0f} {r['crest']:>7.3f} {r['peak_omega']:>8.4f} "
            f"{r['rms_omega']:>7.4f} {r['dominance']:>5.3f}")
    log()

    # ==================================================================
    bursts, mean_c, std_c, threshold = find_bursts(times_ok, crests_ok, threshold_sigma=1.0)

    log("=" * 90)
    log("BURST DETECTION")
    log("=" * 90)
    log()
    log(f"  Crest statistics: mean={mean_c:.3f}, std={std_c:.3f}")
    log(f"  Threshold (mean + 1σ): {threshold:.3f}")
    log(f"  Bursts detected: {len(bursts)}")
    log()

    if bursts:
        log(f"{'#':>3} {'t_peak':>7} {'crest':>7} {'rise_t':>7} {'decay_t':>7} {'decay/rise':>10}")
        log("-" * 50)
        for i, b in enumerate(bursts):
            dr = b["decay_time"] / b["rise_time"] if b["rise_time"] > 0 else float("nan")
            log(f"{i+1:>3} {b['time']:>7.0f} {b['crest']:>7.3f} "
                f"{b['rise_time']:>7.0f} {b['decay_time']:>7.0f} {dr:>10.3f}")
        log()

        # Dominance at peaks
        log("DOMINANCE AT BURST PEAKS:")
        log(f"{'#':>3} {'t_peak':>7} {'crest':>7} {'dominance':>10}")
        log("-" * 35)
        for i, b in enumerate(bursts):
            for r in ok:
                if abs(r["time"] - b["time"]) < 0.1:
                    log(f"{i+1:>3} {b['time']:>7.0f} {b['crest']:>7.3f} {r['dominance']:>10.3f}")
                    break
        log()

        # Ratio analysis
        rise_times = [b["rise_time"] for b in bursts if b["rise_time"] > 0]
        decay_times = [b["decay_time"] for b in bursts if b["decay_time"] > 0]
        dr_ratios = [b["decay_time"]/b["rise_time"] for b in bursts
                     if b["rise_time"] > 0 and b["decay_time"] > 0]
        crests_burst = [b["crest"] for b in bursts]

        log("=" * 90)
        log("RATIO ANALYSIS")
        log("=" * 90)
        log()
        if rise_times:
            log(f"  Rise times:  mean={np.mean(rise_times):.1f} std={np.std(rise_times):.1f} "
                f"range=[{min(rise_times):.0f}, {max(rise_times):.0f}]")
        if decay_times:
            log(f"  Decay times: mean={np.mean(decay_times):.1f} std={np.std(decay_times):.1f} "
                f"range=[{min(decay_times):.0f}, {max(decay_times):.0f}]")
        if dr_ratios:
            log(f"  Decay/Rise:  mean={np.mean(dr_ratios):.3f} std={np.std(dr_ratios):.3f} "
                f"range=[{min(dr_ratios):.3f}, {max(dr_ratios):.3f}]")
        if len(crests_burst) >= 3 and len(dr_ratios) >= 3:
            n_c = min(len(crests_burst), len(dr_ratios))
            corr = np.corrcoef(crests_burst[:n_c], dr_ratios[:n_c])[0, 1]
            log(f"  Correlation (burst_size vs decay/rise): r = {corr:+.3f}")
            if abs(corr) < 0.3:
                log(f"    -> WEAK: ratio does NOT depend on burst size")
            elif abs(corr) > 0.7:
                log(f"    -> STRONG: ratio DOES depend on burst size")
            else:
                log(f"    -> MODERATE: inconclusive")
        log()

        # Dominance stats
        burst_doms = []
        for b in bursts:
            for r in ok:
                if abs(r["time"] - b["time"]) < 0.1:
                    burst_doms.append(r["dominance"]); break
        if burst_doms:
            log(f"  Dominance at peaks: mean={np.mean(burst_doms):.3f} "
                f"range=[{min(burst_doms):.3f}, {max(burst_doms):.3f}]")
        log()

    # ==================================================================
    # COMPARISON TO OTHER SYSTEMS
    # ==================================================================
    log("=" * 90)
    log("CROSS-SYSTEM COMPARISON")
    log("=" * 90)
    log()
    log("  Channel flow:")
    log("    Decay/Rise: mean=1.135, range=[0.25, 2.0]")
    log("    Dominance at peaks: mean=1.34, range=[1.03, 1.88]")
    log("    Correlation (size vs D/R): r=+0.026")
    log("    Crest: mean=7.53, std=1.17")
    log()
    log("  Isotropic:")
    log("    Decay/Rise: mean=0.944, range=[0.667, 1.500]")
    log("    Dominance at peaks: mean=1.27, range=[1.17, 1.34]")
    log("    Correlation (size vs D/R): r=-0.830 (n=3, unreliable)")
    log("    Crest: mean=5.20, std=0.60")
    log()
    log("  Transition BL (this run):")
    if dr_ratios:
        log(f"    Decay/Rise: mean={np.mean(dr_ratios):.3f}, "
            f"range=[{min(dr_ratios):.3f}, {max(dr_ratios):.3f}]")
    if burst_doms:
        log(f"    Dominance at peaks: mean={np.mean(burst_doms):.3f}, "
            f"range=[{min(burst_doms):.3f}, {max(burst_doms):.3f}]")
    if len(crests_burst) >= 3 and len(dr_ratios) >= 3:
        log(f"    Correlation (size vs D/R): r={corr:+.3f}")
    log(f"    Crest: mean={mean_c:.3f}, std={std_c:.3f}")
    log()

    # ==================================================================
    log("=" * 90)
    log("DIAL LEDGER")
    log("=" * 90)
    log("  N=32 : JHTDB demo token cap. Not a choice.")
    log("  Cutout center: domain midpoint, 0.125 half-width. Geometric.")
    log("  dt=25 : 48 steps across t=[0,1175]. Matches dataset timescale.")
    log("  Top-2 exclusion radius=2.")
    log("  BURST THRESHOLD = mean + 1σ : DIAL (same as other surveys).")
    log("  No fits. No tuning.")
    log()

    save()


if __name__ == "__main__":
    main()
