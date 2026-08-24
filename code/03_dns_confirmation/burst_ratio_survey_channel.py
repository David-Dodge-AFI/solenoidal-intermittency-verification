#!/usr/bin/env python3
"""
INTERMITTENCY BURST RATIO SURVEY V1 — Self-similarity test
======================================================================
Project: turbulence/intermittency (Decoherence In Action)

QUESTION: Do all bursts — big and small — follow the same ratios?

We've seen the big burst at t=10.0 and micro-bursts at t=9.2, 9.4, etc.
If the mechanism is self-similar, the RATIOS should be constant
regardless of burst size:
  - Dominance ratio at peak (#1/#2)
  - Rise time (baseline to peak)
  - Decay time (peak to baseline)
  - Decay/Rise ratio
  - Successor ratio (next peak / this peak)

METHOD:
  1. Sample the full channel flow (t=0 to 25) at dt=0.2 (126 timesteps)
  2. At each timestep, record: crest, |omega|_max, #2 peak, dominance
  3. Identify ALL burst events (local maxima in crest that exceed
     mean + 1 sigma)
  4. For each burst: measure rise time, decay time, peak dominance,
     predecessor/successor ratios
  5. Report the statistics: are the ratios constant across burst sizes?

WHAT WE'RE LOOKING FOR (stated honestly):
  If ratios are constant across sizes → mechanism is scale-invariant
  If ratios scale with size → there's a size-dependent effect
  If no pattern → the events are more chaotic than structured

DATA SOURCE: JHTDB channel flow (full t=0 to 25)
GRID: N=32 (demo token cap)
NO DIALS except the burst detection threshold (mean + 1σ — flagged).

SETUP:
  pip install givernylocal numpy
  python intermittency_burst_ratio_survey_v1.py --token <your_token>

OUTPUT: intermittency_burst_ratio_survey_v1_RESULT.txt (same folder)

NOTE: 126 timesteps × ~15s each ≈ ~30 minutes. Long run.
"""

import numpy as np
import os, sys, argparse, time

# ======================================================================
# CONFIG
# ======================================================================
DATASET = "channel"
BOX = (0.0, 8*np.pi, -1.0, 1.0, 0.0, 3*np.pi)

# Full channel flow, dt=0.2
TIMES = [round(i*0.2, 1) for i in range(126)]  # 0.0, 0.2, 0.4, ..., 25.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "intermittency_burst_ratio_survey_v1_RESULT.txt")

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


def top_n_peaks(omega_mag, n=2):
    """Find the top N peak locations and magnitudes."""
    flat = omega_mag.ravel().copy()
    peaks = []
    N_grid = omega_mag.shape[0]
    for _ in range(n):
        idx = np.argmax(flat)
        mag = float(flat[idx])
        ix, iy, iz = np.unravel_index(idx, omega_mag.shape)
        peaks.append((mag, int(ix), int(iy), int(iz)))
        # Zero out neighborhood
        for di in range(-2, 3):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    ni, nj, nk = ix+di, iy+dj, iz+dk
                    if 0 <= ni < N_grid and 0 <= nj < N_grid and 0 <= nk < N_grid:
                        flat[ni*N_grid*N_grid + nj*N_grid + nk] = 0.0
    return peaks


def analyze_timestep_light(u, dx, dy, dz):
    """Light analysis: crest, peak, rms, top-2 peaks, dominance."""
    omega = compute_vorticity(u, dx, dy, dz)
    omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)

    peak_omega = float(np.max(omega_mag))
    rms_omega = float(np.sqrt(np.mean(omega_mag**2)))
    crest = peak_omega / rms_omega if rms_omega > 1e-30 else float("nan")

    top2 = top_n_peaks(omega_mag, n=2)
    dominance = top2[0][0] / top2[1][0] if top2[1][0] > 1e-30 else float("nan")

    return dict(
        crest=crest,
        peak_omega=peak_omega,
        rms_omega=rms_omega,
        peak1_mag=top2[0][0],
        peak1_loc=(top2[0][1], top2[0][2], top2[0][3]),
        peak2_mag=top2[1][0],
        peak2_loc=(top2[1][1], top2[1][2], top2[1][3]),
        dominance=dominance,
    )


# ======================================================================
# BURST DETECTION
# ======================================================================
def find_bursts(times, crests, threshold_sigma=1.0):
    """Find local maxima in crest that exceed mean + threshold_sigma * std.
    
    DIAL: threshold_sigma = 1.0 (flagged in ledger). This defines what
    counts as a "burst." Lower = more events, higher = fewer events.
    The choice of 1σ is standard but IS a choice.
    
    Returns list of dicts with burst properties.
    """
    crests = np.array(crests)
    mean_c = np.mean(crests)
    std_c = np.std(crests)
    threshold = mean_c + threshold_sigma * std_c
    
    bursts = []
    for i in range(1, len(crests) - 1):
        # Local maximum
        if crests[i] > crests[i-1] and crests[i] > crests[i+1]:
            if crests[i] > threshold:
                # Find rise: go back until we find a local minimum
                rise_start = i
                for j in range(i-1, -1, -1):
                    if j == 0 or crests[j] <= crests[j-1]:
                        rise_start = j
                        break
                
                # Find decay: go forward until we find a local minimum
                decay_end = i
                for j in range(i+1, len(crests)):
                    if j == len(crests)-1 or crests[j] <= crests[j+1]:
                        decay_end = j
                        break
                
                rise_time = times[i] - times[rise_start]
                decay_time = times[decay_end] - times[i]
                rise_amplitude = crests[i] - crests[rise_start]
                decay_amplitude = crests[i] - crests[decay_end]
                
                bursts.append(dict(
                    time=times[i],
                    index=i,
                    crest=float(crests[i]),
                    rise_start_time=times[rise_start],
                    rise_time=rise_time,
                    decay_end_time=times[decay_end],
                    decay_time=decay_time,
                    rise_amplitude=float(rise_amplitude),
                    decay_amplitude=float(decay_amplitude),
                    baseline_before=float(crests[rise_start]),
                    baseline_after=float(crests[decay_end]),
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
    log("INTERMITTENCY BURST RATIO SURVEY V1 — Self-similarity test")
    log(f"Dataset: JHTDB channel flow (full t=0 to 25) | Grid: N={N} | dt=0.2 | Steps: {len(TIMES)}")
    log("=" * 90)
    log()

    all_results = []
    t_start = time.time()

    for ti, t_val in enumerate(TIMES):
        t0 = time.time()
        try:
            u, cx, cy, cz, dx, dy, dz = fetch_cutout(N, t_val, args.token, args.chunk)
            r = analyze_timestep_light(u, dx, dy, dz)
            r["time"] = t_val
            r["fetch_ok"] = True
            all_results.append(r)
            if ti % 10 == 0:
                log(f"  [{ti+1}/{len(TIMES)}] t={t_val:.1f} crest={r['crest']:.3f} "
                    f"dom={r['dominance']:.3f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            log(f"  [{ti+1}/{len(TIMES)}] t={t_val:.1f} ERROR: {str(e)[:80]}")
            all_results.append(dict(time=t_val, fetch_ok=False))

    total_time = time.time() - t_start
    log()
    log(f"Total fetch+compute time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log()

    # ==================================================================
    # TIME SERIES
    # ==================================================================
    ok = [r for r in all_results if r.get("fetch_ok", False)]
    times_ok = [r["time"] for r in ok]
    crests_ok = [r["crest"] for r in ok]
    doms_ok = [r["dominance"] for r in ok]

    log("=" * 90)
    log("FULL TIME-SERIES (every 10th point shown)")
    log("=" * 90)
    log()
    log(f"{'t':>5} {'crest':>7} {'|w|max':>7} {'rms':>6} {'dom':>5} {'#1_loc':>12} {'#2_mag':>7}")
    log("-" * 60)
    for i, r in enumerate(ok):
        if i % 5 == 0:
            log(f"{r['time']:>5.1f} {r['crest']:>7.3f} {r['peak_omega']:>7.3f} "
                f"{r['rms_omega']:>6.4f} {r['dominance']:>5.3f} "
                f"({r['peak1_loc'][0]:>2},{r['peak1_loc'][1]:>2},{r['peak1_loc'][2]:>2}) "
                f"{r['peak2_mag']:>7.3f}")
    log()

    # ==================================================================
    # BURST DETECTION
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
        log(f"{'#':>3} {'t_peak':>7} {'crest':>7} {'rise_t':>7} {'decay_t':>7} "
            f"{'rise_amp':>8} {'decay_amp':>8} {'decay/rise':>10} {'base_bef':>8} {'base_aft':>8}")
        log("-" * 85)
        for i, b in enumerate(bursts):
            dr_ratio = b["decay_time"] / b["rise_time"] if b["rise_time"] > 0 else float("nan")
            log(f"{i+1:>3} {b['time']:>7.1f} {b['crest']:>7.3f} {b['rise_time']:>7.1f} "
                f"{b['decay_time']:>7.1f} {b['rise_amplitude']:>8.3f} "
                f"{b['decay_amplitude']:>8.3f} {dr_ratio:>10.3f} "
                f"{b['baseline_before']:>8.3f} {b['baseline_after']:>8.3f}")
        log()

        # Dominance at each burst peak
        log("DOMINANCE AT BURST PEAKS:")
        log(f"{'#':>3} {'t_peak':>7} {'crest':>7} {'dominance':>10}")
        log("-" * 35)
        for i, b in enumerate(bursts):
            # Find the dominance at the burst time
            for r in ok:
                if abs(r["time"] - b["time"]) < 0.01:
                    log(f"{i+1:>3} {b['time']:>7.1f} {b['crest']:>7.3f} {r['dominance']:>10.3f}")
                    break
        log()

        # ==================================================================
        # RATIO ANALYSIS
        # ==================================================================
        log("=" * 90)
        log("RATIO ANALYSIS — Are ratios constant across burst sizes?")
        log("=" * 90)
        log()

        rise_times = [b["rise_time"] for b in bursts if b["rise_time"] > 0]
        decay_times = [b["decay_time"] for b in bursts if b["decay_time"] > 0]
        dr_ratios = [b["decay_time"]/b["rise_time"] for b in bursts
                     if b["rise_time"] > 0 and b["decay_time"] > 0]
        rise_amps = [b["rise_amplitude"] for b in bursts]
        decay_amps = [b["decay_amplitude"] for b in bursts]
        amp_ratios = [b["decay_amplitude"]/b["rise_amplitude"] for b in bursts
                      if b["rise_amplitude"] > 0]
        crests_burst = [b["crest"] for b in bursts]

        log(f"  Rise times:  mean={np.mean(rise_times):.2f} std={np.std(rise_times):.2f} "
            f"range=[{min(rise_times):.1f}, {max(rise_times):.1f}]")
        log(f"  Decay times: mean={np.mean(decay_times):.2f} std={np.std(decay_times):.2f} "
            f"range=[{min(decay_times):.1f}, {max(decay_times):.1f}]")
        if dr_ratios:
            log(f"  Decay/Rise:  mean={np.mean(dr_ratios):.3f} std={np.std(dr_ratios):.3f} "
                f"range=[{min(dr_ratios):.3f}, {max(dr_ratios):.3f}]")
        if amp_ratios:
            log(f"  Decay_amp/Rise_amp: mean={np.mean(amp_ratios):.3f} std={np.std(amp_ratios):.3f}")
        log()

        # Correlation: burst size vs ratios
        if len(crests_burst) >= 3 and len(dr_ratios) >= 3:
            # Trim to same length
            n_common = min(len(crests_burst), len(dr_ratios))
            corr_size_dr = np.corrcoef(crests_burst[:n_common], dr_ratios[:n_common])[0, 1]
            log(f"  Correlation (burst_size vs decay/rise): r = {corr_size_dr:+.3f}")
            if abs(corr_size_dr) < 0.3:
                log(f"    -> WEAK correlation: ratio does NOT depend on burst size")
            elif abs(corr_size_dr) > 0.7:
                log(f"    -> STRONG correlation: ratio DOES depend on burst size")
            else:
                log(f"    -> MODERATE correlation: inconclusive")
        log()

        # Successive burst ratios
        if len(bursts) >= 2:
            log("SUCCESSIVE BURST RATIOS (each burst / previous burst):")
            log(f"{'pair':>10} {'this_crest':>10} {'prev_crest':>10} {'ratio':>7}")
            log("-" * 40)
            for i in range(1, len(bursts)):
                ratio = bursts[i]["crest"] / bursts[i-1]["crest"]
                log(f"  {i}/{i-1}  {bursts[i]['crest']:>10.3f} "
                    f"{bursts[i-1]['crest']:>10.3f} {ratio:>7.3f}")
            log()

    # ==================================================================
    # DIAL LEDGER
    # ==================================================================
    log("=" * 90)
    log("DIAL LEDGER")
    log("=" * 90)
    log("  N=32 : JHTDB demo token cap (4096 pts/call). Not a choice.")
    log("  Cutout center: domain midpoint (0.125 half-width). Geometric, not tuned.")
    log("  dt=0.2 : balances resolution vs runtime (126 steps × ~15s = ~30min).")
    log("    Finer dt would find more bursts but same ratios (if self-similar).")
    log("  Top-2 peak exclusion radius=2 : avoids double-counting same structure.")
    log("  BURST THRESHOLD = mean + 1σ : DIAL. Standard choice but IS a choice.")
    log("    Lower threshold → more bursts detected (includes smaller events).")
    log("    Higher threshold → fewer bursts (only the biggest).")
    log("    The RATIO analysis should be insensitive to this choice if the")
    log("    mechanism is truly scale-invariant.")
    log("  No fits. No tuning of the ratios themselves.")
    log()

    save()


if __name__ == "__main__":
    main()
