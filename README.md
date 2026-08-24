# Solenoidal Projector Geometry as the Origin of Turbulence Intermittency

## A Zero-Parameter Derivation of Anomalous Scaling Exponents

**Author:** Dave Dodge  
**Date:** August 2026  
**Companion to:** "Global Regularity of 3D Navier-Stokes via Scale-Dependent Geometric Decoherence" (Paper 1)  

### Paper 1 (Regularity Proof) — Links

| Document | DOI |
|----------|-----|
| Solenoidal Closure Theorem | [10.5281/zenodo.21515265](https://doi.org/10.5281/zenodo.21515265) |
| NS Regularity Manuscript | [10.5281/zenodo.21515457](https://doi.org/10.5281/zenodo.21515457) |
| Adversarial Defense (16 attacks) | [10.5281/zenodo.21515412](https://doi.org/10.5281/zenodo.21515412) |

**Paper 1 Code:** [github.com/David-Dodge-AFI/solenoidal-closure-verification](https://github.com/David-Dodge-AFI/solenoidal-closure-verification)

Paper 1 proves the geometry prevents blowup. Paper 2 (this package) proves the same geometry predicts the statistics.

### Paper 2 (This Work) — Links

| Document | DOI |
|----------|-----|
| Turbulence Intermittency Manuscript | [10.5281/zenodo.22085896](https://doi.org/10.5281/zenodo.22085896) |

**Paper 2 Code:** [github.com/David-Dodge-AFI/solenoidal-intermittency-verification](https://github.com/David-Dodge-AFI/solenoidal-intermittency-verification)

---

## What This Is

This repository contains the manuscript, computational verification scripts, and figures for Paper 2 in the solenoidal projector series:

- **Paper 1** (released separately): The same projector geometry proves global regularity (no finite-time blowup).
- **Paper 2** (this package): The same projector geometry *predicts* the anomalous scaling exponents of turbulence intermittency — with zero free parameters.

The central result: **ζ_p = p/3 − p(p−3)/81**, derived from the coupling fraction f_c = 1/d = 1/3 (exact projector trace) and the bilinear NS nonlinearity giving μ = 2f_c² = 2/9.

---

## Contents

```
release/
├── Turbulence_Intermittency_Manuscript_V1.docx   ← the paper
├── Turbulence_Intermittency_Manuscript_V1.md      ← markdown source
├── README.md                                       ← this file
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── code/
│   ├── 01_projector_algebra/      ← pure math (no DNS needed)
│   │   ├── transfer_matrix_proof.py
│   │   ├── projector_variance_proof.py
│   │   ├── projector_attractor_eigenvalues.py
│   │   └── projector_survival_adversarial.py
│   ├── 02_intermittency_derivation/   ← μ = 2/9 chain
│   │   └── cascade_ratio_proof.py
│   ├── 03_dns_confirmation/       ← JHTDB scripts (need free token)
│   │   ├── burst_snapshot.py
│   │   ├── fixed_location_tracking.py
│   │   ├── burst_ratio_survey_channel.py
│   │   ├── burst_ratio_survey_isotropic.py
│   │   ├── burst_ratio_survey_transition_bl.py
│   │   ├── injection_wall_vs_center.py
│   │   ├── coherent_structure_l2_test.py
│   │   └── spatial_survey_v3.py
│   └── 04_comparison/            ← ζ_p vs experiment (pure compute)
│       └── zeta_p_comparison.py
└── figures/
    ├── fig1_zeta_p_comparison.png
    ├── fig2_l_mode_spectrum.png
    └── fig3_spatial_survey.png
```

---

## Running the Code

### Categories 01 and 02 (Pure Math — No External Data)

These scripts verify the projector algebra and the μ = 2/9 derivation. They require only numpy and scipy:

```bash
pip install numpy scipy
python code/01_projector_algebra/transfer_matrix_proof.py
python code/01_projector_algebra/projector_variance_proof.py
python code/01_projector_algebra/projector_attractor_eigenvalues.py
python code/02_intermittency_derivation/cascade_ratio_proof.py
```

### Category 03 (DNS Confirmation — Requires JHTDB Token)

These scripts query the Johns Hopkins Turbulence Database (JHTDB). They use the free demo token (built into each script):

```bash
pip install givernylocal numpy scipy
python code/03_dns_confirmation/spatial_survey_v3.py
```

The free demo token (`edu.jhu.pha.turbulence.testing-201406`) is rate-limited to 4096 points per call, which sets the grid to N=32. This is a constraint, not a choice — it is documented in each script's dial ledger.

### Category 04 (Comparison — Pure Compute)

Compares our zero-parameter prediction against 40 years of experimental and DNS data:

```bash
python code/04_comparison/zeta_p_comparison.py
```

---

## Key Results

| Quantity | Value | Method |
|---|---|---|
| Coupling fraction f_c | 1/d = 1/3 | Exact projector trace |
| Intermittency parameter μ | 2/9 = 0.2222 | From bilinear NS term |
| Scaling exponents ζ_p | p/3 − p(p−3)/81 | K62 with derived μ |
| RMS error vs experiment | 0.007–0.009 | vs Frisch/Gotoh/Arneodo/Saw |
| Attractor survival | 1/5 = 0.200 | Legendre orthogonality (exact) |
| l=2 spatial dominance | 17/20 positions (85%) | Spatial Survey V3 |
| Free parameters | **Zero** | — |

---

## Citation

See CITATION.cff for machine-readable citation, or cite as:

> D. Dodge, "Solenoidal Projector Geometry as the Origin of Turbulence Intermittency: A Zero-Parameter Derivation of Anomalous Scaling Exponents," August 2026.

---

## License

MIT. See LICENSE file.
