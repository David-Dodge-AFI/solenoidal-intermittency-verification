@echo off
REM Run all verification scripts for Paper 2: Turbulence Intermittency
REM Categories 01-02 + 04: pure math (no external data needed)
REM Category 03: requires JHTDB access (slow, rate-limited)

echo ============================================================
echo PAPER 2: Turbulence Intermittency — Full Verification Suite
echo ============================================================
echo.

echo [01] Projector Algebra (pure math)...
python code\01_projector_algebra\transfer_matrix_proof.py
python code\01_projector_algebra\projector_variance_proof.py
python code\01_projector_algebra\projector_attractor_eigenvalues.py
python code\01_projector_algebra\projector_survival_adversarial.py
echo.

echo [02] Intermittency Derivation (pure math)...
python code\02_intermittency_derivation\cascade_ratio_proof.py
echo.

echo [04] Comparison to Experiment (pure compute)...
python code\04_comparison\zeta_p_comparison.py
echo.

echo ============================================================
echo Categories 01, 02, 04 complete.
echo.
echo Category 03 (DNS confirmation) requires JHTDB access.
echo These are slow (~15-30 min each) and rate-limited.
echo Run individually if needed:
echo   python code\03_dns_confirmation\spatial_survey_v3.py
echo   python code\03_dns_confirmation\coherent_structure_l2_test.py
echo   (etc.)
echo ============================================================
pause
