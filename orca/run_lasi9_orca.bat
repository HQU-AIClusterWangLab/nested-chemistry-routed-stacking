@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM LaSi9: Run all 9 ORCA single-point jobs
REM ============================================================
set ORCA_PATH=D:\ORCA\orca
set INDIR=D:\lunwen\2.1sci\phase_6\final_nested_router\05_post_dft_analysis_templates\generated_inputs\orca\LaSi9
set WORKDIR=D:\lunwen\2.1sci\orca
cd /d %WORKDIR%

echo ================================================================
echo LaSi9 ORCA single-points (Mayer bond orders)
echo ================================================================

for %%S in (1472 2546 1792) do (
    echo.
    echo [%%S] Full cluster ...
    copy /Y "%INDIR%\LaSi9-%%S_sample_orca_mayer.inp" . > NUL
    %ORCA_PATH% LaSi9-%%S_sample_orca_mayer.inp > LaSi9-%%S_sample_orca_mayer.out 2>&1

    echo [%%S] Fragment La ...
    copy /Y "%INDIR%\LaSi9-%%S_sample_fragment_La.inp" . > NUL
    %ORCA_PATH% LaSi9-%%S_sample_fragment_La.inp > LaSi9-%%S_sample_fragment_La.out 2>&1

    echo [%%S] Fragment Si9 ...
    copy /Y "%INDIR%\LaSi9-%%S_sample_fragment_Si9.inp" . > NUL
    %ORCA_PATH% LaSi9-%%S_sample_fragment_Si9.inp > LaSi9-%%S_sample_fragment_Si9.out 2>&1
)

echo.
echo ================================================================
echo ALL DONE. Check for "****ORCA TERMINATED NORMALLY****" in *.out
echo ================================================================
