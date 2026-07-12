@echo off
REM ============================================================
REM BSe9 ORCA Post-DFT Analysis — Run All Jobs (Mayer only)
REM Cube generation: see step 2 below or use orca_plot after jobs finish
REM ============================================================

set ORCA_PATH=D:\ORCA\orca
set WORKDIR=%~dp0

echo ================================================================
echo BSe9 ORCA post-DFT analysis — full-cluster + fragment single-points
echo Step 1/2: Run all 9 single-point jobs (Mayer bond orders)
echo ================================================================
echo.

REM --- 3 full-cluster jobs (Mayer only, no cube in this pass) ---
echo [1/9] BSe9-167 full cluster ...
%ORCA_PATH% %WORKDIR%BSe9-167_sample_orca_mayer_density_elf.inp > %WORKDIR%BSe9-167_sample_orca_mayer_density_elf.out 2>&1

echo [2/9] BSe9-84 full cluster ...
%ORCA_PATH% %WORKDIR%BSe9-84_sample_orca_mayer_density_elf.inp > %WORKDIR%BSe9-84_sample_orca_mayer_density_elf.out 2>&1

echo [3/9] BSe9-256 full cluster ...
%ORCA_PATH% %WORKDIR%BSe9-256_sample_orca_mayer_density_elf.inp > %WORKDIR%BSe9-256_sample_orca_mayer_density_elf.out 2>&1

REM --- 3 B-fragment jobs ---
echo [4/9] BSe9-167 fragment B ...
%ORCA_PATH% %WORKDIR%BSe9-167_sample_fragment_B.inp > %WORKDIR%BSe9-167_sample_fragment_B.out 2>&1

echo [5/9] BSe9-84 fragment B ...
%ORCA_PATH% %WORKDIR%BSe9-84_sample_fragment_B.inp > %WORKDIR%BSe9-84_sample_fragment_B.out 2>&1

echo [6/9] BSe9-256 fragment B ...
%ORCA_PATH% %WORKDIR%BSe9-256_sample_fragment_B.inp > %WORKDIR%BSe9-256_sample_fragment_B.out 2>&1

REM --- 3 Se9-fragment jobs ---
echo [7/9] BSe9-167 fragment Se9 ...
%ORCA_PATH% %WORKDIR%BSe9-167_sample_fragment_Se9.inp > %WORKDIR%BSe9-167_sample_fragment_Se9.out 2>&1

echo [8/9] BSe9-84 fragment Se9 ...
%ORCA_PATH% %WORKDIR%BSe9-84_sample_fragment_Se9.inp > %WORKDIR%BSe9-84_sample_fragment_Se9.out 2>&1

echo [9/9] BSe9-256 fragment Se9 ...
%ORCA_PATH% %WORKDIR%BSe9-256_sample_fragment_Se9.inp > %WORKDIR%BSe9-256_sample_fragment_Se9.out 2>&1

echo.
echo ================================================================
echo Step 1 done. Check for "****ORCA TERMINATED NORMALLY****" in *.out
echo.
echo Step 2: Generate cube files via orca_plot (interactive):
echo   cd to this folder, then for each *_orca_mayer_density_elf.gbw:
echo     D:\ORCA\orca_plot name.gbw
echo     ^> Enter 1 (density), then filename: name_density.cube
echo     ^> Enter 2 (ELF), then filename: name_elf.cube
echo     ^> Use grid 80^>80^>80
echo ================================================================
