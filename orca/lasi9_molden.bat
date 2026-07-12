@echo off
cd /d D:\lunwen\2.1sci\orca
echo ================================================================
echo Step 1: Convert all LaSi9 .gbw to .molden.input
echo ================================================================
for %%S in (1472 2546 1792) do (
    echo LaSi9-%%S full ...
    D:\ORCA\orca_2mkl LaSi9-%%S_sample_orca_mayer -molden
    echo LaSi9-%%S fragment La ...
    D:\ORCA\orca_2mkl LaSi9-%%S_sample_fragment_La -molden
    echo LaSi9-%%S fragment Si9 ...
    D:\ORCA\orca_2mkl LaSi9-%%S_sample_fragment_Si9 -molden
)
echo Done. 9 .molden.input files generated.
echo.
echo Now run: lasi9_multiwfn_cubes.bat
