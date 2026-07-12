@echo off
setlocal enabledelayedexpansion
cd /d D:\lunwen\2.1sci\orca
set MWF=D:\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe
set CUBEDIR=D:\lunwen\2.1sci\phase_6\final_nested_router\05_post_dft_analysis_templates\generated_inputs\orca

for %%N in (167 256 84) do (
    set FULL=!CUBEDIR!\BSe9-%%N_sample_orca_mayer_density_elf_density.cube

    echo === BSe9-%%N fragment B (matched grid) ===
    (echo. && echo 5 && echo 1 && echo 8 && echo !FULL! && echo 2 && echo 0 && echo 0 && echo q) > c.txt
    %MWF% BSe9-%%N_sample_fragment_B.molden.input < c.txt > NUL 2>&1
    move /Y density.cub "!CUBEDIR!\BSe9-%%N_sample_fragment_B_density.cube" > NUL
    echo done

    echo === BSe9-%%N fragment Se9 (matched grid) ===
    (echo. && echo 5 && echo 1 && echo 8 && echo !FULL! && echo 2 && echo 0 && echo 0 && echo q) > c.txt
    %MWF% BSe9-%%N_sample_fragment_Se9.molden.input < c.txt > NUL 2>&1
    move /Y density.cub "!CUBEDIR!\BSe9-%%N_sample_fragment_Se9_density.cube" > NUL
    echo done
)

del c.txt 2>NUL

echo.
echo ================================================================
echo All fragment cubes regenerated with matched grids.
echo Now run: python _cdd_cubes.py
echo ================================================================
