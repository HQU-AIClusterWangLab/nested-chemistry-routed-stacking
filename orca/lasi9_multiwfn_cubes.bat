@echo off
setlocal enabledelayedexpansion
set MWF=D:\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe
set WORKDIR=D:\lunwen\2.1sci\orca
set OUTDIR=D:\lunwen\2.1sci\phase_6\final_nested_router\05_post_dft_analysis_templates\generated_inputs\orca\LaSi9
cd /d %WORKDIR%

echo ================================================================
echo LaSi9: Multiwfn density + ELF cubes (matched grids)
echo ================================================================

for %%N in (1472 2546 1792) do (
    set FULL=!OUTDIR!\LaSi9-%%N_sample_orca_mayer_density.cube

    echo.
    echo === LaSi9-%%N full density ===
    (echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > _c.txt
    %MWF% LaSi9-%%N_sample_orca_mayer.molden.input ^< _c.txt > NUL 2>&1
    move /Y density.cub "!FULL!" > NUL
    echo   done

    echo === LaSi9-%%N ELF ===
    (echo. && echo 5 && echo 9 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > _c.txt
    %MWF% LaSi9-%%N_sample_orca_mayer.molden.input ^< _c.txt > NUL 2>&1
    move /Y ELF.cub "!OUTDIR!\LaSi9-%%N_sample_orca_mayer_elf.cube" > NUL
    echo   done

    echo === LaSi9-%%N fragment La (matched grid) ===
    (echo. && echo 5 && echo 1 && echo 8 && echo !FULL! && echo 2 && echo 0 && echo 0 && echo q) > _c.txt
    %MWF% LaSi9-%%N_sample_fragment_La.molden.input ^< _c.txt > NUL 2>&1
    move /Y density.cub "!OUTDIR!\LaSi9-%%N_sample_fragment_La_density.cube" > NUL
    echo   done

    echo === LaSi9-%%N fragment Si9 (matched grid) ===
    (echo. && echo 5 && echo 1 && echo 8 && echo !FULL! && echo 2 && echo 0 && echo 0 && echo q) > _c.txt
    %MWF% LaSi9-%%N_sample_fragment_Si9.molden.input ^< _c.txt > NUL 2>&1
    move /Y density.cub "!OUTDIR!\LaSi9-%%N_sample_fragment_Si9_density.cube" > NUL
    echo   done
)

del _c.txt 2>NUL
echo.
echo ================================================================
echo ALL DONE. Check LaSi9 cubes:
dir "%OUTDIR%\*.cube" 2>NUL
echo ================================================================
