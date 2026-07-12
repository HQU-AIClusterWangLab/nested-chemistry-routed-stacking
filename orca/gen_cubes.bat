@echo off
cd /d D:\lunwen\2.1sci\orca
set MWF=D:\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe
set OUT=D:\lunwen\2.1sci\phase_6\final_nested_router\05_post_dft_analysis_templates\generated_inputs\orca

echo ================================================================
echo Generating all cubes with rename after each run
echo ================================================================

REM --- BSe9-256 density ---
echo [1/8] BSe9-256 density
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-256_sample_orca_mayer_density_elf.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-256_sample_orca_mayer_density_elf_density.cube" > NUL
echo   done

REM --- BSe9-256 ELF ---
echo [2/8] BSe9-256 ELF
(echo. && echo 5 && echo 9 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-256_sample_orca_mayer_density_elf.molden.input < __c.txt > NUL 2>&1
move /Y ELF.cub "%OUT%\BSe9-256_sample_orca_mayer_density_elf_elf.cube" > NUL
echo   done

REM --- BSe9-84 density ---
echo [3/8] BSe9-84 density
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-84_sample_orca_mayer_density_elf.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-84_sample_orca_mayer_density_elf_density.cube" > NUL
echo   done

REM --- BSe9-84 ELF ---
echo [4/8] BSe9-84 ELF
(echo. && echo 5 && echo 9 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-84_sample_orca_mayer_density_elf.molden.input < __c.txt > NUL 2>&1
move /Y ELF.cub "%OUT%\BSe9-84_sample_orca_mayer_density_elf_elf.cube" > NUL
echo   done

REM --- BSe9-256 fragment B ---
echo [5/8] BSe9-256 fragment B
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-256_sample_fragment_B.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-256_sample_fragment_B_density.cube" > NUL
echo   done

REM --- BSe9-256 fragment Se9 ---
echo [6/8] BSe9-256 fragment Se9
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-256_sample_fragment_Se9.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-256_sample_fragment_Se9_density.cube" > NUL
echo   done

REM --- BSe9-84 fragment B ---
echo [7/8] BSe9-84 fragment B
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-84_sample_fragment_B.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-84_sample_fragment_B_density.cube" > NUL
echo   done

REM --- BSe9-84 fragment Se9 ---
echo [8/8] BSe9-84 fragment Se9
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-84_sample_fragment_Se9.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-84_sample_fragment_Se9_density.cube" > NUL
echo   done

REM --- Move BSe9-167 files that were already generated ---
if exist density.cub move /Y density.cub "%OUT%\BSe9-167_sample_orca_mayer_density_elf_density.cube" > NUL
if exist ELF.cub   move /Y ELF.cub   "%OUT%\BSe9-167_sample_orca_mayer_density_elf_elf.cube" > NUL

del __c.txt 2>NUL

echo.
echo ================================================================
echo DONE. Cube files:
dir "%OUT%\*_density.cube" "%OUT%\*_elf.cube" 2>NUL
echo ================================================================
