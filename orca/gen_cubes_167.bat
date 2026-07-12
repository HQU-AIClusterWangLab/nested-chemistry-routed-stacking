@echo off
cd /d D:\lunwen\2.1sci\orca
set MWF=D:\Multiwfn_3.8_dev_bin_Win64\Multiwfn.exe
set OUT=D:\lunwen\2.1sci\phase_6\final_nested_router\05_post_dft_analysis_templates\generated_inputs\orca

echo === BSe9-167 density ===
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-167_sample_orca_mayer_density_elf.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-167_sample_orca_mayer_density_elf_density.cube" > NUL
echo done

echo === BSe9-167 ELF ===
(echo. && echo 5 && echo 9 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-167_sample_orca_mayer_density_elf.molden.input < __c.txt > NUL 2>&1
move /Y ELF.cub "%OUT%\BSe9-167_sample_orca_mayer_density_elf_elf.cube" > NUL
echo done

echo === BSe9-167 fragment B ===
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-167_sample_fragment_B.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-167_sample_fragment_B_density.cube" > NUL
echo done

echo === BSe9-167 fragment Se9 ===
(echo. && echo 5 && echo 1 && echo 1 && echo 2 && echo 0 && echo 0 && echo q) > __c.txt
%MWF% BSe9-167_sample_fragment_Se9.molden.input < __c.txt > NUL 2>&1
move /Y density.cub "%OUT%\BSe9-167_sample_fragment_Se9_density.cube" > NUL
echo done

del __c.txt 2>NUL
echo ALL DONE
