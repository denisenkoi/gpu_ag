@echo off
REM Slicer DE Mode - 3 iterations test
REM GPU AG Project - CPU Baseline

e:
cd "E:\Projects\Rogii\gpu_ag\cpu_baseline"

set DE_STARSTEER_DIR=E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de

echo SLICER DE - 3 iterations (GPU AG CPU Baseline)
"C:\ProgramData\Anaconda3\envs\rl\python.exe" slicer.py --de --starsteer-dir "%DE_STARSTEER_DIR%" --max-iterations 3

echo Finished with code: %ERRORLEVEL%
pause
