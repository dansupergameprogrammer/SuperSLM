@echo off
rem T-2112 (Curie): commissioning re-run of the promoted dim-7 probe, per
rem StandardsDocument.md Sec5.4's instrument-commissioning rule. Must-reject
rem construction: the four cells the T-2111 strike's own Phase 4 table found
rem broken (control + the three that previously failed), compiled against the
rem PRE-FOLD header recovered from git history (Wizard repo commit 7492a1ef90)
rem -- genuinely producible (it is the design's own prior, real state) and
rem independent of this suite's own authorship. Must reproduce the same
rem failure classes the T-2111 fold's own transcript recorded (design Sec17):
rem C2660 (batch-call arity) and C2440 (void-to-SslmGpuStatus). The control
rem cell must still pass (it needed no repair). This re-run is against the
rem RELOCATED probe (this directory, not Claude/Loki/t2111-probe) -- the
rem point of re-running rather than citing Sec17's own transcript is to
rem confirm relocation did not silently change what the instrument can detect.
setlocal
set HEREDIR=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
for %%f in (cell_control_decode_step.c cell_dim5_device_lost_midbatch.c cell_dim11_guard_vitality.c cell_s7_batch_granularity.c) do (
    echo === %%f ^(pre-fold header^) ===
    cl /nologo /c /TP /W3 /WX /Iprefold "%%f" /Fo:"obj\prefold_%%~nf.obj"
    echo    exit: %errorlevel%
)
