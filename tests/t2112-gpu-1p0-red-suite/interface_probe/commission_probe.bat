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

rem T-2113 (B9): the real-engine-headers leg's own must-reject commissioning
rem (design Sec11 dim 7, "confirmed at fold" -- re-run here against THIS
rem cell now that it is a standing build-gate fixture, not merely re-cited
rem from the design's own one-off T-2112 mini-fold transcript). Must-reject:
rem the PRE-fold prefold\sslm_gpu_1p0.h (its own global-scope SslmModelView
rem stub) combined with the real superslm/model.h -- must fail with C2872
rem (ambiguous symbol), reproducing the real T-2112 Finding 1 defect. Must-
rem accept: the identical cell against the real, post-fold header (one level
rem up) -- must compile clean.
echo === cell_dim7_real_engine_headers.c ^(pre-fold header, MUST FAIL C2872^) ===
cl /nologo /c /TP /std:c++20 /W3 /WX /Iprefold /I..\..\..\include cell_dim7_real_engine_headers.c /Fo:"obj\prefold_cell_dim7_real_engine_headers.obj"
echo    exit: %errorlevel%
echo === cell_dim7_real_engine_headers.c ^(post-fold header, MUST COMPILE CLEAN^) ===
cl /nologo /c /TP /std:c++20 /W3 /WX /I.. /I..\..\..\include cell_dim7_real_engine_headers.c /Fo:"obj\postfold_cell_dim7_real_engine_headers.obj"
echo    exit: %errorlevel%
