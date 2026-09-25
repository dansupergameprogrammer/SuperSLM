@echo off
rem TE-437 catch-totality gate (tools/ci/check_gpu_catch_totality.py), must-accept: the TE-435 tip (91779ad,
rem git archive in D:\_artifacts\SuperSLM\TE437-CATCH-TOTALITY\src-91779ad) unmodified; the same tip with its one
rem nested catch (...) removed (Device::Init's innermost, exempt by the gate's own rule); and the tip plus comments,
rem literals, digit separators and a nested typed-only try. Exit 0 only when every leg reads OK.
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\commission_gpu_catch_totality.py" --mode accept --gate "%~dp0..\..\..\tools\ci\check_gpu_catch_totality.py" --tip D:\_artifacts\SuperSLM\TE437-CATCH-TOTALITY\src-91779ad
exit /b %errorlevel%
