@echo off
rem TE-437 catch-totality gate, must-reject: at the TE-435 tip (91779ad), each of the 18 catch (...) clauses that
rem closes an outermost try removed alone (deleted, or narrowed to std::exception where it is the only handler),
rem a typed-only try appended as a function and as a function-try-block, a typed-only try in a new file in a new
rem subdirectory, a raw string literal and an empty src/gpu (both exit 2), and c3b5412 itself (the ten ladders
rem TE-435 closed). Exit 1 only when every leg returns exactly its expected verdict; any other outcome exits 0.
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\commission_gpu_catch_totality.py" --mode reject --gate "%~dp0..\..\..\tools\ci\check_gpu_catch_totality.py" --tip D:\_artifacts\SuperSLM\TE437-CATCH-TOTALITY\src-91779ad --pre D:\_artifacts\SuperSLM\TE437-CATCH-TOTALITY\src-c3b5412
if "%errorlevel%"=="1" exit /b 1
exit /b 0
