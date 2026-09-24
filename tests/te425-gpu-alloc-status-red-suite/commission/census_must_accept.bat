@echo off
rem TE-425 census check, must-accept: the committed site list against the 1.8.0 fix source (the builder's tip
rem dad862e, exported by git archive to D:\_te425\tip862).
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\check_gpu_status_site_census.py" --root D:\_te425\tip862
exit /b %errorlevel%
