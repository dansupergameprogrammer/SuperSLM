@echo off
rem TE-425 census check, must-accept: the committed site list against the v1.7.1 source.
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\check_gpu_status_site_census.py" --root D:\_te425\slm171
exit /b %errorlevel%
