@echo off
rem TE-425 census check, must-accept: the committed site list against the TE-432 fix round's source (042bd66,
rem exported by git archive to D:\_te433\src-042bd66). Re-pointed from dad862e by TE-433.
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\check_gpu_status_site_census.py" --root D:\_te433\src-042bd66
exit /b %errorlevel%
