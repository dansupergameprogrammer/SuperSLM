@echo off
rem TE-425 census check, must-reject: the committed list with the independently found population deleted
rem (TE-422 S-2 sites 1-3, TE-423 F-5's two exclusion lists, TE-424 P-1), against the TE-432 fix round's source
rem (042bd66, D:\_te433\src-042bd66).
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\check_gpu_status_site_census.py" --root D:\_te433\src-042bd66 --list "%~dp0census_population_deleted.txt"
if "%errorlevel%"=="1" exit /b 1
exit /b 0
