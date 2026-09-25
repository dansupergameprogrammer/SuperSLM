@echo off
rem TE-425 census check, must-reject: the committed list with the independently found population deleted
rem (TE-422 S-2 sites 1-3, TE-423 F-5's two exclusion lists, TE-424 P-1, TE-431 S-1), against the TE-435 fix
rem round's source (91779ad, D:\_artifacts\SuperSLM\TE425-SITE-CENSUS\src-91779ad).
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\check_gpu_status_site_census.py" --root D:\_artifacts\SuperSLM\TE425-SITE-CENSUS\src-91779ad --list "%~dp0census_population_deleted.txt"
if "%errorlevel%"=="1" exit /b 1
exit /b 0
