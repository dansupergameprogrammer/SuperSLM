@echo off
rem TE-425 census check, must-accept: the committed site list against the TE-435 fix round's source (91779ad,
rem exported by git archive to D:\_artifacts\SuperSLM\TE425-SITE-CENSUS\src-91779ad). Re-pointed from 042bd66 by TE-437.
"C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0..\..\ci\check_gpu_status_site_census.py" --root D:\_artifacts\SuperSLM\TE425-SITE-CENSUS\src-91779ad
exit /b %errorlevel%
