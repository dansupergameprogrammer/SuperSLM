@echo off
rem TE-425 entry census, must-reject: the planner's disposition-2 entry points (plan Sec3.5 R15 item 2).
"D:\_te425\bin171\te425_cells.exe" r15zero --set=disposition2 --qwen3=D:\_artifacts\superslm\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm
if "%errorlevel%"=="1" exit /b 1
exit /b 0
