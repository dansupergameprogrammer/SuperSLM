@echo off
rem TE-425 entry census, must-accept: the planner's disposition-3 entry points (plan Sec3.5 R15 item 3), v1.7.1.
"D:\_te425\bin171\te425_cells.exe" r15zero --set=qwen3 --qwen3=D:\_artifacts\superslm\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm
exit /b %errorlevel%
