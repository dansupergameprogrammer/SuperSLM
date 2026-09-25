@echo off
rem TE-425 site sweep, must-reject: TE-421's first recording-window block (k 89-136) and TE-419 leg B's
rem construction (every D3D12 allocation of a two-sub-chunk encode). Exit 1 only when BOTH read RED.
"D:\_te425\bin171\te425_cells.exe" r3 --qwen3=D:\_artifacts\superslm\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm --len=40 --sites=89-136
if not "%errorlevel%"=="1" exit /b 0
"D:\_te425\bin171\te425_cells.exe" r2 --qwen3=D:\_artifacts\superslm\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm --len=5
if not "%errorlevel%"=="1" exit /b 0
exit /b 1
