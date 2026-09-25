@echo off
rem TE-433 tail pin, must-accept: the TE-432 fix round (042bd66, both tails' catch (...) present), every leg:
rem T13 by the decode step, T15 by both sub-chunks of a 5-token prompt, a foreign exception and a length_error.
rem Built by build_red_suite.bat into D:\_te433\final-tip. Exit 0 only when every leg exits 0.
setlocal
set B=D:\_te433\final-tip\te433_tail_pin.exe
set Q=--qwen3=D:\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm
for %%K in (foreign length_error) do (
  "%B%" --route=step --kind=%%K --len=5 %Q% || exit /b 1
  "%B%" --route=prompt --kind=%%K --tail=first --len=5 %Q% || exit /b 1
  "%B%" --route=prompt --kind=%%K --tail=last --len=5 %Q% || exit /b 1
)
exit /b 0
