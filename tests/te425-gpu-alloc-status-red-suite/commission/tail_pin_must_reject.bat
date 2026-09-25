@echo off
rem TE-433 tail pin, must-reject: 042bd66 with one tail's catch (...) deleted -- the builder's own TE-432 hunk
rem reversed, which is the tail as it stood at 7dda90a -- each driven at its own tail, both kinds:
rem   D:\_te433\final-mutT13 (RunLayerLoopGpuSubmit's clause deleted): the decode-step legs;
rem   D:\_te433\final-mutT15 (SubmitOneSubChunkToFullDepthForG5Bridge's clause deleted): both prompt sub-chunk legs.
rem Exit 1 only when every leg returns the cell's RED exit (1); any other exit (a setup failure 3, an undriven
rem leg 4, a crash) exits 0, so a broken construction never reads as a rejection.
setlocal
set Q=--qwen3=D:\_artifacts\superslm\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm
set M13=D:\_te433\final-mutT13\te433_tail_pin.exe
set M15=D:\_te433\final-mutT15\te433_tail_pin.exe
for %%K in (foreign length_error) do (
  "%M13%" --route=step --kind=%%K --len=5 %Q%
  if not errorlevel 1 exit /b 0
  if errorlevel 2 exit /b 0
  "%M15%" --route=prompt --kind=%%K --tail=first --len=5 %Q%
  if not errorlevel 1 exit /b 0
  if errorlevel 2 exit /b 0
  "%M15%" --route=prompt --kind=%%K --tail=last --len=5 %Q%
  if not errorlevel 1 exit /b 0
  if errorlevel 2 exit /b 0
)
exit /b 1
