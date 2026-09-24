@echo off
rem TE-425 site sweep, must-accept: the planner's TE-421 census's 144 conforming operator-new sites at v1.7.1.
"D:\_te425\bin171\te425_cells.exe" r3 --qwen3=D:\_t2743conv\flow-final\qwen3-embedding-0.6b-1p5.sslm --len=40 --sites=1-88,142-147,201-206,260-265,319-324,378-383,437-442,496-501,555-560,614-619,673-674
exit /b %errorlevel%
