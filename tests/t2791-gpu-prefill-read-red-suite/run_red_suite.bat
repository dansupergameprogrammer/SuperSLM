@echo off
rem T-2791 (Curie): runs every cell build_red_suite.bat linked, passing the artifact flags through,
rem and prints each cell's exit code. Cells that did not link are reported as not built (their red
rem reading is build_red_suite.bat's output). Cell_functional_commission is excluded unless
rem --with-commission is given: it is about 11 GPU minutes (plan Sec3.4 row 10) and belongs to the
rem release reading (plan Sec3.5 step 5).
rem
rem Usage: run_red_suite.bat ["--bin=DIR"] ["--qwen3=PATH"] ["--synthetic=PATH"] ["--g5fixture=PATH"]
rem            ["--commission=PATH"] [--with-commission]           (quote each flag carrying '=')
rem   --bin defaults to <repo>\build\t2791\bin.
rem Artifacts this suite was authored against (read-only; SHA-256):
rem   --qwen3      qwen3-embedding-0.6b-1p5.sslm         0be28bf42637264df85481eb925c46f674da7db97fd7601c6732881ad8f99fbf
rem   --synthetic  u1_pair_model.sslm (fused-K, head_dim 128, SuperEmbedder 701ef1a's U1 pair generator)
rem                                                       a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70
rem   --g5fixture  t2132_g5_fixture_1p5b.sslm            078df885060d5dea23a88983bb68014843d142cb6ad55c7f70ef9ff9a932a019
rem   --commission T-2780 tokens.txt                      01c9b0471695086cd708e02ad9826d70d8745fc3296610b2c2420f6df9091f82
setlocal enabledelayedexpansion
for %%I in ("%~dp0..\..") do set ENG=%%~fI
set BIN=%ENG%\build\t2791\bin
set PASS=
set WITHCOMM=0
:parse_args
if "%~1"=="" goto :args_done
set ARG=%~1
if "!ARG:~0,6!"=="--bin=" (
    set BIN=!ARG:~6!
) else if "!ARG!"=="--with-commission" (
    set WITHCOMM=1
) else (
    set PASS=!PASS! "!ARG!"
)
shift
goto :parse_args
:args_done
set ANYFAIL=0
for %%f in (cell_status_ordinals cell_census_lifetime cell_hostile_capacity cell_prefill_faults_schema ^
            cell_concurrency cell_determinism_composition cell_env_pins_shipping_leg ^
            cell_wrapper_census_standing cell_functional_commission) do (
    set SKIPTHIS=0
    if "%%f"=="cell_functional_commission" if "!WITHCOMM!"=="0" set SKIPTHIS=1
    if "!SKIPTHIS!"=="1" (
        echo ===== %%f: not run ^(pass --with-commission^)
    ) else if exist "%BIN%\%%f.exe" (
        echo ===== %%f
        "%BIN%\%%f.exe" !PASS!
        echo ===== %%f exit=!errorlevel!
        if not "!errorlevel!"=="0" set ANYFAIL=1
    ) else (
        echo ===== %%f: NOT BUILT -- red by compile or link, see build_red_suite.bat
        set ANYFAIL=1
    )
)
exit /b %ANYFAIL%
