@echo off
rem T-2791 (Curie): runs every cell build_red_suite.bat linked, passing the artifact flags through,
rem and prints each cell's exit code. Cells that did not link are reported as not built (their red
rem reading is build_red_suite.bat's output). Cell_functional_commission is excluded unless
rem --with-commission is given: it is about 11 GPU minutes (plan Sec3.4 row 10) and belongs to the
rem release reading (plan Sec3.5 step 5).
rem
rem A SKIPPED CELL FAILS THE RUN (plan Sec3.5 step 1, T-2806 M2): each cell exits non-zero when it
rem skipped anything because an artifact flag was left out, unless --allow-skip is passed for a
rem deliberate partial run. An acceptance run passes every artifact flag below and never --allow-skip.
rem
rem Usage: run_red_suite.bat ["--bin=DIR"] ["--qwen3=PATH"] ["--synthetic=PATH"] ["--a2fn=PATH"] ["--gan=PATH"]
rem            ["--g5fixture=PATH"] ["--g5an=PATH"] ["--commission=PATH"] [--with-commission] [--allow-skip]
rem            (quote each flag carrying '=')
rem   --bin defaults to <repo>\build\t2791\bin.
rem Artifacts this suite was authored against (read-only; SHA-256):
rem   --qwen3      qwen3-embedding-0.6b-1p5.sslm         0be28bf42637264df85481eb925c46f674da7db97fd7601c6732881ad8f99fbf
rem   --synthetic  the synthetic fused-K fixture (plan Sec3.4 row 4): SuperEmbedder's u1_pair_model.sslm,
rem                729,680 bytes, written by tests/fixtures/regenerate_tokenizer_pair_fixture.py at
rem                SuperEmbedder 701ef1a or later against v1.5.0
rem                                                       a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70
rem   --a2fn       A2-fn (plan Sec3.4 row 5), built from --synthetic by this directory's
rem                make_a2fn_fixture.py <u1_pair_model.sslm> <out>
rem                                                       73a1ec9e1846be16af7f8dc13511207d4129a9e825cc5271f783f6bc1ff4f940
rem   --gan        G-an (plan Sec3.4 row 5, Sec3.6), built from --synthetic by this directory's
rem                make_gan_fixture.py <u1_pair_model.sslm> <out>
rem                                                       cf48079cd3b50eb053c8f8cd57d4feb0f102d8ec9622aaf86300a15daf96e76b
rem   --g5fixture  t2132_g5_fixture_1p5b.sslm           078df885060d5dea23a88983bb68014843d142cb6ad55c7f70ef9ff9a932a019
rem   --g5an       G5-an (plan Sec3.4 row 5, Q5-1), built from --g5fixture by this directory's
rem                make_g5an_fixture.py <t2132_g5_fixture_1p5b.sslm> <out>   (about 1.6 GB)
rem                                                       05b5d5c58ea14dbb15a3adb7f24668edcd4d7ab83344bc4381c5f7347b5a128b
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
for %%f in (cell_status_ordinals cell_census_lifetime cell_hostile_capacity cell_prefill_faults_schema cell_final_norm_guard ^
            cell_prompt_guard_status cell_schema_guard_status cell_prompt_guard_removal cell_concurrency ^
            cell_determinism_composition cell_env_pins_shipping_leg ^
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
