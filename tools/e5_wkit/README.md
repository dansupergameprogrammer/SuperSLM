# E5 W1–W4 kit: cell 9.10 on the RX 7900 XTX

This kit runs the last 1.14.0 release check away from the home box. It is G1–G4 of
`docs/release/e5-box-2026-10-23/e5-box-runsheet-2026-10-23.md`, run on the work machine's RX 7900 XTX as W1–W4:

| Piece | What it does |
|---|---|
| W1 | builds v1.13.1's library and shaders, then links `pcg_cell_9_10_capture.exe` against them |
| W2 | captures the five v1.13.1 GPU saves on the AMD adapter |
| W3 | builds the release tree's library and shaders, then links `pcg_cell_9_10.exe` |
| W4 | grades 9.10: each capture restores, re-saves byte-identical, and continues to its pinned 64 tokens |

## How it runs

The commands are the run sheet's, including the double quotes around `--name=value` arguments to
`build_paged_kv_gpu.bat`.

**Adapter.** The kit picks the adapter by vendor id, not by a fixed index:
- It compiles the engine's own lister, `tools/t2116_list_adapters.cpp`. That tool reports adapters in raw
  `EnumAdapters1` order, the order `SSLM_GPU_ADAPTER_INDEX` selects by.
- It picks the usable hardware adapter with vendor id `0x1002` and the most dedicated memory. The 7950X3D's integrated
  Radeon is also vendor `0x1002`; requiring at least 4 GB of dedicated memory keeps it from being chosen.

**Resuming.** Each piece writes `results\<piece>.done` when it passes and is skipped on a rerun. The first red stops
the run, and a rerun retries that piece. Builds go to `C:\_scratch\e5w` (`-WorkRoot` changes this), never to the drive.

## Home box: fill the drive

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\e5_wkit\fill_drive.ps1 -Drive F:
```

This writes `F:\e5-w\` and leaves any earlier `results\` in place:
- `START.cmd`;
- `kit\`;
- `inputs\e5w.bundle`, holding `ce82c898` and `d697927f` with full history;
- `inputs\fixtures\pkv_def.sslm`.

## Work machine

```
F:\e5-w\START.cmd
```

The precheck runs first and lists, all at once, anything that is missing:
- VS 2022 with MSVC 14.34 and Windows SDK 10.0.22621.0;
- VS 2022 Community's `VsDevCmd.bat`, which the batch file calls at a fixed path;
- the SDK's `dxc.exe`;
- cmake, git and python.

`START.cmd -CheckOnly` runs only the precheck and adapter detection.

## What to bring back

`F:\e5-w\results\`:
- `SUMMARY.md`;
- `adapter.txt`;
- `logs\`, holding the W1–W4 logs, `W0-source` and `adapter`;
- `captures910\v1.13.1_1002_<device>\`, the pin: `captures.txt` and five `.bin` files;
- `SHA256SUMS.txt`.
