# Schema fast-decode box results, 2026-10-20 (plan §10, B0-B4d)

The driver's output for every step, copied unchanged from its `out\` folder, and the four sealed on-arm files
copied unopened from its `sealed\` folder.

- **Box:** AMD Ryzen 9 3950X (16 cores / 32 threads, AVX2), Windows 10, High performance power plan. MSVC 19.44.35214, Release, Ninja.
- **Model:** `D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm`, sha256
  `9ccf7e378bbe47aa0ffd3810aa82875926cbe51af1bdfa08393c3fae657463da`. This is A-EX with the tokenizer, and with
  `potion_shop_order` and `prompt_result` in one SCM1 section, the file the Unreal example ships as `ExampleModel.sslm`.
  The driver's default model at the time, `qwen2.5-0.5b-instruct-cap4096-aex.sslm`, has no tokenizer: B1 stopped at
  `sslm_tokenize failed` before measuring anything, and the steps were run with `-Model` set to the file above.
- **Frozen inputs:** acceptance rules `cef1d73e…`, potion population v1 `d846b9e3…`, te368 population `ef597551…`,
  each checked by the driver.
- **Driver:** `run_box_step.ps1` at the J1 commit, sha256 `2d9db27d1059…`, with `-Pairs 15 -Tasks 4`.

| Step | J1 commit (this branch) | main | Started (UTC) |
|---|---|---|---|
| B0 | `79e5b9234c5026eac6453fb4e3e347becfca84c9` | `v1.13.1` (`d697927f6d`) | 2026-10-20 08:10:50 |
| B1 | `79e5b92` | `v1.13.1` | 08:15:15 |
| B2 | `79e5b92` | `v1.13.1` | 08:17:55 |
| B3 | `79e5b92` | `v1.13.1` | 08:24:46 |
| B4a | `79e5b92` | `v1.13.1` | 08:32:00 |
| B4b | `79e5b92` | `v1.13.1` | 08:34:03 |
| B4c | `79e5b92` | `v1.13.1` | 08:35:39 |
| B4d | `79e5b92` | `v1.13.1` | 08:39:22 |

Every step's file ends `"complete": true}`. One step ran per invocation, with a 60 s rest between steps. Host load
before and after each step was 0-12 %, with no build, editor or other offender process running.

**Notes on the readings.**

- B1's finish reports `damped_available: false` at 1 and 4 tasks on this artifact, so its damped arm was not measured.
- Before B4c, the library's tokenization of the te368 population (Qwen2.5's default chat template, as the bench builds it)
  was compared with the 40 original HF-tokenized `new_NN.ids` files (Wizard `Claude/Loki/te368-probe/ids/`): all 40
  sequences are identical.
- Sealed on-arm files (sha256): `B4a.on.json` `a9dd06cdfd68…`, `B4b.on.json` `437aca7c79a5…`, `B4c.on.json`
  `a994ddcc86e4…`, `B4d.on.json` `0cd7e6b65927…`. Each matches the hash in its step's `sealed_on` record.
