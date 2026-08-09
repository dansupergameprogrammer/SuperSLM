"""Stage 3 -- Q3: can it run one real forward of this project's actual model over
one real prompt, and does the output look sane?

Loads Qwen2.5-1.5B-Instruct (this campaign's own model, bundled at <bundle>/model/,
the identical checkpoint the RTX 2080 Super harness uses) from local files only --
no network access, no HuggingFace hub lookup -- tokenizes one real utterance from
this project's own corpus (bundled as prompts.jsonl), runs the model's chat
template, and generates a short continuation. "Looks sane" is checked
mechanically (decodes as text, non-empty, not degenerate single-character
repetition) and the actual text is also printed so Dan can eyeball it himself.
"""
import sys
import time
import traceback

from common import emit, fail, ok, load_prompts, model_dir

STAGE = "forward"
MAX_NEW_TOKENS = 24


def looks_sane(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty output"
    stripped = text.strip()
    if len(set(stripped)) <= 2 and len(stripped) > 8:
        return False, "output is degenerate (near-single-character repetition)"
    try:
        stripped.encode("utf-8")
    except Exception:
        return False, "output does not decode as valid UTF-8"
    return True, "ok"


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        emit(fail(STAGE, "torch or transformers did not import",
                  {"exception": repr(e), "traceback": traceback.format_exc()}))
        return 1

    if not torch.cuda.is_available():
        emit(fail(STAGE, "no device available -- see stage_import"))
        return 1

    mdir = model_dir()
    detail = {"model_dir": str(mdir)}
    print(f"[{STAGE}] loading tokenizer + model from {mdir} (local files only)...")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(mdir), local_files_only=True, torch_dtype=torch.bfloat16
        )
        model.to("cuda:0")
        model.eval()
    except Exception as e:
        print(f"[{STAGE}] model/tokenizer load raised: {e!r}")
        traceback.print_exc()
        detail["exception"] = repr(e)
        detail["traceback"] = traceback.format_exc()
        emit(fail(STAGE, "loading the real checkpoint or moving it onto the "
                          "device raised an exception", detail))
        return 1
    load_s = time.time() - t0
    detail["model_load_seconds"] = load_s
    print(f"[{STAGE}] model loaded and on device in {load_s:.1f}s")

    prompts = load_prompts(limit=1)
    if not prompts:
        emit(fail(STAGE, "prompts.jsonl is missing or empty in the bundle"))
        return 1
    prompt_text = prompts[0]["text"]
    detail["prompt_text"] = prompt_text
    print(f"[{STAGE}] real prompt: {prompt_text!r}")

    try:
        messages = [{"role": "user", "content": prompt_text}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        torch.cuda.synchronize()
        gen_s = time.time() - t1

        new_tokens = out[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        detail["generated_text"] = generated_text
        detail["generation_seconds"] = gen_s
        print(f"[{STAGE}] generated in {gen_s:.2f}s: {generated_text!r}")

        sane, reason = looks_sane(generated_text)
        detail["looks_sane"] = sane
        detail["sanity_reason"] = reason
        if sane:
            emit(ok(STAGE, detail))
            return 0
        else:
            emit(fail(STAGE, f"generation ran without raising but the output "
                              f"failed the sanity check: {reason}", detail))
            return 1
    except Exception as e:
        print(f"[{STAGE}] forward/generate raised: {e!r}")
        traceback.print_exc()
        detail["exception"] = repr(e)
        detail["traceback"] = traceback.format_exc()
        emit(fail(STAGE, "running the model forward/generate over a real prompt "
                          "raised an exception", detail))
        return 1


if __name__ == "__main__":
    sys.exit(main())
