"""Auditor-only restore-field mutants for the A0 re-audit (never committed). usage: a0_mutant.py <sslm_abi.cpp> <name>"""
import sys
M = {
 "restore-drop-rope-k-count": ("prefix->state.rope_k_saturation_count = f.header.rope_k_saturation_count;",
                               "prefix->state.rope_k_saturation_count = 0;  // A0 MUTANT"),
 "restore-swap-q-k-counts": ("prefix->state.rope_q_saturation_count = f.header.rope_q_saturation_count;\n\tprefix->state.rope_k_saturation_count = f.header.rope_k_saturation_count;",
                             "prefix->state.rope_q_saturation_count = f.header.rope_k_saturation_count;  // A0 MUTANT\n\tprefix->state.rope_k_saturation_count = f.header.rope_q_saturation_count;"),
 "restore-drop-scale-e": ("prefix->state.hidden_scale = superslm::CarriedScale{m, e};",
                          "prefix->state.hidden_scale = superslm::CarriedScale{m, 0};  // A0 MUTANT"),
 "restore-drop-residual": ("if (residual) std::memcpy(prefix->hidden_codes_storage.data(), f.residual, residual);",
                           "(void)residual;  // A0 MUTANT"),
 "restore-walk-zero": ("prefix->dfa_walk_state = walk;", "prefix->dfa_walk_state = 0;  // A0 MUTANT"),
}
path, name = sys.argv[1], sys.argv[2]
old, new = M[name]
s = open(path, newline="").read()
assert s.count(old) == 1, (name, s.count(old))
open(path, "w", newline="").write(s.replace(old, new)); print("applied", name)
