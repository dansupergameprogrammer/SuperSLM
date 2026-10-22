"""The named E2 mutants of the CPU restore (T-3060 F3, F6, F7 and F14; plan §9 E2 receipt (3)).

usage: apply_e2_mutant.py <engine source root> <mutant>
       apply_e2_mutant.py --list

Applies one mutant to src/spx1_format.cpp or src/sslm_abi.cpp in place, by exact single-occurrence replacement (it
refuses when the anchor text is missing or repeated). Build superslm_precache_e1, run the cells, then restore with
`git checkout -- src`. Never commit a mutated tree. tests/precache/run_e2_mutants.sh runs the whole set.

F14, one check-removal mutant per clause of the SPX1 reader's validation (spx1::Read), in its order:
  no-size no-magic no-digest no-backend no-producer-version no-producer-adapter no-model no-kv-precision
  no-schema-unresolved no-schema-walk no-unbound-walk no-length-negative no-length-cap no-size-exact
  no-origin-negative no-origin-past-length no-token-empty no-token-negative no-token-vocab no-scale-m
F6:  step45-swap               the plan's step-4/5 swap: the producer check (§3.4 step 5) before the backend's (step 4)
     step78-swap               the domain checks (step 8) before the schema check (step 7)
F7:  first-compatible-schema   the schema resolved as the first one whose states admit the walk, not by name
F3:  draw-first-and-return     restore draws a page before validating and returns it, so a zero-free pool answers
                               every input with its capacity status
"""
import sys

SPX1 = "src/spx1_format.cpp"
ABI = "src/sslm_abi.cpp"

SCHEMA_BLOCK = """	const uint64_t schema_hash = Get64(p, 96);
	const uint32_t walk = Get32(p, 104);
	int32_t schema_index = -1;
	if (schema_hash != 0) {
		size_t idx = 0;
		const SchemaEntry* entry = x.schemas != nullptr ? x.schemas->ByNameHash(schema_hash, &idx) : nullptr;
		if (entry == nullptr || walk >= entry->state_count) return Verdict::kSchema;
		schema_index = static_cast<int32_t>(idx);
	} else if (walk != kWalkUnbound) {
		return Verdict::kSchema;
	}
"""
BACKEND = "\tif (Get32(p, 4) != x.backend) return Verdict::kBackend;\n"
PRODUCER = """	// The producer: engine version, adapter vendor and device.
	uint32_t vendor = 0, device = 0;
	if (x.adapter_ids != nullptr && !x.adapter_ids(x.adapter_user, &vendor, &device)) {
		return Verdict::kAdapterUnavailable;
	}
	if (Get32(p, 8) != SUPERSLM_VERSION_MAJOR || Get32(p, 12) != SUPERSLM_VERSION_MINOR ||
	    Get32(p, 16) != SUPERSLM_VERSION_PATCH || Get32(p, 20) != vendor || Get32(p, 24) != device) {
		return Verdict::kProducer;
	}
"""
DOMAIN_BLOCK = """	const int64_t length = static_cast<int64_t>(Get64(p, 112));
	size_t required = 0;
	if (length < 0 || length > x.context_cap ||
	    !BlobSize(x.runs, x.head_dim, x.hidden_size, static_cast<uint64_t>(length), &required) || n != required) {
		return Verdict::kDomain;
	}
	const int64_t origin = static_cast<int64_t>(Get64(p, 120));
	if (origin < 0 || origin > length) return Verdict::kDomain;
	const int32_t token = static_cast<int32_t>(Get32(p, 108));
	if ((length == 0 && token != -1) ||
	    (length > 0 && (token < 0 || static_cast<uint64_t>(token) >= x.vocab_size))) {
		return Verdict::kDomain;
	}
	const int64_t m = static_cast<int64_t>(Get64(p, 128));
	if (m < INT32_MIN || m > INT32_MAX) return Verdict::kDomain;
"""
SCHEMA_DOMAIN = SCHEMA_BLOCK + """	// The domains, in order: L within the cap before any arithmetic on it, the exact size, the origin, the
	// token, and the hidden scale's m within int32.
""" + DOMAIN_BLOCK

M = {
    "no-size": (SPX1, "if (n < kHeaderBytes || std::memcmp(p, kMagic, 4) != 0) return Verdict::kSizeOrMagic;",
                "if (std::memcmp(p, kMagic, 4) != 0) return Verdict::kSizeOrMagic;  // MUTANT no-size"),
    "no-magic": (SPX1, "if (n < kHeaderBytes || std::memcmp(p, kMagic, 4) != 0) return Verdict::kSizeOrMagic;",
                 "if (n < kHeaderBytes) return Verdict::kSizeOrMagic;  // MUTANT no-magic"),
    "no-digest": (SPX1, "if (std::memcmp(digest, p + kDigestOffset, kDigestBytes) != 0) return Verdict::kDigest;",
                  "(void)digest;  // MUTANT no-digest"),
    "no-backend": (SPX1, "if (Get32(p, 4) != x.backend) return Verdict::kBackend;", "// MUTANT no-backend"),
    "no-producer-version": (SPX1,
        """	if (Get32(p, 8) != SUPERSLM_VERSION_MAJOR || Get32(p, 12) != SUPERSLM_VERSION_MINOR ||
	    Get32(p, 16) != SUPERSLM_VERSION_PATCH || Get32(p, 20) != vendor || Get32(p, 24) != device) {""",
        """	if (Get32(p, 20) != vendor || Get32(p, 24) != device) {  // MUTANT no-producer-version"""),
    "no-producer-adapter": (SPX1,
        """	if (Get32(p, 8) != SUPERSLM_VERSION_MAJOR || Get32(p, 12) != SUPERSLM_VERSION_MINOR ||
	    Get32(p, 16) != SUPERSLM_VERSION_PATCH || Get32(p, 20) != vendor || Get32(p, 24) != device) {""",
        """	if (Get32(p, 8) != SUPERSLM_VERSION_MAJOR || Get32(p, 12) != SUPERSLM_VERSION_MINOR ||
	    Get32(p, 16) != SUPERSLM_VERSION_PATCH) {  // MUTANT no-producer-adapter"""),
    "no-model": (SPX1, "if (std::memcmp(p + 60, x.model_hash, 32) != 0) return Verdict::kModel;", "// MUTANT no-model"),
    "no-kv-precision": (SPX1, "if (Get32(p, 92) != x.kv_precision) return Verdict::kKvPrecision;",
                        "// MUTANT no-kv-precision"),
    "no-schema-unresolved": (SPX1,
        "if (entry == nullptr || walk >= entry->state_count) return Verdict::kSchema;",
        "if (entry != nullptr && walk >= entry->state_count) return Verdict::kSchema;  // MUTANT no-schema-unresolved\n"
        "\t\tif (entry == nullptr) idx = 0;"),
    "no-schema-walk": (SPX1, "if (entry == nullptr || walk >= entry->state_count) return Verdict::kSchema;",
                       "if (entry == nullptr) return Verdict::kSchema;  // MUTANT no-schema-walk"),
    "no-unbound-walk": (SPX1, """	} else if (walk != kWalkUnbound) {
		return Verdict::kSchema;
	}""", """	}  // MUTANT no-unbound-walk"""),
    "no-length-negative": (SPX1, "if (length < 0 || length > x.context_cap ||",
                           "if (length > x.context_cap ||  // MUTANT no-length-negative"),
    "no-length-cap": (SPX1, "if (length < 0 || length > x.context_cap ||", "if (length < 0 ||  // MUTANT no-length-cap"),
    "no-size-exact": (SPX1, "static_cast<uint64_t>(length), &required) || n != required) {",
                      "static_cast<uint64_t>(length), &required) || n < required) {  // MUTANT no-size-exact"),
    "no-origin-negative": (SPX1, "if (origin < 0 || origin > length) return Verdict::kDomain;",
                           "if (origin > length) return Verdict::kDomain;  // MUTANT no-origin-negative"),
    "no-origin-past-length": (SPX1, "if (origin < 0 || origin > length) return Verdict::kDomain;",
                              "if (origin < 0) return Verdict::kDomain;  // MUTANT no-origin-past-length"),
    "no-token-empty": (SPX1, "if ((length == 0 && token != -1) ||",
                       "if (false ||  // MUTANT no-token-empty"),
    "no-token-negative": (SPX1, "(length > 0 && (token < 0 || static_cast<uint64_t>(token) >= x.vocab_size))) {",
                          "(length > 0 && token >= 0 && static_cast<uint64_t>(token) >= x.vocab_size)) {  // MUTANT no-token-negative"),
    "no-token-vocab": (SPX1, "(length > 0 && (token < 0 || static_cast<uint64_t>(token) >= x.vocab_size))) {",
                       "(length > 0 && token < 0)) {  // MUTANT no-token-vocab"),
    "no-scale-m": (SPX1, "if (m < INT32_MIN || m > INT32_MAX) return Verdict::kDomain;", "// MUTANT no-scale-m"),
    "step45-swap": (SPX1, BACKEND + PRODUCER,
                    PRODUCER + "\t// MUTANT step45-swap: the producer (step 5) before the backend (step 4)\n" + BACKEND),
    "step78-swap": (SPX1, SCHEMA_DOMAIN, "\t// MUTANT step78-swap: the domains (step 8) before the schema (step 7)\n" + DOMAIN_BLOCK + SCHEMA_BLOCK),
    "first-compatible-schema": (SPX1,
        "const SchemaEntry* entry = x.schemas != nullptr ? x.schemas->ByNameHash(schema_hash, &idx) : nullptr;",
        "const SchemaEntry* entry = nullptr;  // MUTANT first-compatible-schema\n"
        "\t\tif (x.schemas != nullptr && x.schemas->ByNameHash(schema_hash) != nullptr)\n"
        "\t\t\tfor (size_t i = 0; (entry = x.schemas->ByIndex(i)) != nullptr; ++i)\n"
        "\t\t\t\tif (walk < entry->state_count) { idx = i; break; }"),
    "draw-first-and-return": (ABI,
        """	if (!model || !pool || !*pool || !buf || (*pool)->model != model) return SSLM_INVALID_ARGUMENT;
	const auto& c = model->view.config;
	const auto model_hash = model->view.RawIntegrityHash();""",
        """	if (!model || !pool || !*pool || !buf || (*pool)->model != model) return SSLM_INVALID_ARGUMENT;
	{  // MUTANT draw-first-and-return: one page drawn before validation, then returned
		superslm::kv_pages::PageHolder* early = nullptr;
		const sslm_status early_draw = CreateHolder(*pool, 1, &early);
		if (early_draw != SSLM_OK) return early_draw;
		ReleaseHolder(*pool, early);
	}
	const auto& c = model->view.config;
	const auto model_hash = model->view.RawIntegrityHash();"""),
}


def main() -> int:
    if sys.argv[1:] == ["--list"]:
        print("\n".join(M))
        return 0
    root, name = sys.argv[1], sys.argv[2]
    rel, old, new = M[name]
    path = f"{root}/{rel}"
    src = open(path, newline="").read()
    count = src.count(old)
    if count != 1:
        sys.exit(f"{name}: anchor found {count} times (want 1)")
    open(path, "w", newline="").write(src.replace(old, new))
    print("applied", name, "to", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
