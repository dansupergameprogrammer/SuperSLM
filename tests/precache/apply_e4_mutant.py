"""The named mutants and test-side controls of the E4 cells (plan §7 6.6, 10.1, 10.2; §9 E4).

usage: apply_e4_mutant.py <source root> <mutant>

Applies one mutant to a COPY of the source tree in place, by exact single-occurrence replacement (it refuses when the
anchor text is missing or repeated). Build superslm_precache_e1 from that copy and run the cells named below; never
apply it to a checkout you commit from. Engine mutants change src/; controls change the cell itself, to show that the
assertion they target is live.

| mutant                | where                       | cells that must go red                                  |
|-----------------------|-----------------------------|---------------------------------------------------------|
| hint-s-at-2B          | engine, restore hint        | 10.2/E4 (restore phase): the share s and the draw       |
| hint-share-ceil       | engine, restore hint        | 10.2/E4 (restore phase): s = ceil(origin/B) shares the partial tail page |
| matvec-last-short     | engine, parallel matvec     | 6.6/E4 (pinned) and 6.6/E4:rule-b (real geometry)       |
| geometry-pinned-16    | control, RequireRealRuleB   | every rule-B cell: the engine reads B = 16, not the rule |
| restore-phase-prefill | control, 10.1 restore phase | 10.1/E4 (restore phase): the prefix-prefill counter reads 1 |
"""
import sys

M = {
    # 10.2's named mutant (§7 10.2): the restore's hint computes s at a B other than the pool's (twice the pool's).
    "hint-s-at-2B": ("src/sslm_abi.cpp", [(
        "\t\tshare = static_cast<uint32_t>(origin / B);",
        "\t\tshare = static_cast<uint32_t>(origin / (2 * B));  // MUTANT hint-s-at-2B")]),
    # The same family's second form: the parent's partial tail page shared too.
    "hint-share-ceil": ("src/sslm_abi.cpp", [(
        "\t\tshare = static_cast<uint32_t>(origin / B);",
        "\t\tshare = static_cast<uint32_t>((origin + B - 1) / B);  // MUTANT hint-share-ceil")]),
    # 6.6's mutant (rev 1.18.4, suite audit G9): the last matvec task's row range ends one row short, so that row keeps
    # its stale value on the parallel path only.
    "matvec-last-short": ("src/forward/parallel_split.cpp", [(
        "\tconst size_t end = std::min(c.total_rows, begin + c.rows_per_task);",
        "\tconst size_t full_end = std::min(c.total_rows, begin + c.rows_per_task);\n"
        "\tconst size_t end = (full_end == c.total_rows && full_end > begin + 1) ? full_end - 1 : full_end;"
        "  // MUTANT matvec-last-short")]),
    # Control: the engine pinned to the pre-rule B = 16 (the geometry e4_real.cpp asserted at 3b6631f). The re-derived
    # assertion must refuse it.
    "geometry-pinned-16": ("tests/precache/e4_real.cpp", [(
        "    const int64_t rule=E4RuleB(fx.geo.head_dim,static_cast<uint64_t>(fx.geo.context_cap),1);",
        "    KvPagesTestPinPagePositions(fx.model,16);  // CONTROL geometry-pinned-16\n"
        "    const int64_t rule=E4RuleB(fx.geo.head_dim,static_cast<uint64_t>(fx.geo.context_cap),1);")]),
    # Control: one prefix prefill in the restore process. 10.1's "no prefix prefill ran" must refuse it.
    "restore-phase-prefill": ("tests/precache/e4_real.cpp", [(
        "            if(!Restored(fx,&pool.pool,b,nullptr,h,&p)) return;",
        "            if(!Restored(fx,&pool.pool,b,nullptr,h,&p)) return;\n"
        "            if(i==0) {  // CONTROL restore-phase-prefill\n"
        "                sslm_prefix extra=nullptr;\n"
        "                if(sslm_prefix_begin_budgeted(fx.model,&pool.pool,1,&extra)==SSLM_OK) {\n"
        "                    int32_t consumed=0;\n"
        "                    sslm_prefix_prefill(fx.model,extra,ids.data(),1,1,SSLM_SPAN_PROMPT,nullptr,&consumed);\n"
        "                    sslm_prefix_release(extra);\n"
        "                }\n"
        "            }")]),
}

if __name__ == "__main__":
    root, name = sys.argv[1], sys.argv[2]
    rel, edits = M[name]
    path = f"{root}/{rel}"
    src = open(path, encoding="utf-8", newline="").read()
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            sys.exit(f"{name}: expected one occurrence in {rel}, found {n}")
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(src)
    print("applied", name, "to", rel)
