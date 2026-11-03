// Schema fast-decode plan rev 1.4.4 §11.2, cells G6.1 to G6.5 (slice G1-M): the masked finish on the GPU.
//   G6.1  The host-head branch's legal-row finish equals its full finish on every F-SFD state (token and walk).
//   G6.2  The device-head branch equals its full finish on the tied fixture (host finish from embed_weights) and
//         on the untied variant (Q26 B: the device kernel's full finish).
//   G6.3  The masked finish is exact: bound decode on every state gives the full finish's token and walk, with
//         the head on the host and on the device (tied); a bound finish on a tied device-head model issues no
//         device-logits submission (S-GROWS, S-GSUB); unbound decode is unchanged.
//   G6.4  An untied head (F-SFD --untied, whose lm_head is the negated embedding, so the two heads choose
//         differently wherever a state's legal logits do not all tie): under Q26 B the bound finish runs the
//         device logits (S-GROWS), and its token equals the host legal-row finish over a host copy of the
//         variant's lm_head, so a build that finishes from embed_weights fails.
//   G6.5  Predicate false (S-GBOUND): bound decode scores all V rows on both branches, tokens unchanged.
// The full finish is the same model with S-GBOUND set. Box only. Needs fsfd.sslm and fsfd_untied.sslm.
// Run: sfdg_cell_g6.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

using namespace sfdg;

namespace {

// Every state of every schema, finished on `fast` and on `full` (the same model, S-GBOUND set): equal tokens and
// walks. Returns the per-finish S-GROWS entries of the fast run.
std::vector<RowsEntry> CompareEveryState(const char* cell, GRig& rig, SslmGpuModelHandle* fast, SslmGpuModelHandle* full,
                                         const std::vector<int32_t>& prompt, bool expect_device_fast,
                                         const superslm::SslmModelView& view) {
	std::vector<RowsEntry> fast_rows;
	CHECK_MSG(sslm_gpu_test_only_force_full_finish(full, 1) == GS::SSLM_OK, "[%s] S-GBOUND", cell);
	for (const char* schema : kSchemas) {
		const RefSchema& r = rig.ref[schema];
		GSeq a = rig.New(schema, fast), b = rig.New(schema, full);
		int bad = 0, rows_bad = 0;
		for (uint32_t s = 0; s < r.st.size() && a.h && b.h; ++s) {
			sslm_gpu_test_only_rows_log_reset();
			const FinishResult fa = FinishAt(rig, a, prompt, r.st[s].path);
			const std::vector<RowsEntry> la = RowsLog();
			const FinishResult fb = FinishAt(rig, b, prompt, r.st[s].path);
			if (fa.st != GS::SSLM_OK || fb.st != GS::SSLM_OK || fa.token != fb.token || fa.walk != fb.walk) {
				if (bad++ < 5)
					CHECK_MSG(false, "[%s] %s state %u: fast %s token %d walk %u; full %s token %d walk %u", cell, schema, s,
					          Name(fa.st), fa.token, fa.walk, Name(fb.st), fb.token, fb.walk);
			}
			// The finish's own S-GROWS entry is the last one of the fast run.
			if (!la.empty()) {
				const RowsEntry e = la.back();
				fast_rows.push_back(e);
				const uint64_t want = expect_device_fast ? uint64_t(view.config.vocab_size) : LegalCount(view, schema, s);
				if ((e.rows != want || e.device != (expect_device_fast ? 1 : 0)) && rows_bad++ < 5)
					CHECK_MSG(false, "[%s] %s state %u: S-GROWS rows %llu device %d, want %llu device %d", cell, schema, s,
					          static_cast<unsigned long long>(e.rows), e.device, static_cast<unsigned long long>(want),
					          expect_device_fast ? 1 : 0);
			} else {
				CHECK_MSG(rows_bad++ >= 5, "[%s] %s state %u: no S-GROWS entry", cell, schema, s);
			}
		}
		CHECK_MSG(bad == 0 && rows_bad == 0, "[%s] %s: %d token/walk and %d row-count differences over %zu states", cell,
		          schema, bad, rows_bad, r.st.size());
	}
	sslm_gpu_test_only_force_full_finish(full, 0);
	return fast_rows;
}

// A bound finish only (no layer work): drive to depth first, then count the finish's submissions.
uint64_t FinishSubmissions(GRig& rig, SslmGpuModelHandle* m, const std::vector<int32_t>& prompt, int32_t* token) {
	GSeq s = rig.New(kPotion, m);
	rig.Prompt(s, prompt);  // leaves the ready flag set: the next finish needs no layer work
	const uint64_t before = Submissions(rig.ctx());
	*token = -1;
	const GS st = SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), s.h, token);
	CHECK_MSG(st == GS::SSLM_OK, "[G6.3] the finish: %s", Name(st));
	return Submissions(rig.ctx()) - before;
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		// G6.1 and G6.3, host head (flags 0).
		Mapped host_fast, host_full;
		if (MapVariant(rig, dir + "/fsfd.sslm", 0, &host_fast) && MapVariant(rig, dir + "/fsfd.sslm", 0, &host_full)) {
			CompareEveryState("G6.1", rig, host_fast.m, host_full.m, prompt, false, host_fast.view);
			// G6.5 on the host branch: S-GBOUND on the fast model scores V rows, tokens unchanged.
			sslm_gpu_test_only_force_full_finish(host_fast.m, 1);
			GSeq s = rig.New(kPotion, host_fast.m);
			sslm_gpu_test_only_rows_log_reset();
			const FinishResult f = FinishAt(rig, s, prompt, {});
			const std::vector<RowsEntry> l = RowsLog();
			CHECK_MSG(f.st == GS::SSLM_OK && !l.empty() && l.back().rows == uint64_t(host_fast.view.config.vocab_size),
			          "[G6.5] host head under S-GBOUND: %s, rows %llu", Name(f.st),
			          l.empty() ? 0ull : static_cast<unsigned long long>(l.back().rows));
			sslm_gpu_test_only_force_full_finish(host_fast.m, 0);
		}
		// G6.2 and G6.3, device head, tied.
		Mapped dev_fast, dev_full;
		if (MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev_fast) &&
		    MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev_full)) {
			// The tied device-head model's bound finish is the host legal-row finish: rows = n, no device logits.
			CompareEveryState("G6.2/G6.3 tied device head", rig, dev_fast.m, dev_full.m, prompt, false, dev_fast.view);
			int32_t t_fast = -1, t_host = -1;
			const uint64_t n_fast = FinishSubmissions(rig, dev_fast.m, prompt, &t_fast);
			const uint64_t n_host = FinishSubmissions(rig, host_fast.m ? host_fast.m : dev_fast.m, prompt, &t_host);
			CHECK_MSG(n_fast == 0, "[G6.3] a bound finish on a tied device-head model made %llu submissions (want none)",
			          static_cast<unsigned long long>(n_fast));
			CHECK_MSG(t_fast == t_host, "[G6.3] tied device head %d, host head %d", t_fast, t_host);
			// Unbound decode is unchanged: the device logits still run, V rows.
			GSeq u = rig.New(nullptr, dev_fast.m);
			rig.Prompt(u, prompt);
			sslm_gpu_test_only_rows_log_reset();
			GS st = GS::SSLM_OK;
			const int32_t tu = Token(rig.env, u, &st);
			const std::vector<RowsEntry> l = RowsLog();
			CHECK_MSG(st == GS::SSLM_OK && tu >= 0 && !l.empty() && l.back().device == 1 &&
			              l.back().rows == uint64_t(dev_fast.view.config.vocab_size),
			          "[G6.3] unbound decode on the device head: %s token %d, S-GROWS device %d rows %llu", Name(st), tu,
			          l.empty() ? -1 : l.back().device, l.empty() ? 0ull : static_cast<unsigned long long>(l.back().rows));
			// G6.5 on the device branch.
			sslm_gpu_test_only_force_full_finish(dev_fast.m, 1);
			GSeq s = rig.New(kPotion, dev_fast.m);
			sslm_gpu_test_only_rows_log_reset();
			const FinishResult f = FinishAt(rig, s, prompt, {});
			const std::vector<RowsEntry> l5 = RowsLog();
			CHECK_MSG(f.st == GS::SSLM_OK && !l5.empty() && l5.back().rows == uint64_t(dev_fast.view.config.vocab_size) &&
			              l5.back().device == 1 && f.token == t_fast,
			          "[G6.5] device head under S-GBOUND: %s token %d (unforced %d)", Name(f.st), f.token, t_fast);
			sslm_gpu_test_only_force_full_finish(dev_fast.m, 0);
		}
		// G6.4, untied (Q26 B): the device-head bound finish runs the device logits, and equals the host legal-row
		// finish over the variant's own lm_head.
		Mapped un_dev, un_host;
		if (MapVariant(rig, dir + "/fsfd_untied.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &un_dev) &&
		    MapVariant(rig, dir + "/fsfd_untied.sslm", 0, &un_host)) {
			CHECK_MSG(!un_dev.view.config.tie_word_embeddings, "[G6.4] fsfd_untied.sslm is tied");
			int device_runs = 0, bad = 0, compared = 0, heads_differ = 0;
			for (const char* schema : {kPotion, kUtf8Enum}) {
				const RefSchema& r = rig.ref[schema];
				GSeq a = rig.New(schema, un_dev.m), b = rig.New(schema, un_host.m);
				GSeq tied;
				if (host_fast.m) tied = rig.New(schema, host_fast.m);  // the same layers, finished from embed_weights
				for (uint32_t s = 0; s < r.st.size(); ++s) {
					if (LegalCount(un_dev.view, schema, s) < 2) continue;  // a forced row: nothing to choose
					sslm_gpu_test_only_rows_log_reset();
					const FinishResult fa = FinishAt(rig, a, prompt, r.st[s].path);
					const std::vector<RowsEntry> l = RowsLog();
					if (!l.empty() && l.back().device == 1) ++device_runs;
					const FinishResult fb = FinishAt(rig, b, prompt, r.st[s].path);
					if (tied.h) {
						const FinishResult ft = FinishAt(rig, tied, prompt, r.st[s].path);
						if (ft.st == GS::SSLM_OK && ft.token != fb.token) ++heads_differ;
					}
					++compared;
					if ((fa.st != GS::SSLM_OK || fa.token != fb.token || fa.walk != fb.walk) && bad++ < 5)
						CHECK_MSG(false, "[G6.4] %s state %u: device head %s token %d, host lm_head %s token %d", schema, s,
						          Name(fa.st), fa.token, Name(fb.st), fb.token);
				}
			}
			CHECK_MSG(compared > 0 && bad == 0, "[G6.4] %d of %d choice states differ", bad, compared);
			CHECK_MSG(device_runs > 0, "[G6.4] under Q26 B the untied bound finish never ran the device logits");
			// The variant's precondition, at run time: a finish from embed_weights picks a different token at one
			// state or more, so the equality above is what tells the two heads apart.
			CHECK_MSG(heads_differ > 0, "[G6.4] precondition: the tied and untied heads agree at all %d states", compared);
		}
		rig.ReleaseSeqs();  // the sequences first, then the extra models
		for (Mapped* m : {&host_fast, &host_full, &dev_fast, &dev_full, &un_dev, &un_host}) Unmap(rig, m);
	}
	return Finish("sfdg_cell_g6");
}
