// MR2 red suite, fixture design aid for cells 6.6 and 6.7 (not a cell; grades nothing). For one model and one
// prefix it runs every vocabulary token as the next row on the one-row path, one layer per sslm_decode_step_gpu
// call, and prints what that row does:
//   tok <t> refused_at_layer <l|-1> d_kv_sat <n> d_kv_landing <n> d_k_channel <n> d_rope_q <n> d_rope_k <n>
// (the deltas are the five SequenceLayerState counters across the row). mr2_fixtures.py's patch parameters and the
// cells' trap tokens were chosen from this output; the cells re-derive the behaviour they rely on at run time.
// Usage: mr2_fixture_probe.exe --model=PATH [--prefix=N] [--tokens=a,b,...]
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mr2_common.h"

int main(int argc, char** argv) {
	std::string model;
	int prefix = 3;
	std::vector<int32_t> only;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		if (s.rfind("--model=", 0) == 0) model = s.substr(8);
		else if (s.rfind("--prefix=", 0) == 0) prefix = std::atoi(s.c_str() + 9);
		else if (s.rfind("--tokens=", 0) == 0) {
			for (size_t p = 9; p < s.size();) {
				const size_t q = s.find(',', p);
				only.push_back(std::atoi(s.substr(p, q - p).c_str()));
				p = q == std::string::npos ? s.size() : q + 1;
			}
		}
	}
	mr2::Model m;
	if (!m.Load(model)) return 1;
	mr2::World w(m);
	if (!w.ok) return 1;
	const int32_t vocab = m.vocab();
	if (only.empty())
		for (int32_t t = 0; t < vocab; ++t) only.push_back(t);
	for (int32_t t : only) {
		SslmGpuSequenceHandle* s = w.Dedicated(256);
		if (!s) return 1;
		for (int i = 0; i < prefix; ++i)
			if (mr2::FeedRow(w, s, mr2::StreamToken(i, vocab), m.full_budget(), m.layers()) != SslmGpuStatus::SSLM_OK) {
				std::printf("prefix row %d refused\n", i);
				return 1;
			}
		const mr2::Counters before = mr2::ReadCounters(s);
		int refused = -1;
		SslmGpuStatus st = sslm_gpu_seq_embed_token(w.ctx, s, t);
		for (uint32_t l = 0; st == SslmGpuStatus::SSLM_OK && l < m.layers(); ++l) {
			st = mr2::Slice(w, s, m.dpl());
			if (st == SslmGpuStatus::SSLM_SEQUENCE_REJECTED) refused = static_cast<int>(l);
		}
		const mr2::Counters d = mr2::ReadCounters(s) - before;
		std::printf("tok %d refused_at_layer %d d_kv_sat %llu d_kv_landing %llu d_k_channel %llu d_rope_q %llu d_rope_k %llu%s\n",
		            t, refused, (unsigned long long)d.v[0], (unsigned long long)d.v[1], (unsigned long long)d.v[2],
		            (unsigned long long)d.v[3], (unsigned long long)d.v[4],
		            st != SslmGpuStatus::SSLM_OK && st != SslmGpuStatus::SSLM_SEQUENCE_REJECTED ? " (other status)" : "");
		w.Release(s);
	}
	return 0;
}
