#include "shared_fixtures.h"

void ChainTraceSinkHookFn(const superslm::SslmChainTraceRecord* chain,
                           const superslm::SslmKvLandingTraceRecord* kv, void* user) {
	// trace_hook.h's own contract: exactly one of the two record pointers is
	// non-null per call, never both, never neither.
	CHECK_MSG((chain != nullptr) != (kv != nullptr),
	          "a trace hook call must carry exactly one non-null record pointer, never "
	          "both and never neither");
	if (chain == nullptr) return;
	auto* sink = static_cast<std::vector<ChainTraceSinkRecord>*>(user);
	ChainTraceSinkRecord rec;
	rec.site.assign(chain->site.begin(), chain->site.end());
	rec.token_index = chain->token_index;
	rec.x_int.assign(chain->x_int.begin(), chain->x_int.end());
	rec.d_prime = chain->d_prime;
	rec.dn = chain->dn;
	rec.s = chain->s;
	rec.r = chain->r;
	rec.codes.assign(chain->codes.begin(), chain->codes.end());
	rec.m_out = chain->m_out;
	rec.e_out = chain->e_out;
	sink->push_back(std::move(rec));
}
