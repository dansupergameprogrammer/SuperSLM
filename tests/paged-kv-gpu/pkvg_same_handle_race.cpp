// 1.13.0 contract fold (adversary strike B1/S1): the same-handle decode promise, raced for every decode
// entry point on a dedicated and a pooled handle.
//
// The promise (gpu_1p0.h, the decode calls' thread-safety note): two decode calls that race on one
// sequence handle -- sslm_decode_step_gpu, a slot of sslm_decode_step_batch_gpu, SslmGpuSeqDecodeStepForG5Bridge
// or SslmGpuSeqFinishTokenForG5Bridge, in any pairing -- stay memory-safe: one runs, and the other returns
// SSLM_BUSY (a batch slot reads SSLM_BUSY) having changed nothing.
//
// How the overlap is made certain. The engine's test seam SslmGpuSeqClaimHookForTest (test builds only,
// gpu_1p0.cpp) runs a hook on the winning thread the moment its call takes the handle's claim, before it
// reads or changes anything else. The hook parks the winner there; the main thread then makes every
// losing call while the winner is inside its call, and only then lets it go. A call that never takes the
// claim never parks, and the cell fails it ("returned without holding the claim").
//
// Per case (winner verb x holder): two handles S and T are built with the same history (T is the oracle).
//   - before:  blob0 = save(S); eligibility of T read by an unbind (SslmGpuSeqSetSchemaForG5Bridge(-1),
//              which changes nothing on an unbound handle and reads the bind permission);
//   - race:    the winner verb on S on a second thread, parked inside its claim; on the main thread the four
//              losers on S -- the bridge decode (a different token), sslm_decode_step_gpu, the finish, a
//              one-slot batch -- each must return SSLM_BUSY (the batch call SSLM_OK with slot SSLM_BUSY);
//              then save(S) must still equal blob0 and S's bind permission must equal T's (a loser that
//              cleared bind_eligible, or embedded its token, or reset the walk, fails here);
//   - after:   the winner returns; the same verb run serially on T must give the same status and token,
//              and S and T must save the same bytes (a winner that ran on a loser's residual fails here).
// Winner verbs: decode (embed, then sslm_decode_step_gpu, drained after), bridge (a fresh handle, bridge
// decode of a token), bridge-ready (after a 3-token prefill, the ready branch), finish (at full depth).
// Holders: dedicated (sslm_gpu_seq_create, cap 64) and pooled (budget 64 on one pool).
// Release: a bridge decode refused inside its drive leaves no claim behind (another thread's next calls are
// judged on the handle's state).
// Counters (strike S3): budgeted create and pooled release churned on two threads beside a live dedicated
// sequence keep the model's and the context's live counts exact (--counter-iterations=N, default 20000).
//
// Run: pkvg_same_handle_race.exe --fixture-dir=DIR   (python tools/gen_paged_kv_fixture.py DIR)
// Software GPU: tools/cloud_gpu/wine_gpu.sh exec <repo>/out/pkvg/pkvg_same_handle_race.exe --fixture-dir=DIR
#include "g1e_helpers.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

// The engine's test seam (gpu_1p0.cpp, SUPERSLM_GPU_ALLOC_FAULT_INJECTION builds only).
void SslmGpuSeqClaimHookForTest(void (*hook)(SslmGpuSequenceHandle* seq, void* user), void* user) noexcept;

using namespace g1e;

namespace {

struct Park {
	std::mutex m;
	std::condition_variable cv;
	SslmGpuSequenceHandle* target = nullptr;
	bool armed = false;
	bool entered = false;
	bool released = false;
	bool winner_done = false;
};

void ParkHook(SslmGpuSequenceHandle* seq, void* user) {
	Park* p = static_cast<Park*>(user);
	std::unique_lock<std::mutex> lk(p->m);
	if (!p->armed || seq != p->target) return;
	p->armed = false;
	p->entered = true;
	p->cv.notify_all();
	// Bounded, so a loser that blocks instead of returning cannot hang the cell for ever.
	p->cv.wait_for(lk, std::chrono::minutes(5), [&] { return p->released; });
}

enum class Winner { kDecode, kBridge, kBridgeReady, kFinish };
const char* WinnerName(Winner w) {
	switch (w) {
		case Winner::kDecode: return "decode";
		case Winner::kBridge: return "bridge";
		case Winner::kBridgeReady: return "bridge-ready";
		case Winner::kFinish: return "finish";
	}
	return "?";
}

constexpr int32_t kTokA = 5;
constexpr int32_t kTokB = 7;

// Brings a fresh handle to the case's starting history.
bool Prepare(const Env& e, SslmGpuSequenceHandle* h, Winner w) {
	GS st = GS::SSLM_OK;
	switch (w) {
		case Winner::kDecode: st = sslm_gpu_seq_embed_token(e.ctx, h, kTokA); break;
		case Winner::kBridge: break;
		case Winner::kBridgeReady: {
			const std::vector<int32_t> p = Stream(97, 3, kVocab);
			st = SslmGpuSeqPrefillPromptForG5Bridge(e.ctx, h, p.data(), 3, e.full_budget);
			break;
		}
		case Winner::kFinish:
			st = sslm_gpu_seq_embed_token(e.ctx, h, kTokA);
			if (st == GS::SSLM_OK) st = sslm_decode_step_gpu(e.ctx, h, nullptr, e.full_budget);
			if (st == GS::SSLM_OK) st = Drain(e, h);
			break;
	}
	CHECK_MSG(st == GS::SSLM_OK, "prepare %s: %s", WinnerName(w), StatusName(st));
	return st == GS::SSLM_OK;
}

struct Outcome {
	GS status = GS::SSLM_DEVICE_LOST;
	int32_t token = -1;
};

// The winner verb. kDecode's drain runs after the race, on the main thread.
Outcome RunWinner(const Env& e, SslmGpuSequenceHandle* h, Winner w) {
	Outcome o;
	switch (w) {
		case Winner::kDecode: o.status = sslm_decode_step_gpu(e.ctx, h, nullptr, e.full_budget); break;
		case Winner::kBridge:
		case Winner::kBridgeReady:
			o.status = SslmGpuSeqDecodeStepForG5Bridge(e.ctx, h, kTokA, e.full_budget, &o.token);
			break;
		case Winner::kFinish: o.status = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, h, &o.token); break;
	}
	return o;
}

// The bind permission, read by an unbind: SSLM_OK while eligible, SSLM_SEQUENCE_REJECTED after a call
// that revoked it. Changes nothing on an unbound handle (every handle here is unbound).
GS Eligibility(const Env& e, SslmGpuSequenceHandle* h) { return SslmGpuSeqSetSchemaForG5Bridge(e.ctx, h, -1); }

void RunCase(const Env& e, Park& park, SslmGpuSequenceHandle* s, SslmGpuSequenceHandle* t, Winner w, const char* holder) {
	char what[96];
	std::snprintf(what, sizeof what, "race %s/%s", holder, WinnerName(w));
	if (!Prepare(e, s, w) || !Prepare(e, t, w)) return;
	const std::vector<uint8_t> blob0 = Save(e, s);
	const GS elig0 = Eligibility(e, t);
	CHECK_MSG(blob0 == Save(e, t), "%s: S and T do not start from the same bytes", what);

	{
		std::lock_guard<std::mutex> lk(park.m);
		park.target = s;
		park.armed = true;
		park.entered = park.released = park.winner_done = false;
	}
	Outcome won;
	std::thread winner([&] {
		won = RunWinner(e, s, w);
		std::lock_guard<std::mutex> lk(park.m);
		park.winner_done = true;
		park.cv.notify_all();
	});
	bool parked = false;
	{
		std::unique_lock<std::mutex> lk(park.m);
		park.cv.wait_for(lk, std::chrono::minutes(5), [&] { return park.entered || park.winner_done; });
		parked = park.entered;
		park.armed = false;
	}
	CHECK_MSG(parked, "%s: the winner's call returned without holding the handle's claim (status %s)", what,
	          StatusName(won.status));
	if (parked) {
		// The losers, every one inside the winner's call.
		int32_t tok = -1;
		const GS l_bridge = SslmGpuSeqDecodeStepForG5Bridge(e.ctx, s, kTokB, e.full_budget, &tok);
		CHECK_MSG(l_bridge == GS::SSLM_BUSY, "%s: losing bridge decode returned %s (token %d), want SSLM_BUSY", what,
		          StatusName(l_bridge), tok);
		const GS l_decode = sslm_decode_step_gpu(e.ctx, s, nullptr, e.full_budget);
		CHECK_MSG(l_decode == GS::SSLM_BUSY, "%s: losing sslm_decode_step_gpu returned %s, want SSLM_BUSY", what,
		          StatusName(l_decode));
		if (l_decode == GS::SSLM_OK) (void)Drain(e, s);
		tok = -1;
		const GS l_finish = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, s, &tok);
		CHECK_MSG(l_finish == GS::SSLM_BUSY, "%s: losing finish returned %s (token %d), want SSLM_BUSY", what,
		          StatusName(l_finish), tok);
		SslmGpuSequenceHandle* one[1] = {s};
		GS slot = GS::SSLM_DEVICE_LOST;
		const GS l_batch = sslm_decode_step_batch_gpu(e.ctx, one, nullptr, 1, e.full_budget, &slot);
		CHECK_MSG(l_batch == GS::SSLM_OK && slot == GS::SSLM_BUSY,
		          "%s: losing batch returned %s with slot %s, want SSLM_OK with slot SSLM_BUSY", what, StatusName(l_batch),
		          StatusName(slot));
		if (slot == GS::SSLM_OK) (void)Drain(e, s);
		// Changed nothing: the bytes and the bind permission, read while the winner is still parked.
		const std::vector<uint8_t> blob1 = Save(e, s);
		CHECK_MSG(blob1 == blob0, "%s: the losing calls changed the handle's saved state (%zu vs %zu bytes)", what,
		          blob1.size(), blob0.size());
		const GS elig1 = Eligibility(e, s);
		CHECK_MSG(elig1 == elig0, "%s: the losing calls changed the bind permission (%s, the twin's %s)", what,
		          StatusName(elig1), StatusName(elig0));
	}
	{
		std::lock_guard<std::mutex> lk(park.m);
		park.released = true;
		park.cv.notify_all();
	}
	winner.join();
	if (w == Winner::kDecode && won.status == GS::SSLM_OK) {
		const GS d = Drain(e, s);
		CHECK_MSG(d == GS::SSLM_OK, "%s: the winner's drain: %s", what, StatusName(d));
	}

	// The winner against the serial oracle on T.
	Outcome want = RunWinner(e, t, w);
	if (w == Winner::kDecode && want.status == GS::SSLM_OK) (void)Drain(e, t);
	CHECK_MSG(want.status == GS::SSLM_OK, "%s: the oracle's serial call: %s", what, StatusName(want.status));
	CHECK_MSG(won.status == want.status && won.token == want.token,
	          "%s: the winner returned %s token %d, the serial oracle %s token %d", what, StatusName(won.status), won.token,
	          StatusName(want.status), want.token);
	const std::vector<uint8_t> sb = Save(e, s), tb = Save(e, t);
	CHECK_MSG(!sb.empty() && sb == tb, "%s: after the race S saves different bytes from the serial oracle T (%zu vs %zu)",
	          what, sb.size(), tb.size());
	std::printf("%s: winner %s token %d, losers %s\n", what, StatusName(won.status), won.token,
	            parked ? "all inside its claim" : "NOT inside its claim");
}

// The claim never outlives a call, on a refusal path either: a bridge decode refused inside its drive
// (a budget below one layer) leaves the handle claimable, so calls from another thread are judged on the
// handle's state, never refused SSLM_BUSY by a claim left behind.
void ReleaseAfterRefusal(const Env& e, SslmGpuSequenceHandle* s, const char* holder) {
	int32_t tok = -1;
	const GS refused = SslmGpuSeqDecodeStepForG5Bridge(e.ctx, s, kTokA, 1u, &tok);
	CHECK_MSG(refused == GS::SSLM_DISPATCH_BUDGET_TOO_SMALL, "release %s: a one-dispatch bridge decode returned %s",
	          holder, StatusName(refused));
	GS finish = GS::SSLM_DEVICE_LOST, bridge = GS::SSLM_DEVICE_LOST;
	int32_t ftok = -1, btok = -1;
	std::thread other([&] {
		finish = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, s, &ftok);
		bridge = SslmGpuSeqDecodeStepForG5Bridge(e.ctx, s, kTokA, e.full_budget, &btok);
	});
	other.join();
	CHECK_MSG(finish == GS::SSLM_SEQUENCE_REJECTED, "release %s: another thread's finish after the refusal returned %s, "
	          "want SSLM_SEQUENCE_REJECTED (not at full depth)", holder, StatusName(finish));
	CHECK_MSG(bridge == GS::SSLM_OK && btok >= 0, "release %s: another thread's bridge decode after the refusal returned %s "
	          "token %d", holder, StatusName(bridge), btok);
	std::printf("release %s: refused %s, then another thread's finish %s and bridge %s\n", holder, StatusName(refused),
	            StatusName(finish), StatusName(bridge));
}

// Strike S3: the context's and the model's live-handle counts, changed by verbs on distinct handles that
// the header lets run concurrently (budgeted create and pooled release on two threads). A dedicated
// sequence D lives throughout. After both threads finish and the pool is destroyed, the model must still
// refuse its unmap (D is live: a lost increment, clamped at zero on the way down, lets it through -- the
// use-after-free the strike describes); once D is released the unmap and the context's destroy must
// succeed (a lost decrement refuses them for ever).
void Counters(const std::string& path, int iterations) {
	SslmGpuContext* ctx = nullptr;
	const GS cs = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
	CHECK_MSG(cs == GS::SSLM_OK && ctx, "counters: context: %s", StatusName(cs));
	if (!ctx) return;
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "counters: fixture did not open");
		sslm_gpu_context_destroy(ctx);
		return;
	}
	SslmGpuSequenceHandle* d = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &d) == GS::SSLM_OK);
	SslmGpuKvPoolHandle* pool = nullptr;
	CHECK(sslm_gpu_kv_pool_create(ctx, fx.model, 16, &pool) == GS::SSLM_OK);
	std::atomic<int> refused{0};
	std::atomic<bool> go{false};
	auto churn = [&] {
		while (!go.load(std::memory_order_acquire)) {
		}
		for (int i = 0; i < iterations; ++i) {
			SslmGpuSequenceHandle* h = nullptr;
			if (sslm_gpu_seq_create_budgeted(ctx, fx.model, pool, 16, &h) != GS::SSLM_OK || !h) {
				refused.fetch_add(1);
				continue;
			}
			if (sslm_gpu_seq_release(ctx, h) != GS::SSLM_OK) refused.fetch_add(1);
		}
	};
	std::thread a(churn), b(churn);
	go.store(true, std::memory_order_release);
	a.join();
	b.join();
	CHECK_MSG(refused.load() == 0, "counters: %d budgeted creates or releases refused", refused.load());
	CHECK(sslm_gpu_kv_pool_destroy(ctx, pool) == GS::SSLM_OK);
	const GS early = sslm_gpu_model_unmap(ctx, fx.model);
	CHECK_MSG(early == GS::SSLM_MODEL_HAS_LIVE_SEQUENCES,
	          "counters: the model unmap with a dedicated sequence still live returned %s (a lost increment)", StatusName(early));
	if (early == GS::SSLM_OK) {
		fx.model = nullptr;  // freed under a live sequence; nothing more is safe on this context
		std::printf("counters: model unmapped under a live sequence\n");
		return;
	}
	CHECK(sslm_gpu_seq_release(ctx, d) == GS::SSLM_OK);
	const GS unmap = sslm_gpu_model_unmap(ctx, fx.model);
	CHECK_MSG(unmap == GS::SSLM_OK, "counters: the model unmap after every sequence was released returned %s (a lost "
	          "decrement)", StatusName(unmap));
	if (unmap == GS::SSLM_OK) fx.model = nullptr;
	const GS destroy = sslm_gpu_context_destroy(ctx);
	CHECK_MSG(destroy == GS::SSLM_OK, "counters: the context destroy returned %s (a lost decrement)", StatusName(destroy));
	std::printf("counters: 2 x %d budgeted create + release beside a live dedicated sequence; unmap %s, then %s; destroy %s\n",
	            iterations, StatusName(early), StatusName(unmap), StatusName(destroy));
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			Park park;
			SslmGpuSeqClaimHookForTest(&ParkHook, &park);
			const Winner winners[] = {Winner::kDecode, Winner::kBridge, Winner::kBridgeReady, Winner::kFinish};
			for (Winner w : winners) {
				SslmGpuSequenceHandle* s = nullptr;
				SslmGpuSequenceHandle* t = nullptr;
				CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 64, &s) == GS::SSLM_OK);
				CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 64, &t) == GS::SSLM_OK);
				if (s && t) RunCase(e, park, s, t, w, "dedicated");
				if (s) CHECK_MSG(sslm_gpu_seq_release(e.ctx, s) == GS::SSLM_OK, "dedicated/%s: release S", WinnerName(w));
				if (t) CHECK_MSG(sslm_gpu_seq_release(e.ctx, t) == GS::SSLM_OK, "dedicated/%s: release T", WinnerName(w));
			}
			{
				SslmGpuSequenceHandle* s = nullptr;
				CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 64, &s) == GS::SSLM_OK);
				if (s) {
					ReleaseAfterRefusal(e, s, "dedicated");
					CHECK(sslm_gpu_seq_release(e.ctx, s) == GS::SSLM_OK);
				}
				Rig rig(e, static_cast<uint32_t>(e.geo.R(64)));
				GSeq p = rig.pool ? rig.Budgeted(64) : GSeq{};
				if (p.h) ReleaseAfterRefusal(e, p.h, "pooled");
			}
			for (Winner w : winners) {
				Rig rig(e, static_cast<uint32_t>(2 * e.geo.R(64)));
				if (!rig.pool) break;
				GSeq s = rig.Budgeted(64), t = rig.Budgeted(64);
				if (s.h && t.h) RunCase(e, park, s.h, t.h, w, "pooled");
			}
			SslmGpuSeqClaimHookForTest(nullptr, nullptr);
		}
	}
	{
		int iterations = 20000;
		for (int i = 1; i < argc; ++i)
			if (std::strncmp(argv[i], "--counter-iterations=", 21) == 0) iterations = std::atoi(argv[i] + 21);
		const std::string path = FixturePath(argc, argv);
		if (!path.empty()) Counters(path, iterations);
	}
	return FinishSuite("pkvg_same_handle_race");
}
