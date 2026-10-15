// Cell 7.9's must-accept constructions MA-1 and MA-2 (paged-KV plan rev 16.29: §29.3, §29.4, §30.5; D-SLM8471,
// D-SLM8472, D-SLM8479), slot R79A. Built as superslm_pkv_pair_ma (tests/paged-kv/paged_kv.cmake), linked against the
// production library superslm and nothing else from the engine. The commissioning seat's source: the timing instrument
// these constructions challenge (tools/run_paged_kv_rows.py, cell 7.9's row rule) is not part of it.
//
//   superslm_pkv_pair_ma identity ma1|ma2
//       Prints the construction's identity line and exits 0. Any other argument exits non-zero.
//   superslm_pkv_pair_ma ma1|ma2 CAP LAYERS HEADS DIM --blob B --model F --process K --pairs N --warmup W
//       The run mode: exactly 16 arguments counting the program name. The geometry is checked by the model load, as in
//       tools/paged_kv_pair.cpp's row79; the five flags are the ones tools/run_paged_kv_rows.py appends.
//
// Both constructions restore the row's one blob into two legacy holders, each in its own one-block pool of the
// production build's contiguous pool option (§21.2), arm a first, as row79 restores its paged arm first. Each times W
// untimed warm-up tokens per arm, then N pairs, each pair's order drawn from kPairOrderSeed + K, each token one
// sslm_decode_step timed alone on steady_clock, as row79 does. Stdout: the identity line, then one
// "PAIR <arm_a_ns> <arm_b_ns> <arm_a_token> <arm_b_token>" line per timed pair, arm a in the driver's paged column.
// No RESERVE line and no IDS line: neither arm is paged. Stderr: "MA offset_ns=<value> seed=<value>", the constants
// this run applies.
//
// MA-1, an A/A: the two arms are the same pool kind, restore the same blob and decode the same tokens, so the true
// per-token ratio is 1 and every process median is symmetric about zero.
//
// MA-2, a structural arm offset below the bar: arm a busy-waits kMa2OffsetNs inside its own timed interval, after its
// decode step returns, on every timed and warm-up token. Its time therefore carries a real, constant elapsed cost on
// the same clock the instrument reads, so s = tau_a / tau_b - 1 sits near kMa2OffsetNs / tau_b. The value is calibrated
// for row R4a (the 1.5B at widths 15,876 to 15,979), the only row MA-2 runs at (D-SLM8463's R79A set). At any other
// row the same nanoseconds are a different fraction of the token time: at the 0.5B's R2b they exceed the bar.
// The value and its executed PASS rate are the commissioning seat's T6a record
// (Wizard Claude/Popper/t3209-ma-construction-2026-10-10.md, C3).
//
// Both constants are compiled in (D-SLM8479), so a run takes no decision-bearing argument and the R79A hash tie
// certifies them with the bytes.
#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS) || defined(SUPERSLM_ENABLE_BAD_ALLOC_INJECTION) || \
    defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS) || defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) || \
    defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT) || \
    defined(SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING) || defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION) || \
    defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION) || defined(SUPERSLM_FORCE_SCALAR_MATMUL)
#error A C6 production pair must not carry a seam or injection definition.
#endif
// This translation unit's own compile only. The mutant library's definition is PRIVATE to superslm_mr_opt and never
// reaches here; what proves this executable linked the production library is its link step, read by
// tools/check_paged_kv_pair_link.py (plan §30.2).
#if defined(SUPERSLM_MUTANT_PKV_MR_OPT) || defined(PKV_PAIR_MR_OPT) || defined(PKV_C6_IDENTITY_PROBE)
#error The must-accept constructions are built from the production definitions only.
#endif
#include "pkv_budget_c_helpers.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace pkv;
using namespace pkv::bc;
using clock_type = std::chrono::steady_clock;

// The driver's CONSTRUCTION_IDENTITY["MA-1"] and ["MA-2"] (tools/run_paged_kv_rows.py); each string lives here and
// in the driver only (plan §30.7).
constexpr const char* kIdentityMa1 = "PKV_PAIR_MA1_V1 defs=none";
constexpr const char* kIdentityMa2 = "PKV_PAIR_MA2_V1 defs=none";
// MA-2's offset, in nanoseconds per arm-a token: 5.4 ms, 2.25% of the 240 ms contiguous decode token measured at the
// R4a width (T-3151 record, 1.5B at 15,876). C3 of the T6a record gives its selection and executed rates.
constexpr int64_t kMa2OffsetNs = 5400000;
// The pair-order seed; process K draws its pair orders from kPairOrderSeed + K, as row79 draws from SEED + K.
constexpr uint32_t kPairOrderSeed = 3209;

void Require(bool okay, const std::string& why) { if (!okay || State().failures) throw std::runtime_error(why); }
int Number(const char* s) { size_t at=0; int v=std::stoi(s,&at); Require(at==std::string(s).size(),"invalid integer"); return v; }

struct Construction { const char* identity; int64_t offset_ns; };
// ma1 or ma2; anything else is refused.
Construction Named(const std::string& name) {
	if (name == "ma1") return {kIdentityMa1, 0};
	if (name == "ma2") return {kIdentityMa2, kMa2OffsetNs};
	throw std::runtime_error("unknown construction " + name + " (ma1 or ma2)");
}

// One arm: a legacy holder in a one-block pool of the contiguous pool option (plan §21.2), sized by
// sslm_kv_pool_contiguous_size, as tools/paged_kv_pair.cpp's ContiguousPool.
struct ContiguousArm {
	AlignedBuf mem;sslm_kv_pool pool=nullptr;sslm_seq seq=nullptr;
	ContiguousArm(sslm_model model,const std::vector<uint8_t>& blob){
		const size_t size=sslm_kv_pool_contiguous_size(model,1);Require(size>0,"contiguous pool size failed");
		mem.Reset(size);Require(sslm_kv_pool_create_contiguous(model,mem.p,mem.n,1,&pool)==SSLM_OK,"contiguous pool failed");
		Require(sslm_seq_restore(model,&pool,blob.data(),blob.size(),&seq)==SSLM_OK,"contiguous restore failed");}
	~ContiguousArm(){if(seq)sslm_seq_release(seq);if(pool)sslm_kv_pool_destroy(pool);}
	ContiguousArm(const ContiguousArm&)=delete;ContiguousArm& operator=(const ContiguousArm&)=delete;
};

void Run(int argc,char** argv){
	Require(argc==16,"run-mode argument count (16 with the program name)");
	const Construction construction=Named(argv[1]);
	const char *blob_path=nullptr,*model_path=nullptr;int process=-1,pairs=-1,warmup=-1;
	for(int i=6;i+1<argc;i+=2){const std::string flag=argv[i];const char* value=argv[i+1];
		const char** text=flag=="--blob"?&blob_path:flag=="--model"?&model_path:nullptr;
		int* number=flag=="--process"?&process:flag=="--pairs"?&pairs:flag=="--warmup"?&warmup:nullptr;
		if(text){Require(*text==nullptr,"repeated flag "+flag);*text=value;}
		else if(number){Require(*number==-1,"repeated flag "+flag);*number=Number(value);}
		else throw std::runtime_error("unknown flag "+flag);}
	Require(blob_path&&model_path&&process>=1&&pairs>=1&&pairs<=508&&warmup>=0,"run-mode flags");
	const std::string model_file(model_path);const size_t cut=model_file.find_last_of("/\\");
	const std::string dir=cut==std::string::npos?".":model_file.substr(0,cut),name=cut==std::string::npos?model_file:model_file.substr(cut+1);
#if defined(_WIN32)
	Require(_putenv_s("SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR",dir.c_str())==0,"cannot set the artifact directory");
#else
	Require(setenv("SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR",dir.c_str(),1)==0,"cannot set the artifact directory");
#endif
	const Fixture& fx=GetArtifact(name.c_str(),Geometry{static_cast<uint32_t>(Number(argv[3])),static_cast<uint32_t>(Number(argv[4])),
		static_cast<uint32_t>(Number(argv[5])),Number(argv[2])});
	Require(fx.ok,"model did not load");
	std::vector<uint8_t> blob;Require(ReadAll(blob_path,&blob)&&!blob.empty(),std::string("missing blob: ")+blob_path);
	Require(IsMagic(blob,"SSB5"),"row blob is not a legacy SSB5");
	ContiguousArm a(fx.model,blob),b(fx.model,blob);
	// One decode step, timed alone; `spin_ns` > 0 busy-waits that long after the step returns, inside the interval.
	auto one=[&](sslm_seq s,int64_t spin_ns)->std::pair<double,int32_t>{sslm_decode_params p{};p.layer_budget=static_cast<int32_t>(fx.geo.layers);
		sslm_seq batch[1]={s};int32_t token=-1;const auto begin=clock_type::now();
		const sslm_status status=sslm_decode_step(fx.model,batch,1,&p,nullptr,&token);
		if(spin_ns>0){const auto until=clock_type::now()+std::chrono::nanoseconds(spin_ns);while(clock_type::now()<until){}}
		const double ns=std::chrono::duration<double,std::nano>(clock_type::now()-begin).count();
		Require(status==SSLM_OK&&token>=0,"one-call token decode failed");return {ns,token};};
	std::cerr<<"MA offset_ns="<<construction.offset_ns<<" seed="<<kPairOrderSeed<<'\n'<<std::flush;
	// identity first: the driver checks it against the construction before it reads any pair (79-7, D-SLM8466)
	std::cout<<construction.identity<<'\n';
	for(int i=0;i<warmup;++i){one(a.seq,construction.offset_ns);one(b.seq,0);}
	std::mt19937 random(kPairOrderSeed+static_cast<uint32_t>(process));
	for(int i=0;i<pairs;++i){std::pair<double,int32_t> x,y;
		if((random()&1)!=0){x=one(a.seq,construction.offset_ns);y=one(b.seq,0);}else{y=one(b.seq,0);x=one(a.seq,construction.offset_ns);}
		std::cout<<"PAIR "<<x.first<<' '<<y.first<<' '<<x.second<<' '<<y.second<<'\n';}
	std::cout<<std::flush;
}
}
int main(int argc,char** argv){
	try{Require(argc>=2,"mode required");const std::string mode=argv[1];
		if(mode=="identity"){Require(argc==3,"identity takes ma1 or ma2");std::cout<<Named(argv[2]).identity<<'\n'<<std::flush;}
		else if(mode=="ma1"||mode=="ma2")Run(argc,argv);
		else throw std::runtime_error("unknown mode "+mode);
		return 0;
	}catch(const std::exception& error){std::cerr<<"INFRASTRUCTURE: "<<error.what()<<'\n';return 40;}
}
