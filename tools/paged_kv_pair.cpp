// Production-library C6 helper: piece-wise SSB5 prefill and in-process decode pairs.
// Linked only to superslm. The Python driver owns manifests, locks and atomic records.
#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS) || defined(SUPERSLM_ENABLE_BAD_ALLOC_INJECTION) || \
    defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS) || defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) || \
    defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT) || \
    defined(SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING) || defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION) || \
    defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION) || defined(SUPERSLM_FORCE_SCALAR_MATMUL)
#error A C6 production pair must not carry a seam or injection definition.
#endif
#include "pkv_budget_c_helpers.h"
#include "pkv_scatter_reserve.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using namespace pkv;
using namespace pkv::bc;
using clock_type = std::chrono::steady_clock;
// The identity line each cell 7.9 construction prints first on stdout (D-SLM8466 (2)), so the driver's records show
// which construction ran without trusting its --construction label. tools/run_paged_kv_rows.py's CONSTRUCTION_IDENTITY
// holds the same strings and refuses any other (79-7). Each string lives in one C++ source and in the driver (plan
// section 30.7): production's and 7.9-MR-OPT's here, this source's row79 built as superslm_pkv_pair and as
// superslm_pkv_pair_mr_opt (PKV_PAIR_MR_OPT, against the mutant library); MA-1's and MA-2's in
// tools/paged_kv_pair_ma.cpp, the commissioning seat's superslm_pkv_pair_ma (plan section 29.4: linked against the
// production library only; PAIR lines only, no RESERVE and no IDS, since it has no paged arm).
constexpr const char* kIdentityProduction = "PKV_PAIR_PRODUCTION_V2 defs=none";
constexpr const char* kIdentityProductionProbe = "PKV_PAIR_PRODUCTION_V2 defs=PKV_C6_IDENTITY_PROBE";
constexpr const char* kIdentityMrOpt = "PKV_PAIR_MR_OPT_V1 defs=none";
// This build's own identity: row79's first line, and the `identity` mode's output.
const char* BuildIdentity(){
#if defined(PKV_C6_IDENTITY_PROBE)
	return kIdentityProductionProbe;
#elif defined(PKV_PAIR_MR_OPT)
	return kIdentityMrOpt;
#else
	return kIdentityProduction;
#endif
}

void Require(bool okay, const std::string& why) { if (!okay || State().failures) throw std::runtime_error(why); }
int Number(const char* s) { size_t at=0; int v=std::stoi(s,&at); Require(at==std::string(s).size(),"invalid integer"); return v; }
uint32_t Seed(const char* s) { Require(s[0]!='-',"seed out of range"); size_t at=0;
	const auto value=std::stoull(s,&at);Require(at==std::string(s).size()&&value<=std::numeric_limits<uint32_t>::max(),"seed out of range");
	return static_cast<uint32_t>(value); }
const Fixture& Model(const char* file,int cap,int layers,int heads,int dim) {
	const Fixture& fx=GetArtifact(file,Geometry{static_cast<uint32_t>(layers),static_cast<uint32_t>(heads),static_cast<uint32_t>(dim),cap});
	Require(fx.ok,"model did not load");return fx;
}
std::vector<uint8_t> Blob(const char* path) {
	std::vector<uint8_t> bytes;Require(ReadAll(path,&bytes)&&!bytes.empty(),std::string("missing blob: ")+path);return bytes;
}
void Write(const char* path,const std::vector<uint8_t>& bytes) {
	std::ofstream out(path,std::ios::binary|std::ios::trunc);Require(bool(out),"cannot open blob output");
	out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));Require(bool(out),"blob write failed");
}
void Prefill(int argc,char** argv) {
	// prefill MODEL CAP LAYERS HEADS DIM WIDTH START COUNT INPUT|- OUTPUT
	Require(argc==12,"prefill argument count");
	const Fixture& fx=Model(argv[2],Number(argv[3]),Number(argv[4]),Number(argv[5]),Number(argv[6]));
	const int width=Number(argv[7]),start=Number(argv[8]),count=Number(argv[9]);
	Require(width>512&&start>=0&&count>0&&count<=4096&&start+count<=width-512,"piece outside prompt or over 4096");
	LegacyPool pool(fx.model,1);Require(pool.status==SSLM_OK,"legacy pool failed");sslm_seq seq=nullptr;
	if(std::string(argv[10])=="-") Require(start==0&&sslm_seq_create(fx.model,&pool.pool,&seq)==SSLM_OK,"new sequence failed");
	else {auto blob=Blob(argv[10]);Require(sslm_seq_restore(fx.model,&pool.pool,blob.data(),blob.size(),&seq)==SSLM_OK,"restore previous piece failed");}
	const auto stream=Stream(79,width-512,kVocab);
	const std::vector<int32_t> part(stream.begin()+start,stream.begin()+start+count);
	Require(PrefillAll(fx.model,seq,part,kChunk)==SSLM_OK,"production prefill failed");
	Write(argv[11],Save(seq));Require(State().failures==0,"save failed");sslm_seq_release(seq);
	std::cout<<"PREFILL_OK start="<<start<<" count="<<count<<" model_sha="<<fx.sha<<"\n";
}
// The contiguous pool option (paged-KV plan §21.2): one page of `cap` positions per block, 1.11.0's layout, sized by
// sslm_kv_pool_contiguous_size.
struct ContiguousPool {
	AlignedBuf mem;sslm_kv_pool pool=nullptr;sslm_status status=SSLM_INVALID_ARGUMENT;
	ContiguousPool(sslm_model model,uint32_t blocks){const size_t size=sslm_kv_pool_contiguous_size(model,blocks);
		if(size){mem.Reset(size);status=sslm_kv_pool_create_contiguous(model,mem.p,mem.n,blocks,&pool);}}
	~ContiguousPool(){if(pool)sslm_kv_pool_destroy(pool);}
};
struct Arm {
	const Fixture& fx;std::unique_ptr<LegacyPool> legacy;std::unique_ptr<PagePool> page;std::unique_ptr<ContiguousPool> contiguous;
	sslm_prefix world=nullptr,persona=nullptr;sslm_seq seq=nullptr;
	explicit Arm(const Fixture& f):fx(f){}
	sslm_kv_pool* Pool() {return contiguous?&contiguous->pool:legacy?&legacy->pool:&page->pool;}
	~Arm(){if(seq)sslm_seq_release(seq);if(persona)sslm_prefix_release(persona);if(world)sslm_prefix_release(world);}
	void Restore(const char* path,int expected){legacy=std::make_unique<LegacyPool>(fx.model,1);Require(legacy->status==SSLM_OK,"pool failed");
		auto blob=Blob(path);Require(IsMagic(blob,"SSB5"),"prompt blob is not SSB5");
		Require(BlobLe64(blob,60)==expected,"restored prompt length differs from width minus 512");
		Require(sslm_seq_restore(fx.model,Pool(),blob.data(),blob.size(),&seq)==SSLM_OK,"prompt restore failed");}
	void Shared(){page=std::make_unique<PagePool>(fx.model,static_cast<uint32_t>(fx.R(1000)+fx.R(200)+fx.R(512)+8));
		Require(page->status==SSLM_OK,"shared pool failed");
		Require(sslm_prefix_begin_budgeted(fx.model,Pool(),1000,&world)==SSLM_OK,"world begin failed");
		Require(PrefixPrefillAll(fx.model,world,WorldTokens(),kChunk)==SSLM_OK,"world prefill failed");
		Require(sslm_prefix_freeze(world)==SSLM_OK,"world freeze failed");
		Require(sslm_prefix_begin_from(world,200,&persona)==SSLM_OK,"persona begin failed");
		Require(PrefixPrefillAll(fx.model,persona,PersonaTokens(0),kChunk)==SSLM_OK,"persona prefill failed");
		Require(sslm_prefix_freeze(persona)==SSLM_OK,"persona freeze failed");
		Require(sslm_seq_create_budgeted(fx.model,Pool(),512,&seq)==SSLM_OK,"budget sequence failed");
		Require(sslm_seq_adopt_prefix(seq,persona)==SSLM_OK,"persona adopt failed");}
	// 7.9b's arm two (D-SLM8438): a legacy holder in a contiguous-option pool, prefilled in-process (a legacy holder in
	// a page pool pages at the derived B, and copy adopt from a page pool into the option is refused, D-SLM8434).
	void Contiguous(){contiguous=std::make_unique<ContiguousPool>(fx.model,1);Require(contiguous->status==SSLM_OK,"contiguous pool failed");
		Require(sslm_seq_create(fx.model,Pool(),&seq)==SSLM_OK,"contiguous sequence failed");
		auto prompt=WorldTokens();auto persona_tokens=PersonaTokens(0);prompt.insert(prompt.end(),persona_tokens.begin(),persona_tokens.end());
		Require(PrefillAll(fx.model,seq,prompt,kChunk)==SSLM_OK,"contiguous prefill failed");}
	int32_t Ready(){sslm_status status=SSLM_OK;int32_t token=Token(fx.model,seq,&status);Require(status==SSLM_OK&&token>=0,"ready token failed");return token;}
	std::pair<double,int32_t> Decode(bool mutant=false){sslm_decode_params p{};p.layer_budget=static_cast<int32_t>(fx.geo.layers);sslm_seq batch[1]={seq};int32_t token=-1;
		const auto begin=clock_type::now();const sslm_status status=sslm_decode_step(fx.model,batch,1,&p,nullptr,&token);
		if(mutant){const double own=std::chrono::duration<double,std::nano>(clock_type::now()-begin).count();
			const auto until=clock_type::now()+std::chrono::nanoseconds(static_cast<int64_t>(own*0.1765));
			while(clock_type::now()<until){} }
		const double ns=std::chrono::duration<double,std::nano>(clock_type::now()-begin).count();
		Require(status==SSLM_OK&&token>=0,"one-call token decode failed");return {ns,token};}
};
void Pairs(int argc,char** argv,bool shared){
	// pair79 A B CAP_A CAP_B LAYERS HEADS DIM BLOB_A BLOB_B PAIRS SEED OUT
	// pair79b MODEL CAP LAYERS HEADS DIM PAIRS SEED OUT
	Require(argc==(shared?10:15),"pair argument count");
	const Fixture* fa=nullptr;const Fixture* fb=nullptr;int pairs=0;uint32_t seed=0;const char* output=nullptr;
	if(shared){fa=&Model(argv[2],Number(argv[3]),Number(argv[4]),Number(argv[5]),Number(argv[6]));fb=fa;
		pairs=Number(argv[7]);seed=Seed(argv[8]);output=argv[9];}
	else {fa=&Model(argv[2],Number(argv[4]),Number(argv[6]),Number(argv[7]),Number(argv[8]));
		fb=&Model(argv[3],Number(argv[5]),Number(argv[6]),Number(argv[7]),Number(argv[8]));
		pairs=Number(argv[11]);seed=Seed(argv[12]);output=argv[13];}
	Require(pairs>0&&pairs<=508,"pair count must be 1..508");
	Arm a(*fa),b(*fb);
	if(shared){a.Shared();b.Contiguous();}else{const int expected=Number(argv[14]);a.Restore(argv[9],expected);b.Restore(argv[10],expected);}
	Require(a.Ready()==b.Ready(),"ready tokens differ");
	for(int i=0;i<4;++i)Require(a.Decode().second==b.Decode().second,"warmup tokens differ");
	std::ofstream out(output,std::ios::binary|std::ios::trunc);Require(bool(out),"pair output cannot open");
	std::mt19937 random(seed);
	const char* mutant=std::getenv("SUPERSLM_PAGED_KV_79_SLOW_ARM");
	const bool slow_a=mutant&&std::string(mutant)=="A",slow_b=mutant&&std::string(mutant)=="B";
	for(int i=0;i<pairs;++i){std::pair<double,int32_t> x,y;const bool first=(random()&1)!=0;
		if(first){x=a.Decode(slow_a);y=b.Decode(slow_b);}else{y=b.Decode(slow_b);x=a.Decode(slow_a);}
		Require(x.second==y.second,"token-stream mismatch at pair "+std::to_string(i));
		out<<i<<','<<(first?"A":"B")<<','<<x.first<<','<<y.first<<','<<x.second<<','<<y.second<<'\n';
		Require(bool(out),"pair output write failed");}
	out.flush();Require(bool(out),"pair output flush failed");
	std::cout<<"PAIR_OK kind="<<(shared?"79b":"79")<<" pairs="<<pairs<<" model_a="<<fa->sha<<" model_b="<<fb->sha<<"\n";
}
// Cell 7.9 on the R9 build (rev 16.21; T4): one process of tools/run_paged_kv_rows.py's row.
//   row79 CAP LAYERS HEADS DIM SEED --blob BLOB --model FILE --process K --pairs N --warmup W
// One model; the row's one blob restored into a legacy holder in a page pool at the derived B (one block, the pool's P
// pages), rotated, and into a legacy holder in a one-block contiguous-option pool. Stdout, in order: the identity line;
// one "PAIR <paged_ns> <contiguous_ns> <paged_token> <contiguous_token>" line per timed pair (W untimed warm-up tokens
// per arm first; each pair's order drawn at random from SEED + K; each token one sslm_decode_step, timed alone); then
// "IDS ..." with the paged holder's mapped page ids in logical order (the scatter guard's input). The driver grades
// tokens and pair counts; this executable reports them. The paged arm's reserve is scattered before the restore by the
// test author's construction (tests/paged-kv/pkv_scatter_reserve.h, ScatterReserve at kScatterReserveSeed; plan cell
// 7.9, §20.2; T-3189 §11.1): P - 1 one-page budget-1 prefixes, one token prefilled and frozen each, released in a
// fixed shuffled order, so the holder draws a permutation of single pages the guard grades. The seed is fixed, so all
// five processes read equal ids; the cost is P - 1 one-token prefills per process, untimed, reported on stderr. The
// contiguous arm is one page per block and needs no reserve order. 7.9-MR-OPT's pair is this source, so it scatters at
// its own B = 16.
void Row79(int argc,char** argv){
	Require(argc==17,"row79 argument count");
	const char *blob_path=nullptr,*model_path=nullptr;int process=-1,pairs=-1,warmup=-1;
	for(int i=7;i+1<argc;i+=2){const std::string flag=argv[i];
		if(flag=="--blob")blob_path=argv[i+1];else if(flag=="--model")model_path=argv[i+1];
		else if(flag=="--process")process=Number(argv[i+1]);else if(flag=="--pairs")pairs=Number(argv[i+1]);
		else if(flag=="--warmup")warmup=Number(argv[i+1]);else throw std::runtime_error("unknown row79 flag "+flag);}
	Require(blob_path&&model_path&&process>=1&&pairs>=1&&pairs<=508&&warmup>=0,"row79 flags");
	const std::string model_file(model_path);const size_t cut=model_file.find_last_of("/\\");
	const std::string dir=cut==std::string::npos?".":model_file.substr(0,cut),name=cut==std::string::npos?model_file:model_file.substr(cut+1);
#if defined(_WIN32)
	Require(_putenv_s("SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR",dir.c_str())==0,"cannot set the artifact directory");
#else
	Require(setenv("SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR",dir.c_str(),1)==0,"cannot set the artifact directory");
#endif
	const Fixture& fx=Model(name.c_str(),Number(argv[2]),Number(argv[3]),Number(argv[4]),Number(argv[5]));
	const uint32_t seed=Seed(argv[6]);
	const std::vector<uint8_t> blob=Blob(blob_path);Require(IsMagic(blob,"SSB5"),"row blob is not a legacy SSB5");
	LegacyPool page(fx.model,1);Require(page.status==SSLM_OK,"page pool failed");
	uint32_t singles=0;
	{const int64_t B=sslm_kv_page_positions(fx.model);Require(B>0,"no page size");
		const uint32_t pages=static_cast<uint32_t>((static_cast<int64_t>(fx.geo.context_cap)+B-1)/B);
		Require(ScatterReserve(fx.model,&page.pool,kScatterReserveSeed,&singles)==SSLM_OK&&singles+1==pages,"scattered reserve");
		std::cerr<<"row79: scattered reserve of "<<pages<<" pages at B = "<<B<<": "<<singles<<" one-token prefills\n";}
	ContiguousPool contig(fx.model,1);Require(contig.status==SSLM_OK,"contiguous pool failed");
	sslm_seq paged=nullptr,contiguous=nullptr;
	Require(sslm_seq_restore(fx.model,&page.pool,blob.data(),blob.size(),&paged)==SSLM_OK,"paged restore failed");
	Require(sslm_seq_restore(fx.model,&contig.pool,blob.data(),blob.size(),&contiguous)==SSLM_OK,"contiguous restore failed");
	auto one=[&](sslm_seq s)->std::pair<double,int32_t>{sslm_decode_params p{};p.layer_budget=static_cast<int32_t>(fx.geo.layers);
		sslm_seq batch[1]={s};int32_t token=-1;const auto begin=clock_type::now();
		const sslm_status status=sslm_decode_step(fx.model,batch,1,&p,nullptr,&token);
		const double ns=std::chrono::duration<double,std::nano>(clock_type::now()-begin).count();
		Require(status==SSLM_OK&&token>=0,"one-call token decode failed");return {ns,token};};
	// identity first: the driver checks it against the construction before it reads any pair (79-7, D-SLM8466)
	std::cout<<BuildIdentity()<<'\n';
	// the reserve the paged holder's ids are drawn over (T-3202 F-11): the driver records it in every process and row
	std::cout<<"RESERVE seed="<<kScatterReserveSeed<<" singles="<<singles<<" source=tests/paged-kv/pkv_scatter_reserve.h\n";
	for(int i=0;i<warmup;++i){one(paged);one(contiguous);}
	std::mt19937 random(seed+static_cast<uint32_t>(process));
	for(int i=0;i<pairs;++i){std::pair<double,int32_t> x,y;
		if((random()&1)!=0){x=one(paged);y=one(contiguous);}else{y=one(contiguous);x=one(paged);}
		std::cout<<"PAIR "<<x.first<<' '<<y.first<<' '<<x.second<<' '<<y.second<<'\n';}
	uint32_t mapped=0,id=0;std::ostringstream ids;ids<<"IDS";
	Require(sslm_pkv_test_only_seq_table_entry(paged,0,&id,&mapped)==SSLM_OK,"table read failed");
	for(uint32_t i=0;i<mapped;++i){Require(sslm_pkv_test_only_seq_table_entry(paged,i,&id,nullptr)==SSLM_OK,"table read failed");ids<<' '<<id;}
	std::cout<<ids.str()<<'\n'<<std::flush;
	sslm_seq_release(paged);sslm_seq_release(contiguous);
}
}
int main(int argc,char** argv){
	try{Require(argc>=2,"mode required");const std::string mode=argv[1];
		if(mode=="identity") {Require(argc==2,"identity argument count");
			std::cout<<BuildIdentity()<<'\n';
		}
		else if(mode=="prefill")Prefill(argc,argv);else if(mode=="pair79")Pairs(argc,argv,false);else if(mode=="pair79b")Pairs(argc,argv,true);else if(mode=="row79")Row79(argc,argv);else throw std::runtime_error("unknown mode");
		return 0;
	}catch(const std::exception& error){std::cerr<<"INFRASTRUCTURE: "<<error.what()<<'\n';return 40;}
}
