// C6 7.4: paired calls at two caps on one model; fixed timer vitality precedes grading.
#include "pkv_budget_c_helpers.h"
#include "superslm/model.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define PKV_74_X86 1
#include <immintrin.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#endif
#else
#define PKV_74_X86 0
#endif

#if !defined(_WIN32)
extern char** environ;
#endif

namespace {
using namespace pkv;
using namespace pkv::bc;
constexpr int kPairs = 205;
enum class Verdict { Pass, Fail, NoResult };
[[noreturn]] void Refuse(int code,const char* why);
// The box timing lock (D-SLM8464), the fourth holder beside tools/run_paged_kv_rows.py's timing_lock,
// tools/run_paged_kv_pairs.py's PidLock and tools/paged_kv_c6_provenance.ps1's Enter-PkvTimingLock, under the same one
// format and protocol: the file holds the owner's bare decimal pid. A fresh acquire writes the pid to a temporary file
// and moves it into place without replacing, so the lock is never visible empty; an existing file is recovered only
// under an exclusive non-blocking OS lock on one byte at kLock74Byte (far past any content); then a file modified
// before the system's last boot is stale whatever it holds, since its pid may now name an unrelated process
// (D-SLM8466 (3)); otherwise content other than ASCII digits naming a pid from 1 to 2**31 - 1, with only space, tab,
// CR and LF stripped around it, is torn and refused, and so is a file longer than kLock74MaxBytes; a live pid refuses;
// an empty file or a dead pid is stale. A stale file is overwritten in place with this pid, never unlinked, so two
// contenders cannot both reclaim it. A release unlinks the file only when it still holds this pid.
std::string locked74_path;
constexpr int kLock74Attempts=3;
constexpr long long kLock74Byte=1LL<<20;
constexpr size_t kLock74MaxBytes=4096;
enum class Owner74 { Empty, Torn, Pid };
// The pid `content` names (a content longer than kLock74MaxBytes is torn whatever it holds).
Owner74 ReadOwner74(const std::string& content,unsigned long& owner){
	if(content.size()>kLock74MaxBytes)return Owner74::Torn;
	const char* trim=" \t\r\n";
	const size_t first=content.find_first_not_of(trim);
	if(first==std::string::npos)return Owner74::Empty;
	const std::string text=content.substr(first,content.find_last_not_of(trim)-first+1);
	unsigned long long value=0;  // saturates past the largest pid, so leading zeros and long numbers read as Python's int
	for(const char c:text){if(c<'0'||c>'9')return Owner74::Torn;value=std::min(value*10+static_cast<unsigned>(c-'0'),1ULL<<32);}
	if(value<1||value>2147483647ULL)return Owner74::Torn;
	owner=static_cast<unsigned long>(value);return Owner74::Pid;
}
bool OwnsLock74(const std::string& content,unsigned long pid){
	unsigned long owner=0;return ReadOwner74(content,owner)==Owner74::Pid&&owner==pid;
}
#if defined(_WIN32)
std::string ReadCapped74(HANDLE handle){
	std::string data;char buffer[1024];
	if(SetFilePointer(handle,0,nullptr,FILE_BEGIN)==INVALID_SET_FILE_POINTER)return data;
	while(data.size()<=kLock74MaxBytes){
		const DWORD want=static_cast<DWORD>(std::min<size_t>(sizeof(buffer),kLock74MaxBytes+1-data.size()));
		DWORD count=0;
		if(!ReadFile(handle,buffer,want,&count,nullptr)||!count)break;
		data.append(buffer,count);
	}
	return data;
}
void Release74(){
	const HANDLE handle=CreateFileA(locked74_path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
	if(handle==INVALID_HANDLE_VALUE)return;  // a reader holds it open: it keeps this pid, stale once this process exits
	const std::string data=ReadCapped74(handle);CloseHandle(handle);
	if(OwnsLock74(data,GetCurrentProcessId()))DeleteFileA(locked74_path.c_str());
}
// The file was last written before the last boot: now less GetTickCount64 (which counts sleep and hibernation), the
// reading the other three holders take.
bool BeforeBoot74(HANDLE handle){
	FILETIME written{},now{};
	if(!GetFileTime(handle,nullptr,nullptr,&written))return false;
	GetSystemTimeAsFileTime(&now);
	const unsigned long long w=(static_cast<unsigned long long>(written.dwHighDateTime)<<32)|written.dwLowDateTime;
	const unsigned long long n=(static_cast<unsigned long long>(now.dwHighDateTime)<<32)|now.dwLowDateTime;
	return w<n-GetTickCount64()*10000ULL;
}
// True when this call published the lock; false when a lock file exists (or is being deleted).
bool Publish74(const std::string& path,const std::string& pid){
	const std::string temp=path+"."+pid+".new";
	const HANDLE handle=CreateFileA(temp.c_str(),GENERIC_WRITE,0,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
	if(handle==INVALID_HANDLE_VALUE)Refuse(77,"C6 timing lock temporary file create failed");
	DWORD written=0;
	const bool okay=WriteFile(handle,pid.data(),static_cast<DWORD>(pid.size()),&written,nullptr)&&written==pid.size();
	CloseHandle(handle);
	if(!okay){DeleteFileA(temp.c_str());Refuse(77,"C6 timing lock owner write failed");}
	const bool moved=MoveFileExA(temp.c_str(),path.c_str(),0)!=0;  // no MOVEFILE_REPLACE_EXISTING: fails when the lock exists
	const DWORD error=moved?ERROR_SUCCESS:GetLastError();
	DeleteFileA(temp.c_str());
	if(moved)return true;
	if(error!=ERROR_ALREADY_EXISTS&&error!=ERROR_FILE_EXISTS&&error!=ERROR_ACCESS_DENIED)Refuse(77,"C6 timing lock create failed");
	return false;
}
// True when this call recovered a stale lock for this process; false when the file went away, was held open
// without write sharing, or is pending delete (retry the fresh acquire).
bool Recover74(const std::string& path,const std::string& pid){
	// No delete sharing: while this handle is open nothing can delete the file, so the path still names it.
	const HANDLE handle=CreateFileA(path.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
	if(handle==INVALID_HANDLE_VALUE){
		const DWORD error=GetLastError();
		// A sharing violation (an open without write sharing) or access denied (the file is pending delete: a release
		// deleted it while another handle was still open, and Publish74 also fails with access denied until the last
		// handle closes) clears only with time, so wait before the next attempt; without it all kLock74Attempts finish
		// in microseconds. A file that went away is retried at once.
		if(error==ERROR_SHARING_VIOLATION||error==ERROR_ACCESS_DENIED){Sleep(50);return false;}
		if(error==ERROR_FILE_NOT_FOUND)return false;
		Refuse(77,"C6 timing lock open failed");
	}
	OVERLAPPED region{};region.Offset=static_cast<DWORD>(kLock74Byte&0xffffffff);region.OffsetHigh=static_cast<DWORD>(kLock74Byte>>32);
	if(!LockFileEx(handle,LOCKFILE_EXCLUSIVE_LOCK|LOCKFILE_FAIL_IMMEDIATELY,0,1,0,&region)){
		CloseHandle(handle);Refuse(77,"C6 timing lock is being recovered by another runner");}
	const auto fail=[&](const char* why){UnlockFileEx(handle,0,1,0,&region);CloseHandle(handle);Refuse(77,why);};
	const bool stale=BeforeBoot74(handle);
	if(!stale){
		unsigned long owner=0;
		const Owner74 kind=ReadOwner74(ReadCapped74(handle),owner);
		if(kind==Owner74::Torn)fail("C6 timing lock owner record torn");
		if(kind==Owner74::Pid){
			const HANDLE process=OpenProcess(SYNCHRONIZE,FALSE,static_cast<DWORD>(owner));
			if(process){const DWORD state=WaitForSingleObject(process,0);CloseHandle(process);
				if(state!=WAIT_OBJECT_0)fail("C6 timing lock already held");}
			else if(GetLastError()==ERROR_ACCESS_DENIED)fail("C6 timing lock already held");
			else if(GetLastError()!=ERROR_INVALID_PARAMETER)fail("C6 timing lock owner cannot be checked");
		}
	}
	DWORD written=0;
	const bool okay=SetFilePointer(handle,0,nullptr,FILE_BEGIN)!=INVALID_SET_FILE_POINTER&&SetEndOfFile(handle)&&
		WriteFile(handle,pid.data(),static_cast<DWORD>(pid.size()),&written,nullptr)&&written==pid.size();
	if(!okay)fail("C6 timing lock owner write failed");
	UnlockFileEx(handle,0,1,0,&region);CloseHandle(handle);
	return true;
}
#else
std::string ReadCapped74(int fd){
	std::string data;char buffer[1024];
	while(data.size()<=kLock74MaxBytes){
		const ssize_t count=pread(fd,buffer,std::min<size_t>(sizeof(buffer),kLock74MaxBytes+1-data.size()),static_cast<off_t>(data.size()));
		if(count<=0)break;
		data.append(buffer,static_cast<size_t>(count));
	}
	return data;
}
void Release74(){
	const int fd=open(locked74_path.c_str(),O_RDONLY);
	if(fd<0)return;
	const std::string data=ReadCapped74(fd);close(fd);
	if(OwnsLock74(data,static_cast<unsigned long>(getpid())))unlink(locked74_path.c_str());
}
// The file was last written before the last boot: now less CLOCK_BOOTTIME, the reading the Python holders take.
bool BeforeBoot74(int fd){
#if defined(CLOCK_BOOTTIME)
	struct stat info{};timespec now{},uptime{};
	if(fstat(fd,&info)!=0||clock_gettime(CLOCK_REALTIME,&now)!=0||clock_gettime(CLOCK_BOOTTIME,&uptime)!=0)return false;
	const double boot=(now.tv_sec-uptime.tv_sec)+(now.tv_nsec-uptime.tv_nsec)*1e-9;
	return info.st_mtim.tv_sec+info.st_mtim.tv_nsec*1e-9<boot;
#else
	(void)fd;return false;
#endif
}
bool Publish74(const std::string& path,const std::string& pid){
	const std::string temp=path+"."+pid+".new";
	const int fd=open(temp.c_str(),O_CREAT|O_TRUNC|O_WRONLY,0600);
	if(fd<0)Refuse(77,"C6 timing lock temporary file create failed");
	const bool okay=write(fd,pid.data(),pid.size())==static_cast<ssize_t>(pid.size());
	close(fd);
	if(!okay){unlink(temp.c_str());Refuse(77,"C6 timing lock owner write failed");}
	const bool linked=link(temp.c_str(),path.c_str())==0;  // fails when the lock exists
	const int error=errno;
	unlink(temp.c_str());
	if(linked)return true;
	if(error!=EEXIST)Refuse(77,"C6 timing lock create failed");
	return false;
}
bool Recover74(const std::string& path,const std::string& pid){
	const int fd=open(path.c_str(),O_RDWR);
	if(fd<0){if(errno==ENOENT)return false;Refuse(77,"C6 timing lock open failed");}
	struct flock region{};region.l_type=F_WRLCK;region.l_whence=SEEK_SET;region.l_start=static_cast<off_t>(kLock74Byte);region.l_len=1;
	if(fcntl(fd,F_SETLK,&region)!=0){close(fd);Refuse(77,"C6 timing lock is being recovered by another runner");}
	// close() drops this process's record locks on the file
	const auto fail=[&](const char* why){close(fd);Refuse(77,why);};
	struct stat held{},named{};
	if(fstat(fd,&held)!=0)fail("C6 timing lock stat failed");
	if(stat(path.c_str(),&named)!=0||held.st_dev!=named.st_dev||held.st_ino!=named.st_ino){close(fd);return false;}  // released and replaced after this open
	const bool stale=BeforeBoot74(fd);
	if(!stale){
		unsigned long owner=0;
		const Owner74 kind=ReadOwner74(ReadCapped74(fd),owner);
		if(kind==Owner74::Torn)fail("C6 timing lock owner record torn");
		if(kind==Owner74::Pid){
			if(kill(static_cast<pid_t>(owner),0)==0||errno==EPERM)fail("C6 timing lock already held");
			if(errno!=ESRCH)fail("C6 timing lock owner cannot be checked");
		}
	}
	const bool okay=ftruncate(fd,0)==0&&pwrite(fd,pid.data(),pid.size(),0)==static_cast<ssize_t>(pid.size());
	if(!okay)fail("C6 timing lock owner write failed");
	close(fd);
	return true;
}
#endif
void Lock74(const char* path=nullptr) {
#if defined(_WIN32)
	if(!path){CreateDirectoryA("D:\\_scratch\\SuperSLM",nullptr);path="D:\\_scratch\\SuperSLM\\pkv-c6-timing.lock";}
	const std::string pid=std::to_string(GetCurrentProcessId());
#else
	if(!path)path="/tmp/superslm-pkv-c6-timing.lock";
	const std::string pid=std::to_string(getpid());
#endif
	for(int attempt=0;attempt<kLock74Attempts;++attempt){
		if(Publish74(path,pid)||Recover74(path,pid)){locked74_path=path;std::atexit(Release74);return;}
	}
	Refuse(77,"C6 timing lock could not be acquired");
}
void LockConstruction(){const char* path=std::getenv("PKV_DEBUNK_74_TEST_LOCK_PATH");
	if(!path||!*path)Refuse(77,"C6 lock construction path missing");Lock74(path);PKV_CHECK(true);}
void Print74Hooks(){
	std::vector<std::string> hooks;
	if(std::getenv("SUPERSLM_PAGED_KV_74_ADOPT_MUTANT"))hooks.emplace_back("adopt=1");
	if(std::getenv("SUPERSLM_PAGED_KV_74_RESET_MUTANT"))hooks.emplace_back("reset=1");
	const size_t aliases=hooks.size();
	constexpr const char* prefix="SUPERSLM_PAGED_KV_74_";
	constexpr size_t prefix_size=21;
#if defined(_WIN32)
	LPCH block=GetEnvironmentStringsA();
	if(!block)hooks.emplace_back("environment_read_failed");
	else {for(const char* entry=block;*entry;entry+=std::strlen(entry)+1)
		if(std::strncmp(entry,prefix,prefix_size)==0)hooks.emplace_back(entry);
		FreeEnvironmentStringsA(block);}
#else
	for(char** entry=environ;*entry;++entry)
		if(std::strncmp(*entry,prefix,prefix_size)==0)hooks.emplace_back(*entry);
#endif
	if(hooks.empty())return;
	std::sort(hooks.begin()+aliases,hooks.end());
	std::printf("7.4 hook_state=");
	for(size_t i=0;i<hooks.size();++i)std::printf("%s%s",i?";":"",hooks[i].c_str());
	std::printf(" publication=quarantined\n");
}
void HookConstruction(){Print74Hooks();PKV_CHECK(true);}

double Median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size()/2]; }
double Corrected(double raw,double empty) {
#ifdef PKV_74_NO_SUBTRACTION_MUTANT
	return raw;
#else
	return raw-empty;
#endif
}
bool SubtractionWitness(double low,double large,double empty) {
	return Corrected(low,empty)==low-empty && Corrected(large,empty)==large-empty;
}
bool Vitality(double low, double large, double empty) {
	const double short_net=Corrected(low,empty), long_net=Corrected(large,empty);
	return short_net>0 && long_net>0 && long_net/short_net>=8.0 && long_net/short_net<=12.0 && SubtractionWitness(low,large,empty);
}
Verdict Grade(std::vector<double> v, double* lo=nullptr, double* hi=nullptr) {
	if (v.size()!=kPairs || std::any_of(v.begin(), v.end(), [](double x){return !std::isfinite(x)||x<=0;})) return Verdict::NoResult;
	std::sort(v.begin(), v.end()); if(lo)*lo=v[85]; if(hi)*hi=v[119];
	if(v[119]<=2.0)return Verdict::Pass; if(v[85]>2.0)return Verdict::Fail; return Verdict::NoResult;
}
void Constructions() {
	std::vector<double> v(kPairs,1.0); PKV_CHECK(Grade(v)==Verdict::Pass);
	std::fill(v.begin(),v.end(),8.0); PKV_CHECK(Grade(v)==Verdict::Fail);
	std::fill(v.begin(),v.end(),4.0); PKV_CHECK(Grade(v)==Verdict::Fail);
	std::fill(v.begin(),v.end(),2.0); PKV_CHECK(Grade(v)==Verdict::Pass);
	std::fill(v.begin(),v.end(),std::nextafter(2.0,std::numeric_limits<double>::infinity())); PKV_CHECK(Grade(v)==Verdict::Fail);
	std::fill(v.begin(),v.end(),1.9); PKV_CHECK(Grade(v)==Verdict::Pass);
	std::fill(v.begin(),v.end(),2.1); PKV_CHECK(Grade(v)==Verdict::Fail);
	v.assign(kPairs,2.1);std::fill(v.begin(),v.begin()+120,2.0); PKV_CHECK(Grade(v)==Verdict::Pass);
	v.assign(kPairs,2.1);std::fill(v.begin(),v.begin()+119,2.0); PKV_CHECK(Grade(v)==Verdict::NoResult);
	v.assign(kPairs,2.1);std::fill(v.begin(),v.begin()+85,2.0); PKV_CHECK(Grade(v)==Verdict::Fail);
	v.assign(kPairs,2.1);std::fill(v.begin(),v.begin()+86,2.0); PKV_CHECK(Grade(v)==Verdict::NoResult);
	// Distinct ratios from a reproducible uniform population: sorting and both
	// decision ranks matter even though acquisition order is randomized.
	for (const auto& [start,expected,lower,upper] : {
		std::tuple<int,Verdict,double,double>{81,Verdict::Pass,1.66,2.0},
		std::tuple<int,Verdict,double,double>{115,Verdict::NoResult,2.0,2.34},
		std::tuple<int,Verdict,double,double>{116,Verdict::Fail,2.01,2.35}}) {
		v.resize(kPairs);
		for(int i=0;i<kPairs;++i) v[i]=double(start+i)/100.0;
		std::shuffle(v.begin(),v.end(),std::mt19937(7409+start));
		double lo=0,hi=0;
		PKV_CHECK(Grade(v,&lo,&hi)==expected && std::abs(lo-lower)<1e-12 && std::abs(hi-upper)<1e-12);
	}
	v.assign(kPairs-1,1.0); PKV_CHECK(Grade(v)==Verdict::NoResult);
	PKV_CHECK(Vitality(120,1020,20));
	PKV_CHECK(Vitality(120,1020,20) && !Vitality(120,2000,20) && !Vitality(10,-80,20));
	PKV_CHECK(!Vitality(120,810,20) && Vitality(120,820,20));
	PKV_CHECK(Vitality(120,1220,20) && !Vitality(120,1230,20));
	for(double cost:{20.0,40.0,80.0,200.0}) {
		PKV_CHECK(SubtractionWitness(100+cost,1000+cost,cost));
		PKV_CHECK(Vitality(100+cost,1000+cost,cost));
	}
}
[[noreturn]] void Refuse(int code,const char* why) {
	std::fprintf(stderr,"7.4 INFRASTRUCTURE %d: %s\n",code,why);
	if(State().failures){
		std::printf("[C6] 7.4/C6 RED (%d checks, %d failed)\n",State().checks,State().failures);
		std::printf("1 cells, 1 red; %d checks, %d failures\n",State().checks,State().failures);
		std::fflush(stdout);
		std::exit(1);
	}
	std::exit(code);
}

#if PKV_74_X86
struct Pin {
	bool ok=false;
#if defined(_WIN32)
	DWORD_PTR old=0;
	Pin(){DWORD_PTR p=0,s=0;if(GetProcessAffinityMask(GetCurrentProcess(),&p,&s)&&p){old=SetThreadAffinityMask(GetCurrentThread(),p&(~p+1));ok=old!=0;}}
	~Pin(){if(old)SetThreadAffinityMask(GetCurrentThread(),old);}
#else
	cpu_set_t old{};
	Pin(){if(pthread_getaffinity_np(pthread_self(),sizeof old,&old))return;for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&old)){cpu_set_t one;CPU_ZERO(&one);CPU_SET(c,&one);ok=pthread_setaffinity_np(pthread_self(),sizeof one,&one)==0;return;}}
	~Pin(){if(ok)pthread_setaffinity_np(pthread_self(),sizeof old,&old);}
#endif
};
bool TscInvariant(){
#ifdef PKV_74_FORCE_NO_TSC
	return false;
#endif
#if defined(_WIN32)
	int r[4]{};__cpuid(r,0x80000000);if(static_cast<unsigned>(r[0])<0x80000007u)return false;__cpuid(r,0x80000007);return (r[3]&(1<<8))!=0;
#else
	unsigned a,b,c,d;if(__get_cpuid_max(0x80000000,nullptr)<0x80000007u)return false;__get_cpuid(0x80000007,&a,&b,&c,&d);return (d&(1u<<8))!=0;
#endif
}
bool use_rdtscp=false;
bool RdtscpAvailable(){
#if defined(_WIN32)
	int r[4]{};__cpuid(r,0x80000000);if(static_cast<unsigned>(r[0])<0x80000001u)return false;__cpuid(r,0x80000001);return (r[3]&(1<<27))!=0;
#else
	unsigned a,b,c,d;if(__get_cpuid_max(0x80000000,nullptr)<0x80000001u)return false;__get_cpuid(0x80000001,&a,&b,&c,&d);return (d&(1u<<27))!=0;
#endif
}
uint64_t Tick(){_mm_lfence();unsigned aux=0;auto t=use_rdtscp?__rdtscp(&aux):__rdtsc();_mm_lfence();return t;}
uint64_t Adds(int n){volatile uint64_t x=0;for(int i=0;i<n;++i)x=x+static_cast<uint64_t>(i+1);return x;}
struct Clock {
	double empty=0, resolution=0;bool okay=false;std::vector<double> raw_empty;
	Clock(){std::vector<double> e,a,b;for(int i=0;i<21;++i){auto x=Tick(),y=Tick();e.push_back(double(y-x));
		x=Tick();(void)Adds(i%2?1000:100);y=Tick();(i%2?b:a).push_back(double(y-x));
		x=Tick();(void)Adds(i%2?100:1000);y=Tick();(i%2?a:b).push_back(double(y-x));}
		raw_empty=e;empty=Median(e);resolution=*std::min_element(e.begin(),e.end());okay=Vitality(Median(a),Median(b),empty);
		std::printf("7.4t: resolution %.0f ticks empty %.0f 100 %.0f 1000 %.0f ratio %.3f %s\n",resolution,empty,Median(a),Median(b),(Median(a)>empty)?(Median(b)-empty)/(Median(a)-empty):0,okay?"PASS":"INFRASTRUCTURE");}
};
void TimerLive(){Pin pin;PKV_CHECK(pin.ok);if(!pin.ok)return;Clock clock;PKV_CHECK(clock.okay);}
void Delay(uint64_t ticks){auto start=Tick();while(Tick()-start<ticks)_mm_pause();}
struct CallTrace {int pair;const char* holder;const char* verb;double raw;double corrected;};
struct Side {
	const Fixture& fx;std::unique_ptr<Rig> rig;sslm_prefix prefix=nullptr;sslm_seq seq=nullptr;bool ok=false;
	Side(const Fixture& f,bool budget):fx(f){if(!f.ok)return;auto own=budget?f.R(512):f.CapPages();rig=std::make_unique<Rig>(f,static_cast<uint32_t>(f.R(1000)+own+4));
		prefix=rig->Prefix(BudgetPrefix(f,rig->Pool(),PrefixTokens(1000)));sslm_seq s=nullptr;
		auto st=budget?sslm_seq_create_budgeted(f.model,rig->Pool(),512,&s):sslm_seq_create(f.model,rig->Pool(),&s);seq=rig->Seq(s);ok=st==SSLM_OK&&prefix&&seq;}
	bool Call(double empty,bool mutant_adopt,bool mutant_reset,double* adopt,double* reset,
	          uint64_t bias=0,std::vector<CallTrace>* trace=nullptr,int pair=-1,const char* holder=""){if(!ok)return false;
		auto a=Tick();auto sa=sslm_seq_adopt_prefix(seq,prefix);auto ad_done=Tick();
		if(mutant_adopt)Delay(3*(ad_done-a));if(bias)Delay(bias);auto b=Tick();
		auto sr=sslm_seq_reset(seq);auto re_done=Tick();
		if(mutant_reset)Delay(7*(re_done-b));auto c=Tick();
		ok=sa==SSLM_OK&&sr==SSLM_OK;*adopt=Corrected(double(b-a),empty);*reset=Corrected(double(c-b),empty);
		if(trace){trace->push_back({pair,holder,"adopt",double(b-a),*adopt});trace->push_back({pair,holder,"reset",double(c-b),*reset});}
		return ok;}
};
bool Same74Geometry(const superslm::SslmModelConfig& small_config,
                    const superslm::SslmModelConfig& large_config,
                    const Geometry& small_claim,const Geometry& large_claim) {
	return small_config.num_hidden_layers==large_config.num_hidden_layers&&
	       small_config.num_key_value_heads==large_config.num_key_value_heads&&
	       small_config.head_dim==large_config.head_dim&&
	       small_config.hidden_size==large_config.hidden_size&&
	       small_config.kv_block_size==large_config.kv_block_size&&
	       small_config.num_hidden_layers==small_claim.layers&&large_config.num_hidden_layers==large_claim.layers&&
	       small_config.num_key_value_heads==small_claim.kv_heads&&large_config.num_key_value_heads==large_claim.kv_heads&&
	       small_config.head_dim==small_claim.head_dim&&large_config.head_dim==large_claim.head_dim;
}
void Run74(const Fixture& low,const Fixture& large){
	if(!low.ok||!large.ok)Refuse(71,"artifact missing or unable to load");
	superslm::SslmModelView low_view,large_view;std::string error;
	if(superslm::SslmModel::Load(low.bytes.p,low.size,low_view,&error)!=superslm::SslmModelStatus::Ok||
	   superslm::SslmModel::Load(large.bytes.p,large.size,large_view,&error)!=superslm::SslmModelStatus::Ok)
		Refuse(71,"artifact config cannot be parsed");
	const auto& small_config=low_view.config;const auto& large_config=large_view.config;
	if(!Same74Geometry(small_config,large_config,low.geo,large.geo))
		Refuse(71,"geometry mismatch");
	if(sslm_kv_block_size(low.model)!=size_t{small_config.context_cap}*low.BytesPerToken()||
	   sslm_kv_block_size(large.model)!=size_t{large_config.context_cap}*large.BytesPerToken())Refuse(71,"block size mismatch");
	if(low.geo.context_cap!=4096||large.geo.context_cap!=32768)Refuse(72,"wrong cap");
	if(!TscInvariant())Refuse(73,"invariant TSC absent");Pin pin;if(!pin.ok||std::getenv("SUPERSLM_PAGED_KV_74_FORCE_PIN_FAILURE"))Refuse(73,"cannot pin timing thread");Clock clock;
	if(!clock.okay&&RdtscpAvailable()){use_rdtscp=true;clock=Clock();}
	if(std::getenv("SUPERSLM_PAGED_KV_74_FORCE_VITALITY_FAILURE"))clock.okay=false;
	if(!clock.okay)std::printf("7.4t: INFRASTRUCTURE 76 diagnostic timer vitality failed\n");
	const bool extra_ad=std::getenv("SUPERSLM_PAGED_KV_74_ADOPT_MUTANT")!=nullptr;
	const bool extra_re=std::getenv("SUPERSLM_PAGED_KV_74_RESET_MUTANT")!=nullptr;
	Print74Hooks();
	const char* order_path=std::getenv("SUPERSLM_PAGED_KV_74_ORDER_TRACE");
	const char* call_path=std::getenv("SUPERSLM_PAGED_KV_74_CALL_TRACE");
	const char* bias_value=std::getenv("SUPERSLM_PAGED_KV_74_ORDER_BIAS_TICKS");
	const uint64_t bias=bias_value?std::strtoull(bias_value,nullptr,10):0;
	std::ofstream order;
	if(order_path){order.open(order_path,std::ios::trunc);if(!order)Refuse(71,"cannot open order trace");order<<"holder,pair,first,bias_ticks\n";}
	std::vector<CallTrace> calls;
	std::printf("7.4 artifact hashes: cap4096=%s cap32768=%s\n",low.sha.c_str(),large.sha.c_str());bool unresolved=false;
	for(bool budget:{true,false}){Side a(low,budget),b(large,budget);if(!a.ok||!b.ok)Refuse(71,"holder creation failed");
		std::vector<double> base_reset;
		for(int i=0;i<10;++i){double x=0,y=0;if(!a.Call(clock.empty,0,0,&x,&y)||!b.Call(clock.empty,0,0,&x,&y))Refuse(71,"warmup failed");base_reset.push_back(y);}
		std::vector<double> ra,rb,aa,ab,adopt_ratios,reset_ratios;std::mt19937 rng(7404+static_cast<unsigned>(budget));
		for(int i=0;i<kPairs;++i){double xad=0,xre=0,yad=0,yre=0;
			const bool small_first=(rng()&1)!=0;const char* holder=budget?"budget":"whole_reserve";
			if(order)order<<holder<<','<<i<<','<<(small_first?"small":"large")<<','<<bias<<'\n';
			auto first=[&](uint64_t call_bias){return a.Call(clock.empty,0,0,&xad,&xre,call_bias,call_path?&calls:nullptr,i,holder);};
			auto second=[&](uint64_t call_bias){return b.Call(clock.empty,extra_ad,extra_re,&yad,&yre,call_bias,call_path?&calls:nullptr,i,holder);};
			if(!(small_first?(first(bias)&&second(0)):(second(bias)&&first(0))))Refuse(71,"timed verb failed");
			aa.push_back(xad);ab.push_back(yad);ra.push_back(xre);rb.push_back(yre);adopt_ratios.push_back(yad/xad);reset_ratios.push_back(yre/xre);}
		if(extra_re)std::printf("7.4 hook reset base_ticks=%.3f hooked_ticks=%.3f\n",Median(base_reset),Median(rb));
		const char* control=std::getenv("SUPERSLM_PAGED_KV_74_FLOOR_CONTROL_TICKS");
		const double floor_aa=control?std::strtod(control,nullptr):Median(aa);
		const double floor_ab=control?std::strtod(control,nullptr):Median(ab);
		const double floor_ra=control?std::strtod(control,nullptr):Median(ra);
		const double floor_rb=control?std::strtod(control,nullptr):Median(rb);
		if(std::getenv("SUPERSLM_PAGED_KV_74_FORCE_FLOOR")||floor_aa<100||floor_ab<100||floor_ra<100||floor_rb<100)Refuse(74,"corrected median under 100 ticks");
		for(auto& entry:{std::pair{"adopt",adopt_ratios},std::pair{"reset",reset_ratios}}){double lo=0,hi=0;auto verdict=Grade(entry.second,&lo,&hi);
			std::printf("7.4 %s %s r(86)=%.3f r(120)=%.3f bound=2.0 %s\n",budget?"budget":"whole_reserve",entry.first,lo,hi,verdict==Verdict::Pass?"PASS":verdict==Verdict::Fail?"FAIL":"NO RESULT");
			PKV_CHECK_MSG(verdict!=Verdict::Fail,"7.4 %s %s: ratio above 2.0",budget?"budget":"whole_reserve",entry.first);unresolved|=verdict==Verdict::NoResult;}
	}
	if(call_path){std::ofstream out(call_path,std::ios::trunc);if(!out)Refuse(71,"cannot open call trace");
		out<<"{\"raw_empty_ticks\":[";for(size_t i=0;i<clock.raw_empty.size();++i){if(i)out<<',';out<<clock.raw_empty[i];}
		out<<"],\"calls\":[";for(size_t i=0;i<calls.size();++i){if(i)out<<',';const auto& c=calls[i];
			out<<"{\"pair\":"<<c.pair<<",\"holder\":\""<<c.holder<<"\",\"verb\":\""<<c.verb<<"\",\"raw_ticks\":"<<c.raw<<",\"corrected_ticks\":"<<c.corrected<<'}';}out<<"]}\n";}
	if(unresolved||std::getenv("SUPERSLM_PAGED_KV_74_FORCE_UNRESOLVED"))Refuse(75,"sign interval did not resolve the bound");
}
#else
void Run74(const Fixture&,const Fixture&){Refuse(73,"x86 invariant TSC required");}
void TimerLive(){Refuse(73,"x86 invariant TSC required");}
#endif
// Paged-KV plan rev 16.24 (D-SLM8447, D-SLM8298): 7.4 pins both sides to one B, 64 on the box and 16 on the twin, and
// reads the pinned B back. The pin goes on a private mapping of each artifact, so no shared fixture (the other C6
// cells read the rule's B) is pinned, and the pin precedes every pool on that mapping. The pin's SSLM_OK is required
// once the T0 stub is gone (SUPERSLM_PKV_R9_STUB_PIN OFF); the read-back B is required always.
bool PinnedCopy(const Fixture& src,int64_t B,Fixture* out){
	if(!src.ok)return false;
	out->stem=src.stem;out->geo=src.geo;out->vocab=src.vocab;out->sha=src.sha;out->size=src.size;
	out->bytes.Reset(src.size);std::memcpy(out->bytes.p,src.bytes.p,src.size);
	const sslm_status st=sslm_model_map(out->bytes.p,out->size,&out->model);
	PKV_CHECK_MSG(st==SSLM_OK,"7.4: the private mapping of %s: %d",src.stem.c_str(),static_cast<int>(st));
	if(st!=SSLM_OK)return false;
	const sslm_status pin=KvPagesTestPinPagePositions(out->model,B);
#if !defined(PKV_R9_STUB_PIN)
	PKV_CHECK_MSG(pin==SSLM_OK,"7.4: the pin of %s to B = %lld is refused (%d)",src.stem.c_str(),static_cast<long long>(B),static_cast<int>(pin));
#else
	(void)pin;
#endif
	out->pinned=sslm_kv_page_positions(out->model);
	PKV_CHECK_MSG(out->pinned==B,"7.4: %s runs at B = %lld after its pin, %lld owed",src.stem.c_str(),static_cast<long long>(out->pinned),static_cast<long long>(B));
	out->ok=out->pinned==B;
	return out->ok;
}
void RunPinned74(const Fixture& low,const Fixture& large,int64_t B){
	Fixture a,b;
	if(!PinnedCopy(low,B,&a)||!PinnedCopy(large,B,&b))Refuse(71,"the 7.4 pin did not take");
	Run74(a,b);
}
void Box(){Lock74();const char* large=std::getenv("SUPERSLM_PAGED_KV_74_CAP32768_FILE");RunPinned74(GetArtifact("qwen2.5-0.5b-instruct-cap4096-aex.sslm",Geometry{24,2,64,4096}),
  GetArtifact(large?large:"qwen2.5-0.5b-instruct-cap32768-aex.sslm",Geometry{24,2,64,32768}),64);}
void Fixtures(){Lock74();const char* low=std::getenv("SUPERSLM_PAGED_KV_74_FIXTURE_LOW");const char* large=std::getenv("SUPERSLM_PAGED_KV_74_FIXTURE_LARGE");
	RunPinned74(GetFixture(low?low:"pkv_def"),GetFixture(large?large:"pkv_32k"),16);}
PKV_CELL("7.4/C6","C6",Box);
PKV_CELL("7.4/C6:fixtures","C6",Fixtures);
PKV_CELL("7.4t/C6:constructions","C6",Constructions);
PKV_CELL("7.4t/C6:lock","C6",LockConstruction);
PKV_CELL("7.4t/C6:hooks","C6",HookConstruction);
PKV_CELL("7.4t/C6:live","C6",TimerLive);
}
