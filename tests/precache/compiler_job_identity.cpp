#include "precache_provenance.h"
#include <chrono>
#include <cstdio>

int main() {
    namespace fs=std::filesystem;
    using pc::Receipt;
    using pc::ShaFile;
    const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir=fs::temp_directory_path()/("pc-compiler-control-"+std::to_string(nonce));
    fs::create_directories(dir);
    const auto put=[&](const char* suffix,const char* content) {
        std::ofstream out(dir/(std::string("MSVC.")+suffix),std::ios::binary);
        out<<content;
    };
    put("runner","actual MSVC runner binary");
    put("compiler","actual MSVC compiler binary");
    put("cmake","set(CMAKE_CXX_COMPILER \"C:/compiler/cl.exe\")\n"
                "set(CMAKE_CXX_COMPILER_ID \"MSVC\")\n"
                "set(CMAKE_CXX_COMPILER_VERSION \"19.33\")\n");
    put("build.log","configure and build passed\n");
    put("capture.log","[E1] 4.3/E4 green (1 checks, 0 failed)\n"
                      "1 cells, 0 red; 1 checks, 0 failures\n");
    put("spx1","actual generated blob");
    Receipt local{{"format","PC-COMPILER-1"},{"compiler","MSVC"},
                  {"compiler_version","19.33"},{"project_version","1"},
                  {"fixture_revision",pc::kFixtureRevision},{"model_sha256","model"},
                  {"input_ids_sha256","ids"},
                  {"blob_sha256",ShaFile(dir/"MSVC.spx1")},
                  {"runner_sha256",ShaFile(dir/"MSVC.runner")}};
    Receipt job{{"format","PC-COMPILER-JOB-1"},{"compiler","MSVC"},
                {"compiler_version","19.33"},
                {"runner_sha256",ShaFile(dir/"MSVC.runner")},
                {"compiler_tool_sha256",ShaFile(dir/"MSVC.compiler")},
                {"cmake_record_sha256",ShaFile(dir/"MSVC.cmake")},
                {"build_log_sha256",ShaFile(dir/"MSVC.build.log")},
                {"capture_log_sha256",ShaFile(dir/"MSVC.capture.log")},
                {"blob_sha256",ShaFile(dir/"MSVC.spx1")},
                {"source_revision",std::string(40,'a')}};
    bool good=pc::WriteReceipt(dir/"MSVC.job",job) &&
              pc::ValidCompilerJob(dir,local,local,"MSVC");
    // The old guard accepted this: a copied blob plus edited compiler and
    // runner strings. It has no matching GNU binary or CMake compiler job.
    Receipt forged=local;
    forged["compiler"]="GNU";
    forged["runner_sha256"]=std::string(64,'b');
    Receipt forged_job=job;
    forged_job["compiler"]="GNU";
    forged_job["runner_sha256"]=forged["runner_sha256"];
    pc::WriteReceipt(dir/"GNU.job",forged_job);
    for(const char* suffix:{"runner","compiler","cmake","build.log",
                            "capture.log","spx1"})
        fs::copy_file(dir/(std::string("MSVC.")+suffix),
                      dir/(std::string("GNU.")+suffix));
    const bool copied_rejected=!pc::ValidCompilerJob(dir,forged,local,"GNU");
    put("runner","tampered runner binary");
    const bool tamper_rejected=!pc::ValidCompilerJob(dir,local,local,"MSVC");
    fs::remove_all(dir);
    if(!good || !copied_rejected || !tamper_rejected) {
        std::fprintf(stderr,"compiler job control: positive=%d copied=%d tampered=%d\n",
                     int(good),int(copied_rejected),int(tamper_rejected));
        return 1;
    }
    std::puts("compiler job: artifact-bound positive; copied and tampered producers rejected");
    return 0;
}
