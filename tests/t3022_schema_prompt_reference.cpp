// T-3022: the existing pinned long-form prompt_result schema construction.
#define T3022_NO_MAIN
#include "t3022_precache_reference.cpp"

int main(int argc, char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: t3022_schema_prompt_reference MODEL PROMPT_IDS OUT");
        std::ifstream ids(argv[2]); if(!ids) throw std::runtime_error("open prompt ids failed");
        std::vector<int32_t> prompt;
        int32_t id=0;
        while(ids>>id) prompt.push_back(id);
        if(prompt.empty()) throw std::runtime_error("empty prompt ids");
        Input in(argv[1]); Pool pool(in.model,in.cap);
        sslm_schema schema=nullptr;
        Check(sslm_schema_lookup(in.model,"prompt_result",&schema),"schema_lookup");
        Prefix p;
        Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,int32_t(prompt.size()),&p.p),"prefix_begin");
        Check(sslm_prefix_set_schema(p.p,schema),"prefix_set_schema");
        Prefill(in.model,p.p,prompt,SSLM_SPAN_PROMPT);
        Check(sslm_prefix_freeze(p.p),"prefix_freeze");
        std::ofstream out(argv[3],std::ios::binary|std::ios::trunc);
        if(!out) throw std::runtime_error("open output failed");
        out << "# anchor=5d5ca58 scenario=pinned_prompt_result_prefill\n";
        Row(out,"prompt_result",0,0,in.model,pool,p.p,false,schema,nullptr);
        Row(out,"prompt_result",0,1,in.model,pool,p.p,false,schema,nullptr);
        out.flush(); if(!out) throw std::runtime_error("write failed");
        return 0;
    } catch(const std::exception& e) { std::cerr << "T-3022 prompt schema error: " << e.what() << '\n'; return 1; }
}
