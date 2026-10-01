// T-3022: each real artifact schema, captured from prefill-built prefixes.
#define T3022_NO_MAIN
#include "t3022_precache_reference.cpp"

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: t3022_schema_reference MODEL OUT");
        Input in(argv[1]);
        Pool pool(in.model, in.cap);
        std::ofstream out(argv[2], std::ios::binary|std::ios::trunc);
        if (!out) throw std::runtime_error("open output failed");
        sslm_decode_params probe{};
        const bool damped=sslm_decode_params_init(in.model,SSLM_DECODE_MODE_DAMPED_GREEDY,1,&probe)==SSLM_OK;
        out << "# anchor=5d5ca58 scenario=each_real_schema damped_available=" << int(damped) << '\n';
        const size_t count=sslm_schema_count(in.model);
        for(size_t i=0;i<count;++i) {
            const std::string name=SchemaName(in.model,int(i));
            sslm_schema schema=nullptr;
            Check(sslm_schema_lookup(in.model,name.c_str(),&schema),"schema_lookup");
            Prefix p;
            Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,128,&p.p),"prefix_begin");
            Check(sslm_prefix_set_schema(p.p,schema),"prefix_set_schema");
            Prefill(in.model,p.p,Stream(84,128),SSLM_SPAN_PROMPT);
            Check(sslm_prefix_freeze(p.p),"prefix_freeze");
            for(int a=0;a<4;++a)
                Row(out,"schema_"+name,int(i),a,in.model,pool,p.p,damped,schema,nullptr);
        }
        out.flush(); if(!out) throw std::runtime_error("write failed");
        return 0;
    } catch(const std::exception& e) { std::cerr << "T-3022 schema error: " << e.what() << '\n'; return 1; }
}
