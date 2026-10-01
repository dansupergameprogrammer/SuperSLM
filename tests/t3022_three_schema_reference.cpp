// T-3022: index-2 schema and cross-binding reference over prefill-built prefixes.
#define T3022_NO_MAIN
#include "t3022_precache_reference.cpp"

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: t3022_three_schema_reference MODEL OUT");
        Input in(argv[1]);
        if (sslm_schema_count(in.model) != 3) throw std::runtime_error("expected exactly three schemas");
        Pool pool(in.model, in.cap);
        std::ofstream out(argv[2], std::ios::binary|std::ios::trunc);
        if (!out) throw std::runtime_error("open output failed");
        out << "# anchor=5d5ca58 scenario=three_schema_prefill\n";
        sslm_schema schemas[3]{};
        for (int i=0; i<3; ++i) {
            const std::string name=SchemaName(in.model,i);
            Check(sslm_schema_lookup(in.model,name.c_str(),&schemas[i]),"schema_lookup");
        }
        for (int i=0; i<3; ++i) {
            Prefix p;
            Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,128,&p.p),"prefix_begin");
            Check(sslm_prefix_set_schema(p.p,schemas[i]),"prefix_set_schema");
            Prefill(in.model,p.p,Stream(84+uint32_t(i),128),SSLM_SPAN_PROMPT);
            Check(sslm_prefix_freeze(p.p),"prefix_freeze");
            for (int a=0;a<4;++a)
                Row(out,"three_schema_"+SchemaName(in.model,i),i,a,in.model,pool,p.p,true,schemas[i],nullptr);
            for (int bound=-1;bound<3;++bound) {
                sslm_seq s=nullptr;
                Check(sslm_seq_create_budgeted(in.model,&pool.pool,128,&s),"seq_create_budgeted");
                if (bound>=0) Check(sslm_seq_set_schema(s,schemas[bound]),"seq_set_schema");
                const sslm_status st=sslm_seq_adopt_prefix(s,p.p);
                out << "disposition " << i << ' ' << bound << " status=" << int(st) << '\n';
                Check(sslm_seq_release(s),"seq_release");
            }
        }
        out.flush(); if(!out) throw std::runtime_error("write failed");
        return 0;
    } catch(const std::exception& e) { std::cerr << "T-3022 three-schema error: " << e.what() << '\n'; return 1; }
}
