#include "precache_common.h"
#include "precache_checkpoint.h"
#include "paged-kv/pkv_legacy_a_helpers.h"

namespace {
using namespace pc;

sslm_prefix Child(const Fixture& fx, sslm_prefix parent, Handles& h,
                  int extension = 200, int seed = 62) {
    sslm_prefix child = nullptr;
    Status(sslm_prefix_begin_from(parent, std::max(extension, 1), &child), SSLM_OK,
           "begin_from fixture");
    if (!child) return nullptr;
    h.prefixes.push_back(child);
    if (extension) Status(PrefixPrefillAll(fx.model, child,
        Stream(seed, extension, fx.vocab), 64), SSLM_OK, "child prefill fixture");
    Status(sslm_prefix_freeze(child), SSLM_OK, "child freeze fixture");
    return child;
}

std::vector<uint8_t> LastSharedVByteChanged(const Fixture& fx,
                                             std::vector<uint8_t> parent_blob) {
    const int64_t origin=1000;
    const size_t pos=size_t((origin/fx.B())*fx.B()-1); // last row in shared page 61.
    const size_t H=fx.geo.kv_heads,D=fx.geo.head_dim;
    const size_t offset=kHeader+ArtifactHiddenSize(fx)+
        ((size_t(fx.geo.layers-1)*2+1)*H+(H-1))*size_t(origin)*D+pos*D+(D-1);
    PKV_CHECK_MSG(offset<parent_blob.size() && pos==991,
                  "2.6 last-byte last-layer V operand is out of range");
    if(offset<parent_blob.size()) {
        parent_blob[offset]^=1;
        Digest(parent_blob);
    }
    return parent_blob;
}

void PreciseHints26(const Fixture& fx,sslm_prefix built,sslm_kv_pool* built_pool,
                    const std::vector<uint8_t>& child_blob) {
    for(const char* origin : {"budget","legacy","restored"}) {
        Scene sc;
        PagePool& pool=sc.AddPool(fx.model,600);
        sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61,
                                std::strcmp(origin,"legacy")==0);
        if(!parent) return;
        if(std::strcmp(origin,"restored")==0) {
            const auto b=Save(parent);
            if(!RequireBlob(b,"2.6 restored-parent origin")) return;
            sslm_prefix copy=nullptr;
            if(!Restored(fx,&pool.pool,b,nullptr,sc.h,&copy)) return;
            parent=copy;
        }
        auto run=[&](const char* label,sslm_prefix hint,sslm_kv_pool* foreign_pool=nullptr,
                     const Fixture* foreign_fx=nullptr,int64_t foreign_free=0,
                     uint32_t expected_shared=0) {
            if(foreign_pool && foreign_fx)
                ProbeExactlyFreeOneState(*foreign_fx,foreign_pool,foreign_free);
            sslm_prefix out=nullptr;
            uint32_t shared=UINT32_MAX;
            const auto st=Restore(fx,&pool.pool,child_blob,hint,&out,&shared);
            PKV_CHECK_MSG(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>(),
                          "2.6 %s/%s private or sharing restore failed: %d",
                          origin,label,int(st));
            if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) {
                PKV_CHECK_MSG(shared==expected_shared,
                              "2.6 %s/%s shared %u, expected %u",
                              origin,label,shared,expected_shared);
                const auto before=Decode(fx,built_pool,built,true,16);
                const auto after=Decode(fx,&pool.pool,out,true,16);
                PKV_CHECK_MSG(before.size()==16 && before==after,
                              "2.6 %s/%s tokens differ from prefill child",origin,label);
                if(hint && expected_shared==62) {
                    const auto a=PrefixTable(hint),c=PrefixTable(out);
                    PKV_CHECK_MSG(a.size()>=62 && c.size()>=62 &&
                                  std::equal(a.begin(),a.begin()+62,c.begin()),
                                  "2.6 %s matching hint did not physically share 62 page indices",
                                  origin);
                }
                sslm_prefix_release(out);
            }
            if(foreign_pool && foreign_fx)
                ProbeExactlyFreeOneState(*foreign_fx,foreign_pool,foreign_free);
        };
        run("matching",parent,nullptr,nullptr,0,62);
        sslm_prefix unfrozen=nullptr;
        Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,1000,&unfrozen),
               SSLM_OK,"2.6 unfrozen hint begin");
        if(unfrozen) {
            sc.h.prefixes.push_back(unfrozen);
            Status(PrefixPrefillAll(fx.model,unfrozen,Stream(61,1000,fx.vocab),64),
                   SSLM_OK,"2.6 unfrozen hint prefill");
            run("(b) unfrozen",unfrozen);
        }
        sslm_prefix wrong_length=Root(fx,&pool.pool,sc.h,1001,61);
        if(wrong_length) run("(e) context length",wrong_length);
        const auto changed=LastSharedVByteChanged(fx,Save(parent));
        if(!RequireBlob(changed,"2.6 last-layer V hint")) return;
        sslm_prefix wrong_rows=nullptr;
        if(Restored(fx,&pool.pool,changed,nullptr,sc.h,&wrong_rows)) {
            const auto saved=Save(wrong_rows);
            PKV_CHECK_MSG(saved==changed,
                          "2.6 last shared page differs in more than the isolated edited byte");
            run("(f) last byte of last shared page last-layer V",wrong_rows);
        }
        PagePool another_pool(fx.model,80);
        Handles foreign_handles;
        sslm_prefix another=Root(fx,&another_pool.pool,foreign_handles,1000,61);
        if(another) run("(d) other pool",another,&another_pool.pool,&fx,17);
        const Fixture& other_model=GetFixture("pkv_qk");
        if(other_model.ok) {
            PagePool model_pool(other_model.model,80);
            Handles model_handles;
            sslm_prefix other=Root(other_model,&model_pool.pool,model_handles,1000,61);
            if(other) run("(c) other model",other,&model_pool.pool,&other_model,17);
        }
    }
}

void Nested63() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    // A restore with a rejected or ignored hint may retain all R(1200)
    // pages. Both restores remain live before the whole-cap adopter.
    PagePool& pool = sc.AddPool(fx.model,uint32_t(
        fx.CapPages()+fx.R(1000)+fx.R(200)+2*fx.R(1200)+fx.R(20)));
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix built = world ? Child(fx,world,sc.h) : nullptr;
    if (!built) return;
    const auto b = Save(built);
    if (!RequireBlob(b,"6.3")) return;
    PKV_CHECK_EQ(static_cast<int64_t>(Le64(b,120)),1000);
    sslm_prefix shared = nullptr, priv = nullptr;
    uint32_t shared_count = UINT32_MAX;
    const bool a = Restored(fx,&pool.pool,b,world,sc.h,&shared,&shared_count);
    const bool c = Restored(fx,&pool.pool,b,nullptr,sc.h,&priv);
    if (a && c) {
        PKV_CHECK_EQ(shared_count,62u);
        const auto parent_pages=PrefixTable(world), shared_pages=PrefixTable(shared);
        PKV_CHECK_MSG(parent_pages.size()>=62 && shared_pages.size()>=62 &&
                      std::equal(parent_pages.begin(),parent_pages.begin()+62,
                                 shared_pages.begin()),
                      "6.3 nested restore did not physically share 62 parent pages");
        TwinTokens(fx,&pool.pool,built,shared,16);
        TwinTokens(fx,&pool.pool,built,priv,16);
        PKV_CHECK_MSG(Save(shared)==Save(priv),"6.3 shared and private SPX1 bytes differ");
    }
}
PKV_CELL("6.3/E3", "E1", Nested63);
PKV_CELL("9.3/E3", "E1", Nested63);
PKV_CELL("7.5/E3", "E1", Nested63);

void Origins42() {
    for(const auto& shape : {std::pair<const char*,Geometry>{"synthetic0p5.sslm",{24,2,64,4096}},
                             {"synthetic1p5.sslm",{28,2,128,32768}}}) {
    auto base=LoadSyntheticShape(shape.first,shape.second);
    if(!base->ok) return;
    Fixture four;
    PKV_CHECK_MSG(pkv::legacy_a::PatchedBlockSizeTwin(*base,4,&four),
                  "4.2 failed to construct B=4 synthetic twin");
    if(!four.ok || !PinFixtureB(&four,4)) return;
    for(const auto& geometry : {std::pair<const Fixture*,int>{base.get(),16},{&four,4}}) {
    const Fixture& fx=*geometry.first;
    const int B=geometry.second;
    PKV_CHECK_MSG(sslm_kv_page_size(fx.model)==size_t(B)*fx.BytesPerToken(),
                  "4.2 selected model is not B=%d",B);
    for (int origin : {0,1,B-1,B,B+1,31,32,65,int(fx.geo.context_cap)-1,
                       int(fx.geo.context_cap)}) {
        // Each origin is one checkpointed case with its own scene and pool.
        const CaseEnd end=RunCase(std::string(shape.first)+"/B="+std::to_string(B)+
                                  "/origin="+std::to_string(origin),{&fx},nullptr,[&]() {
        Scene sc;
        // At cap length the parent, a private fallback restore, and the
        // legacy TwinTokens adopter can each retain cap/B pages. The
        // sharing assertion below still rejects a non-sharing restore.
        PagePool& pool = sc.AddPool(fx.model,
            uint32_t(3*fx.CapPages()+128));
        sslm_prefix parent = Root(fx,&pool.pool,sc.h,origin,41);
        const int extension=origin+16>int(fx.geo.context_cap) ? 0 : 16;
        sslm_prefix child = parent ? Child(fx,parent,sc.h,extension) : nullptr;
        if (!child) return CaseEnd::Next;
        const auto b = Save(child);
        if (!RequireBlob(b,"4.2")) return CaseEnd::StopCell;
        PKV_CHECK_EQ(static_cast<int64_t>(Le64(b,120)),origin);
        sslm_prefix restored = nullptr;
        uint32_t shared = UINT32_MAX;
        if (Restored(fx,&pool.pool,b,parent,sc.h,&restored,&shared)) {
            PKV_CHECK_EQ(shared,uint32_t(origin/B));
            const int continuation=std::min(16,int(fx.geo.context_cap)-origin-extension);
            if(continuation>0)
                TwinTokens(fx,&pool.pool,child,restored,continuation);
        }
        return CaseEnd::Next;
        });
        if(end==CaseEnd::StopCell) return;
    }
    }
    }
}
PKV_CELL("4.2/E3", "E1", Origins42);

void HintFailures26() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    // Parent, child, two wrong roots, unfrozen root, and four retained
    // private restores precede a whole-cap legacy adopter in TwinTokens.
    PagePool& pool = sc.AddPool(fx.model,uint32_t(
        fx.CapPages()+3*fx.R(1000)+fx.R(992)+fx.R(200)+
        4*fx.R(1200)+fx.R(20)));
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child = world ? Child(fx,world,sc.h) : nullptr;
    if (!child) return;
    const auto b = Save(child);
    if (!RequireBlob(b,"2.6")) return;
    sslm_prefix wrong_length = Root(fx,&pool.pool,sc.h,992,61);
    sslm_prefix wrong_rows = Root(fx,&pool.pool,sc.h,1000,63);
    sslm_prefix unfrozen = nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,1000,&unfrozen),SSLM_OK,
           "unfrozen hint fixture");
    if (unfrozen) sc.h.prefixes.push_back(unfrozen);
    PagePool foreign_pool(fx.model,80);
    Handles foreign_h;
    sslm_prefix foreign = Root(fx,&foreign_pool.pool,foreign_h,1000,61);
    for (sslm_prefix hint : {unfrozen,wrong_length,wrong_rows,foreign}) {
        if (!hint) continue;
        sslm_prefix out = nullptr;
        uint32_t shared = UINT32_MAX;
        if (Restored(fx,&pool.pool,b,hint,sc.h,&out,&shared)) {
            PKV_CHECK_EQ(shared,0u);
            TwinTokens(fx,&pool.pool,child,out,16);
        }
    }
    PreciseHints26(fx,child,&pool.pool,b);
}
PKV_CELL("2.6/E3", "E1", HintFailures26);

void Admission510() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,77); // 63 parent pages + R(200)=14.
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child = world ? Child(fx,world,sc.h) : nullptr;
    if (!child) return;
    const auto b = Save(child);
    if (!RequireBlob(b,"5.10")) return;
    Status(ReleasePrefix(sc.h,child),SSLM_OK,"release built child");
    for (sslm_prefix hint : {sslm_prefix(nullptr),world}) {
        sslm_prefix out = nullptr;
        const sslm_status st = Restore(fx,&pool.pool,b,hint,&out);
        Status(st,hint ? SSLM_OK : SSLM_KV_POOL_EXHAUSTED,
               hint ? "5.10 matching hint" : "5.10 private admission");
        if (!hint) PKV_CHECK(out == nullptr);
        if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sc.h.prefixes.push_back(out);
    }
    const Fixture& other_model=GetFixture("pkv_qk");
    const auto source_parent=Save(world);
    if(!RequireBlob(source_parent,"5.10 source parent")) return;
    for(const char* condition : {"null","(b) unfrozen","(c) other model",
                                 "(d) other pool","(e) length","(f) last-layer V",
                                 "matching"}) {
        const bool unfrozen=std::strcmp(condition,"(b) unfrozen")==0;
        PagePool target(fx.model,unfrozen ? 78 : 77);
        Handles target_h;
        std::unique_ptr<PagePool> foreign_pool;
        Handles foreign_h;
        const Fixture* foreign_fx=nullptr;
        sslm_prefix hint=nullptr;
        if(unfrozen) {
            Status(sslm_prefix_begin_budgeted(fx.model,&target.pool,1000,&hint),
                   SSLM_OK,"5.10 unfrozen hint begin");
            if(hint) {
                target_h.prefixes.push_back(hint);
                Status(PrefixPrefillAll(fx.model,hint,Stream(61,1000,fx.vocab),64),
                       SSLM_OK,"5.10 unfrozen hint prefill");
            }
        } else if(std::strcmp(condition,"(e) length")==0) {
            hint=Root(fx,&target.pool,target_h,1001,61);
        } else if(std::strcmp(condition,"(f) last-layer V")==0) {
            const auto changed=LastSharedVByteChanged(fx,source_parent);
            if(!Restored(fx,&target.pool,changed,nullptr,target_h,&hint)) return;
        } else {
            sslm_prefix resident=Root(fx,&target.pool,target_h,1000,61);
            if(!resident) return;
            if(std::strcmp(condition,"matching")==0) hint=resident;
        }
        if(std::strcmp(condition,"(c) other model")==0 && other_model.ok) {
            foreign_fx=&other_model;
            foreign_pool=std::make_unique<PagePool>(other_model.model,80);
            hint=Root(other_model,&foreign_pool->pool,foreign_h,1000,61);
        } else if(std::strcmp(condition,"(d) other pool")==0) {
            foreign_fx=&fx;
            foreign_pool=std::make_unique<PagePool>(fx.model,80);
            hint=Root(fx,&foreign_pool->pool,foreign_h,1000,61);
        }
        if(std::strcmp(condition,"(c) other model")==0 && !other_model.ok) continue;
        if(!hint && std::strcmp(condition,"null")!=0) return;
        ProbeExactlyFreeOneState(fx,&target.pool,14);
        if(foreign_pool) ProbeExactlyFreeOneState(*foreign_fx,&foreign_pool->pool,17);
        if(std::strcmp(condition,"matching")==0) {
            sslm_prefix control=Child(fx,hint,target_h,200,62);
            PKV_CHECK_MSG(control!=nullptr,
                          "5.10 begin_from control did not admit at 63+14 pages");
            if(control) Status(ReleasePrefix(target_h,control),SSLM_OK,
                               "5.10 release admitted begin_from control");
            ProbeExactlyFreeOneState(fx,&target.pool,14);
        }
        uint32_t shared=UINT32_MAX;
        sslm_prefix out=nullptr;
        const auto st=Restore(fx,&target.pool,b,hint,&out,&shared);
        const bool matching=std::strcmp(condition,"matching")==0;
        PKV_CHECK_MSG(st==(matching ? SSLM_OK : SSLM_KV_POOL_EXHAUSTED),
                      "5.10 %s: expected %s, got %d",condition,
                      matching ? "admission" : "private capacity refusal",int(st));
        if(matching && st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) {
            const auto parent_table=PrefixTable(hint),child_table=PrefixTable(out);
            const std::set<uint32_t> private_pages(
                child_table.size()>=75 ? child_table.begin()+62 : child_table.end(),
                child_table.end());
            PKV_CHECK_MSG(shared==62 && parent_table.size()==63 &&
                          child_table.size()==75 && private_pages.size()==13 &&
                          std::none_of(private_pages.begin(),private_pages.end(),
                                       [&](uint32_t page){return std::find(parent_table.begin(),
                                                parent_table.end(),page)!=parent_table.end();}) &&
                          std::equal(parent_table.begin(),parent_table.begin()+62,
                                     child_table.begin()),
                          "5.10 matching hint did not share 62 and draw 13 distinct private pages");
            sslm_seq two=nullptr;
            Status(sslm_seq_create_budgeted(fx.model,&target.pool,16,&two),
                   SSLM_KV_POOL_EXHAUSTED,"5.10 matching restore drew 13 of 14 free pages");
            if(two) sslm_seq_release(two);
            sslm_prefix_release(out);
        } else {
            PKV_CHECK_MSG(out==nullptr,"5.10 %s refusal produced a handle",condition);
        }
        ProbeExactlyFreeOneState(fx,&target.pool,14);
        if(foreign_pool) ProbeExactlyFreeOneState(*foreign_fx,&foreign_pool->pool,17);
        foreign_h.ReleaseAll();
        target_h.ReleaseAll();
        ProbeExactlyFreeOneState(fx,&target.pool,unfrozen ? 78 : 77);
        PKV_CHECK_EQ(sslm_kv_pool_destroy(target.pool),SSLM_OK);
        target.pool=nullptr;
    }
}
PKV_CELL("5.10/E3", "E1", Admission510);

void ParentRelease33() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    // The hinted restore can occupy a private R(1200) fallback before the
    // sharing oracle runs; the legacy adopter still reserves cap/B pages.
    PagePool& pool = sc.AddPool(fx.model,uint32_t(
        fx.CapPages()+fx.R(1000)+fx.R(200)+fx.R(1200)+fx.R(20)));
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child = world ? Child(fx,world,sc.h) : nullptr;
    if (!child) return;
    const auto b = Save(child);
    if (!RequireBlob(b,"3.3")) return;
    sslm_prefix copy = nullptr;
    if (!Restored(fx,&pool.pool,b,world,sc.h,&copy)) return;
    sslm_status release_status=SSLM_INVALID_ARGUMENT;
    std::thread release([&]{ release_status=ReleasePrefix(sc.h,world); });
    release.join();
    Status(release_status,SSLM_OK,"release parent on other thread");
    TwinTokens(fx,&pool.pool,child,copy,16);
}
PKV_CELL("3.3/E3", "E1", ParentRelease33);

void Composition82() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,400);
    sslm_prefix root = Root(fx,&pool.pool,sc.h,32);
    if (!root) return;
    const auto b = Save(root);
    if (!RequireBlob(b,"8.2")) return;
    sslm_prefix restored = nullptr;
    if (!Restored(fx,&pool.pool,b,nullptr,sc.h,&restored)) return;
    sslm_prefix child = Child(fx,restored,sc.h,32);
    if (!child) return;
    const auto cb = Save(child);
    sslm_prefix child_copy = nullptr;
    if (!Restored(fx,&pool.pool,cb,restored,sc.h,&child_copy)) return;
    sslm_prefix grandchild = Child(fx,child_copy,sc.h,32);
    const auto gb = grandchild ? Save(grandchild) : std::vector<uint8_t>{};
    sslm_prefix grandcopy = nullptr;
    uint32_t grand_shared=UINT32_MAX;
    if (Restored(fx,&pool.pool,gb,child_copy,sc.h,&grandcopy,&grand_shared)) {
        const auto child_pages=PrefixTable(restored), descendant_pages=PrefixTable(child);
        const auto copy_pages=PrefixTable(child_copy), grand_pages=PrefixTable(grandcopy);
        PKV_CHECK_MSG(child_pages.size()>=2 && descendant_pages.size()>=2 &&
                      std::equal(child_pages.begin(),child_pages.begin()+2,
                                 descendant_pages.begin()),
                      "8.2 child does not share root's full pages");
        PKV_CHECK_MSG(copy_pages.size()>=4 && grand_pages.size()>=4 &&
                      grand_shared==4 &&
                      std::equal(copy_pages.begin(),copy_pages.begin()+4,
                                 grand_pages.begin()),
                      "8.2 grandchild restore does not share child's full pages");
        Status(ReleasePrefix(sc.h,child_copy),SSLM_OK,"8.2 release middle parent");
        Status(ReleasePrefix(sc.h,restored),SSLM_OK,"8.2 release root parent");
        TwinTokens(fx,&pool.pool,grandchild,grandcopy,16);
    }
}
PKV_CELL("8.2/E3", "E1", Composition82);

void NestedCapacity51() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix source_parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix source_child=source_parent ? Child(fx,source_parent,source.h) : nullptr;
    if(!source_child) return;
    const auto b=Save(source_child);
    if(!RequireBlob(b,"5.1 nested")) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,75); // parent 63; 12 free, 13 needed.
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    if(!parent) return;
    ProbeExactlyFreeOneState(fx,&pool.pool,12);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,b,parent,&out),SSLM_KV_POOL_EXHAUSTED,
           "5.1 nested one page short");
    PKV_CHECK(out==nullptr);
    ProbeExactlyFreeOneState(fx,&pool.pool,12);
    Status(ReleasePrefix(sc.h,parent),SSLM_OK,"5.1 release parent after refusal");
    ProbeExactlyFreeOneState(fx,&pool.pool,75);
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
    pool.pool=nullptr;
}
PKV_CELL("5.1/E3", "E1", NestedCapacity51);

void NestedAllocation52() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    // Parent R(1000) plus child R(200) leave exactly 124 free pages for the
    // post-map allocation fault and rollback oracle below.
    PagePool& pool=sc.AddPool(fx.model,200);
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child=parent ? Child(fx,parent,sc.h) : nullptr;
    if(!child) return;
    const auto b=Save(child);
    if(!RequireBlob(b,"5.2 nested")) return;
    ProbeExactlyFreeOneState(fx,&pool.pool,124);
    sslm_pc_test_only_restore_fault_arm(3,0);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,b,parent,&out),SSLM_ALLOCATION_FAILED,
           "5.2 nested post-map allocation failure");
    PKV_CHECK(out==nullptr);
    uint32_t free=0,shared=0,mapped=0,reserve=0;
    PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve)==1,
                  "5.2 nested allocation slot not reached");
    PKV_CHECK_EQ(shared,62u);
    PKV_CHECK_EQ(mapped,75u);
    PKV_CHECK_EQ(reserve,0u);
    PKV_CHECK_MSG(free<124,"5.2 nested fault fired before drawing private pages");
    sslm_pc_test_only_restore_fault_disarm();
    ProbeExactlyFreeOneState(fx,&pool.pool,124);
    Status(ReleasePrefix(sc.h,child),SSLM_OK,"5.2 release built child");
    Status(ReleasePrefix(sc.h,parent),SSLM_OK,"5.2 release parent");
    ProbeExactlyFreeOneState(fx,&pool.pool,200);
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
    pool.pool=nullptr;
}
PKV_CELL("5.2/E3", "E1", NestedAllocation52);

void NestedRefusals511() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix child=parent ? Child(fx,parent,source.h) : nullptr;
    if(!child) return;
    const auto b=Save(child);
    if(!RequireBlob(b,"5.11 nested")) return;
    ProbeExactlyFreeOneState(fx,&source_pool.pool,124);
    for(int status : {1,2}) { // test slot: 1=kInvalidArgument; 2=kRefcountUnderflow.
        sslm_pc_test_only_restore_fault_arm(1,status);
        sslm_prefix out=nullptr;
        Status(Restore(fx,&source_pool.pool,b,parent,&out),
               status==1 ? SSLM_INVALID_ARGUMENT : SSLM_ARTIFACT_REJECTED,
               "5.11 share refusal");
        PKV_CHECK(out==nullptr);
        uint32_t free=0,shared=0,mapped=0,reserve=0;
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve)==1,
                      "5.11 share slot not reached");
        PKV_CHECK_EQ(free,111u); // 13 private pages drawn before share.
        PKV_CHECK_EQ(shared,0u); // slot refuses before taking a refcount.
        PKV_CHECK_EQ(mapped,0u);
        PKV_CHECK_EQ(reserve,13u);
        sslm_pc_test_only_restore_fault_disarm();
        ProbeExactlyFreeOneState(fx,&source_pool.pool,124);
    }
    sslm_pc_test_only_restore_fault_arm(2,0);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&source_pool.pool,b,parent,&out),SSLM_KV_POOL_EXHAUSTED,
           "5.11 nested map refusal");
    PKV_CHECK(out==nullptr);
    uint32_t free=0,shared=0,mapped=0,reserve=0;
    PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve),1);
    PKV_CHECK_EQ(free,124u);
    PKV_CHECK_EQ(shared,62u);
    PKV_CHECK_EQ(mapped,62u);
    PKV_CHECK_EQ(reserve,0u);
    sslm_pc_test_only_restore_fault_disarm();
    ProbeExactlyFreeOneState(fx,&source_pool.pool,124);
    Status(ReleasePrefix(source.h,child),SSLM_OK,"5.11 release built child");
    Status(ReleasePrefix(source.h,parent),SSLM_OK,"5.11 release parent");
    ProbeExactlyFreeOneState(fx,&source_pool.pool,200);
    PKV_CHECK_EQ(sslm_kv_pool_destroy(source_pool.pool),SSLM_OK);
    source_pool.pool=nullptr;
}
PKV_CELL("5.11/E3", "E1", NestedRefusals511);

void NestedAdmission74() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix built_parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix built_child=built_parent ? Child(fx,built_parent,source.h) : nullptr;
    if(!built_child) return;
    const auto b=Save(built_child);
    if(!RequireBlob(b,"7.4 nested")) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,76); // 63 parent pages + exactly 13 free.
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    if(!parent) return;
    sslm_prefix out=nullptr;
    uint32_t shared=0;
    if(Restored(fx,&pool.pool,b,parent,sc.h,&out,&shared)) {
        PKV_CHECK_EQ(shared,62u);
        sslm_seq extra=nullptr;
        Status(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&extra),
               SSLM_KV_POOL_EXHAUSTED,"7.4 no two pages after exact draw");
    }
}
PKV_CELL("7.4/E3", "E1", NestedAdmission74);

void Concurrent32() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,500);
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child=parent ? Child(fx,parent,sc.h) : nullptr;
    if(!child) return;
    const auto b=Save(child);
    if(!RequireBlob(b,"3.2")) return;
    const auto parent_pages=PrefixTable(parent);
    const auto child_tokens=Decode(fx,&pool.pool,child,true,16);
    PKV_CHECK_MSG(parent_pages.size()>=62 && child_tokens.size()==16,
                  "3.2 prefill twin or parent page table is unavailable");
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for(int i=0;i<8;++i) threads.emplace_back([&]{
        sslm_prefix copy=nullptr;
        uint32_t shared=0;
        if(sslm_prefix_restore(fx.model,&pool.pool,b.data(),b.size(),parent,&copy,&shared)!=SSLM_OK)
            {++bad;return;}
        if(shared!=62) ++bad;
        const auto copy_pages=PrefixTable(copy);
        if(copy_pages.size()<62 || parent_pages.size()<62 ||
           !std::equal(parent_pages.begin(),parent_pages.begin()+62,copy_pages.begin())) ++bad;
        sslm_seq seq=nullptr;
        if(sslm_seq_create_budgeted(fx.model,&pool.pool,32,&seq)!=SSLM_OK) ++bad;
        else {
            std::vector<int32_t> got;
            if(sslm_seq_adopt_prefix(seq,copy)!=SSLM_OK ||
               DecodeN(fx.model,seq,16,&got)!=SSLM_OK || got!=child_tokens) ++bad;
            sslm_seq_release(seq);
        }
        sslm_prefix_release(copy);
    });
    for(int i=0;i<8;++i) threads.emplace_back([&]{
        sslm_seq seq=nullptr;
        if(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&seq)!=SSLM_OK) {++bad;return;}
        if(sslm_seq_adopt_prefix(seq,parent)!=SSLM_OK) ++bad;
        sslm_seq_release(seq);
        sslm_prefix extension=nullptr;
        if(sslm_prefix_begin_from(parent,16,&extension)!=SSLM_OK) {++bad;return;}
        sslm_prefix_release(extension);
    });
    for(auto& thread:threads) thread.join();
    PKV_CHECK_MSG(bad==0,"3.2 concurrent re-share failures: %d",int(bad));
    sc.h.ReleaseAll();
    ProbeExactlyFreeOneState(fx,&pool.pool,500);
}
PKV_CELL("3.2/E3", "E1", Concurrent32);

void ReleaseOrders12() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix source_parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix source_child=source_parent ? Child(fx,source_parent,source.h) : nullptr;
    if(!source_child) return;
    const auto b=Save(source_child);
    if(!RequireBlob(b,"1.2")) return;
    std::array<int,3> order{0,1,2}; // parent, child, both adopters.
    do {
        PagePool pool(fx.model,200);
        Handles h;
        sslm_prefix parent=Root(fx,&pool.pool,h,1000,61);
        sslm_prefix child=nullptr;
        if(!Restored(fx,&pool.pool,b,parent,h,&child)) return;
        Handles refs;
        sslm_seq a=BudgetSeq(fx,&pool.pool,128,h);
        sslm_seq c=BudgetSeq(fx,&pool.pool,128,h);
        sslm_seq ra=BudgetSeq(fx,&source_pool.pool,128,refs);
        sslm_seq rc=BudgetSeq(fx,&source_pool.pool,128,refs);
        if(!a || !c || !ra || !rc) return;
        Status(sslm_seq_adopt_prefix(a,child),SSLM_OK,"1.2 budget adopter");
        Status(sslm_seq_adopt_prefix(c,child),SSLM_OK,"1.2 second adopter");
        Status(sslm_seq_adopt_prefix(ra,source_child),SSLM_OK,"1.2 twin adopter a");
        Status(sslm_seq_adopt_prefix(rc,source_child),SSLM_OK,"1.2 twin adopter b");
        bool adopters_live=true;
        for(int item:order) {
            if(item==0) Status(ReleasePrefix(h,parent),SSLM_OK,"1.2 release parent");
            if(item==1) Status(ReleasePrefix(h,child),SSLM_OK,"1.2 release child");
            if(item==2) {
                Status(ReleaseSeq(h,a),SSLM_OK,"1.2 release adopter a");
                Status(ReleaseSeq(h,c),SSLM_OK,"1.2 release adopter b");
                adopters_live=false;
            }
            if(adopters_live) {
                std::vector<int32_t> x,y,rx,ry;
                PKV_CHECK_EQ(DecodeN(fx.model,a,16,&x),SSLM_OK);
                PKV_CHECK_EQ(DecodeN(fx.model,c,16,&y),SSLM_OK);
                PKV_CHECK_EQ(DecodeN(fx.model,ra,16,&rx),SSLM_OK);
                PKV_CHECK_EQ(DecodeN(fx.model,rc,16,&ry),SSLM_OK);
                PKV_CHECK_MSG(x.size()==16 && x==rx && y==ry,
                              "1.2 adopters differ from prefill twins after release order");
            }
        }
        h.ReleaseAll();
        ProbeExactlyFreeOneState(fx,&pool.pool,200);
        PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
        pool.pool=nullptr;
    } while(std::next_permutation(order.begin(),order.end()));
}
PKV_CELL("1.2/E3", "E1", ReleaseOrders12);
}
