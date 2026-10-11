/* Real tokenizer/tool ingestion and interrupt/resume on one reused session.
 * Full-vocabulary and canonical-state oracles live on disk, not a second GPU
 * session. Only synthetic HTML is retained in the temporary private store. */
#include "../qw3_metal.m"
#define qw3_metal_env_n_gpu_layers pressure_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled pressure_core_llama_split_enabled
#define qw3_metal_llama_split_start pressure_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
#define main pressure_agent_main
#include "../qw3_agent.c"
#undef main

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while (0)

static void pressure_cleanup(const char *root) {
    DIR *d=opendir(root);struct dirent *entry;
    while (d && (entry=readdir(d))) {
        if (!strcmp(entry->d_name,".") || !strcmp(entry->d_name,"..")) continue;
        char *p=path_join(root,entry->d_name);struct stat st;
        if (p && !lstat(p,&st)) {
            if (S_ISDIR(st.st_mode)) pressure_cleanup(p);else (void)unlink(p);
        }
        free(p);
    }
    if (d) closedir(d);
    (void)rmdir(root);
}

static void pressure_discard(void *ud,const char *text,size_t len) {
    (void)ud;(void)text;(void)len;
}

static int pressure_files(const char *dir,const char *prefix) {
    DIR *d=opendir(dir);struct dirent *entry;int n=0;
    while (d && (entry=readdir(d))) if (!strncmp(entry->d_name,prefix,strlen(prefix))) n++;
    if (d) closedir(d);
    return n;
}

static int pressure_tools(qw3_engine *e,const char *root) {
    agent_state a={0};a.engine=e;a.cfg.ctx_size=32768;a.cfg.store_dir=(char *)root;
    a.cfg.tools_enabled=true;a.status_write=pressure_discard;a.output_write=pressure_discard;
    strbuf body;sb_init(&body);sb_append(&body,"HEAD_MARKER\n");
    for (int i=0;i<512;i++) sb_printf(&body,
        "<li id=\"mw%d\" data-mw='{\"params\":{\"title\":{\"wt\":\"Å漢🙂\"},\"volume\":{\"wt\":\"91\"}},\"i\":%d}'>HTML</li>\n",i,i);
    sb_append(&body,"TAIL_MARKER\nexit=7\n");
    char *full=native_tool_response_text("read",body.p);
    int old_tokens=agent_rendered_message_tokens(&a,full);free(full);
    CHECK(old_tokens>10000);
    char *last=NULL;
    for (int i=0;i<12;i++) {
        char *response=agent_context_tool_response(&a,i&1 ? "dsml" : "read",body.p);
        CHECK(response && agent_rendered_message_tokens(&a,response)<=2048);
        CHECK(strstr(response,"HEAD_MARKER") && strstr(response,"TAIL_MARKER") && strstr(response,"exit=7"));
        CHECK([NSString stringWithUTF8String:response]!=nil);
        const char *begin=strstr(response,"full result saved to ");CHECK(begin);
        begin+=strlen("full result saved to ");const char *end=strstr(begin,". Use rg/sed");CHECK(end);
        free(last);last=malloc((size_t)(end-begin)+1);CHECK(last);
        memcpy(last,begin,(size_t)(end-begin));last[end-begin]=0;
        struct stat st;CHECK(!stat(last,&st) && (st.st_mode&0777)==0600);
        char *saved=read_file_text(last,NULL);CHECK(saved && !strcmp(saved,body.p));free(saved);
        free(response);
    }
    char *spills=path_join(root,"spills");CHECK(pressure_files(spills,"tool-")==8);
    printf("PASS tool_limit raw_tokens=%d cap=2048 UTF8=valid full_result=exact retention=8 mode=0600 native_and_dsml=bounded\n",old_tokens);

    a.cfg.ctx_size=4096;
    char *small=agent_context_tool_response(&a,"bash","SMALL\nexit=0\n");
    full=native_tool_response_text("bash","SMALL\nexit=0\n");CHECK(small && !strcmp(small,full));free(small);free(full);
    char *response=agent_context_tool_response(&a,"read",body.p);
    CHECK(response && agent_rendered_message_tokens(&a,response)<=512);free(response);
    char *blocked=path_join(root,"not-a-directory");CHECK(!write_file_text(blocked,"fixture"));a.cfg.store_dir=blocked;
    response=agent_context_tool_response(&a,"read",body.p);
    CHECK(response && agent_rendered_message_tokens(&a,response)<=512 && strstr(response,"full result could not be saved"));
    CHECK(!strstr(response,"full result saved to"));free(response);free(blocked);

    a.cfg.ctx_size=32768;a.cfg.store_dir=(char *)root;
    for (int i=0;i<12504;i++) qw3_tokens_push(&a.transcript,0);
    CHECK(agent_message_ledger_add(&a.messages,AGENT_MESSAGE_SYSTEM,"system","fixture",0,12504));
    char *input=path_join(root,"dense-html.txt");CHECK(!write_file_text(input,body.p));
    tool_call_list calls={0};calls.n_calls=1;strcpy(calls.calls[0].name,"read");
    calls.calls[0].n_params=1;strcpy(calls.calls[0].params[0].name,"path");calls.calls[0].params[0].value=agent_strdup(input);
    int before=a.transcript.len;CHECK(execute_native_tools_append(&a,&calls,NULL));
    CHECK(a.transcript.len>before && a.transcript.len<=before+2048 && !agent_should_compact_context(&a));
    CHECK(a.messages.len==2 && a.messages.items[1].kind==AGENT_MESSAGE_TOOL);
    CHECK(pressure_files(spills,"context-")==0);
    printf("PASS native_read old=12504 new=%d appended=%d source_result_tokens=%d compaction=not_needed\n",a.transcript.len,a.transcript.len-before,old_tokens);
    free_tool_calls(&calls);free(input);free(last);free(spills);free(a.last_read_path);sb_free(&body);
    qw3_tokens_free(&a.transcript);agent_message_ledger_clear(&a.messages);
    return 0;
}

static int pressure_reads(qw3_engine *e,const char *root) {
    agent_state a={0};a.engine=e;a.cfg.ctx_size=4096;
    char *path=path_join(root,"read-pages.txt");strbuf fixture;sb_init(&fixture);
    for (int i=1;i<=120;i++) sb_printf(&fixture,
        "PAGE_%03d α beta γ delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron\n",i);
    CHECK(!write_file_text(path,fixture.p));sb_free(&fixture);
    tool_call call={0};strcpy(call.name,"read");call.n_params=3;
    strcpy(call.params[0].name,"path");call.params[0].value=path;
    strcpy(call.params[1].name,"start");call.params[1].value="1";
    strcpy(call.params[2].name,"lines");call.params[2].value="400";
    char *out=tool_read(&a,&call);CHECK(out && strstr(out,"next_start="));
    int low=a.last_read_next-1;CHECK(low>0 && low<120);
    char *wrapped=native_tool_response_text("read",out);
    CHECK(agent_rendered_message_tokens(&a,wrapped)<=512);free(wrapped);free(out);
    tool_call more={0};strcpy(more.name,"more");int next=low+1;
    while (next<=120) {
        out=tool_more(&a,&more);CHECK(out && a.last_read_next>next);
        wrapped=native_tool_response_text("more",out);
        CHECK(agent_rendered_message_tokens(&a,wrapped)<=512);free(wrapped);
        for (int i=next;i<a.last_read_next;i++) {
            char marker[64];snprintf(marker,sizeof(marker),"%5d  PAGE_%03d ",i,i);
            CHECK(strstr(out,marker));
        }
        char marker[64];snprintf(marker,sizeof(marker),"PAGE_%03d ",a.last_read_next);
        CHECK(!strstr(out,marker));next=a.last_read_next;free(out);
    }
    CHECK(next==121);
    a.cfg.ctx_size=32768;out=tool_read(&a,&call);CHECK(out);
    int high=a.last_read_next-1;CHECK(high>low);
    wrapped=native_tool_response_text("read",out);
    CHECK(agent_rendered_message_tokens(&a,wrapped)<=2048);free(wrapped);free(out);

    sb_init(&fixture);for (int i=0;i<2000;i++) sb_append(&fixture,"OVERSIZED ");
    sb_append(&fixture,"\nEND_NEXT\n");CHECK(!write_file_text(path,fixture.p));sb_free(&fixture);
    a.cfg.ctx_size=4096;out=tool_read(&a,&call);
    CHECK(out && a.last_read_next==1 && strstr(out,"line alone is too long") && !strstr(out,"OVERSIZED"));free(out);
    call.params[1].value="2";out=tool_read(&a,&call);
    CHECK(out && a.last_read_next==3 && strstr(out,"END_NEXT"));free(out);
    printf("PASS adaptive_read ctx4096_lines=%d ctx32768_lines=%d continuation=all_120_lines_contiguous oversized_line=not_skipped\n",low,high);
    free(a.last_read_path);free(path);return 0;
}

static int pressure_blob(FILE *f,int record,const void *data,size_t bytes) {
    if (record) return fwrite(data,1,bytes,f)==bytes;
    unsigned char expected[65536];
    return bytes<=sizeof(expected) && fread(expected,1,bytes,f)==bytes && !memcmp(expected,data,bytes);
}

static int pressure_logits(FILE *f,int record,qw3_session *s,int vocab) {
    float data[16384];
    for (int off=0;off<vocab;off+=16384) {
        int n=MIN(16384,vocab-off);
        CHECK(qw3_metal_session_read_state(s->metal,QW3_METAL_STATE_LOGITS,0,(uint64_t)off*4,data,(uint64_t)n*4));
        for (int i=0;i<n;i++) CHECK(isfinite(data[i]));
        CHECK(pressure_blob(f,record,data,(size_t)n*4));
    }
    return 0;
}

static int pressure_state(FILE *f,int record,qw3_session *s) {
    qw3_metal_state_info info={0};unsigned char data[65536];
    CHECK(qw3_metal_session_get_state_info(s->metal,&info) && info.kv_type==1);
    int header[2]={s->tokens.len,s->kv.pos};CHECK(s->tokens.len<=256 && s->kv.pos==s->tokens.len);
    CHECK(pressure_blob(f,record,header,sizeof(header)) && pressure_blob(f,record,s->tokens.v,(size_t)s->tokens.len*sizeof(int)));
    for (int kind=QW3_METAL_STATE_GQA_K;kind<=QW3_METAL_STATE_CONV;kind++) {
        if (kind==QW3_METAL_STATE_GQA_FLASH_TAIL) continue;
        uint64_t bytes=kind==QW3_METAL_STATE_DELTANET ? info.deltanet_bytes : kind==QW3_METAL_STATE_CONV ? info.conv_bytes : info.kv_row_bytes*s->tokens.len;
        uint32_t layers=kind<=QW3_METAL_STATE_GQA_V ? info.n_full_layers : 1u;
        for (uint32_t l=0;l<layers;l++) for (uint64_t off=0;off<bytes;off+=sizeof(data)) {
            size_t n=(size_t)MIN((uint64_t)sizeof(data),bytes-off);
            CHECK(qw3_metal_session_read_state(s->metal,(qw3_metal_state_buffer)kind,l,off,data,n) && pressure_blob(f,record,data,n));
        }
    }
    return 0;
}

typedef struct {bool stop;int completed;int total;int ended;} pressure_interrupt;
static bool pressure_stop(void *ud) {return ((pressure_interrupt *)ud)->stop;}
static void pressure_progress(void *ud,const char *event,int current,int total) {
    pressure_interrupt *p=ud;
    if (!strcmp(event,"prefill_chunk")) {p->completed=current;p->total=total;if (current>=128) p->stop=true;}
}
static void pressure_agent_progress(void *ud,const char *phase,int current,int total,double tps) {
    (void)tps;pressure_interrupt *p=ud;
    if (!strcmp(phase,"prefill")) pressure_progress(p,"prefill_chunk",current,total);
    if (!strcmp(phase,"prefill_done")) p->ended++;
}

static int pressure_sync(qw3_engine *e) {
    qw3_session *s=NULL;qw3_tokens prompt={0};char err[256]={0};uint32_t rng=1;
    CHECK(!qw3_session_create(&s,e,256) && s->metal_n_gpu_layers==64);
    int vocab=qw3_vocab_size(e);
    for (int i=0;i<192;i++) {rng=rng*1664525u+1013904223u;int id=rng%(uint32_t)vocab;if (id==qw3_token_eos(e)) id=(id+1)%vocab;qw3_tokens_push(&prompt,id);}
    FILE *oracle=tmpfile();CHECK(oracle);
    for (int phase=0;phase<3;phase++) {
        qw3_session_invalidate(s);pressure_interrupt p={0};
        if (phase) CHECK(!fflush(oracle) && !fseek(oracle,0,SEEK_SET));
        if (phase==2) {
            qw3_session_set_progress(s,pressure_progress,&p);
            CHECK(qw3_session_sync_interruptible(s,&prompt,pressure_stop,&p,err,sizeof(err))==-2);
            CHECK(p.completed==128 && p.total==192 && qw3_session_pos(s)==128 && s->tokens.len==128 && s->valid);
            CHECK(!memcmp(s->tokens.v,prompt.v,128*sizeof(int)));
            qw3_session_set_progress(s,NULL,NULL);p.stop=false;
            CHECK(!qw3_session_sync_interruptible(s,&prompt,pressure_stop,&p,err,sizeof(err)));
        } else if (phase==1) CHECK(!qw3_session_sync_interruptible(s,&prompt,pressure_stop,&p,err,sizeof(err)));
        else CHECK(!qw3_session_sync(s,&prompt,err,sizeof(err)));
        CHECK(!pressure_logits(oracle,phase==0,s,vocab) && !pressure_state(oracle,phase==0,s));
        for (int i=0;i<32;i++) {
            int token=qw3_session_argmax(s),expected=token;CHECK(token>=0);
            if (!phase) CHECK(fwrite(&token,sizeof(token),1,oracle)==1);
            else CHECK(fread(&expected,sizeof(expected),1,oracle)==1 && expected==token);
            CHECK(!qw3_session_eval_gpu_logits(s,token,err,sizeof(err)) && !pressure_logits(oracle,phase==0,s,vocab));
        }
        CHECK(!pressure_state(oracle,phase==0,s));if (phase) CHECK(fgetc(oracle)==EOF && !ferror(oracle));
        printf("PASS sync phase=%d prefix=192 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact canceled_at=%d sessions=1\n",phase,phase==2 ? 128 : 0);fflush(stdout);
    }
    FILE *state=tmpfile();CHECK(state && !pressure_state(state,1,s) && !fflush(state) && !fseek(state,0,SEEK_SET));
    pressure_interrupt p={.stop=true};
    CHECK(qw3_session_sync_interruptible(s,&prompt,pressure_stop,&p,err,sizeof(err))==-2 && !pressure_state(state,0,s));fclose(state);
    qw3_session_invalidate(s);p=(pressure_interrupt){0};
    agent_state a={0};a.engine=e;a.session=s;a.should_interrupt=pressure_stop;a.interrupt_ud=&p;
    a.progress_update=pressure_agent_progress;a.progress_ud=&p;a.status_write=pressure_discard;
    CHECK(agent_sync_with_progress(&a,&prompt,err,sizeof(err))==-2 && p.ended==1 && s->tokens.len==128 && !s->progress_fn);
    qw3_session_invalidate(s);p=(pressure_interrupt){0};qw3_tokens_copy(&a.transcript,&prompt);
    char *assistant=NULL;
    CHECK(generate_once(&a,&assistant,true)==-1 && !assistant && a.transcript.len==192 && !memcmp(a.transcript.v,prompt.v,192*sizeof(int)) && p.ended==1 && !s->progress_fn);
    printf("PASS agent_prefill interrupt=observed progress=cleared transcript=restored immediate_cancel_state=exact\n");
    qw3_tokens_free(&a.transcript);fclose(oracle);qw3_tokens_free(&prompt);qw3_session_free(s);return 0;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        char root[]="/tmp/qw3-context-pressure-XXXXXX";if (!mkdtemp(root)) return 1;
        qw3_engine *e=NULL;qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        int rc=qw3_engine_open(&e,&opt);
        if (!rc) rc=pressure_reads(e,root);
        if (!rc) rc=pressure_tools(e,root);
        if (!rc) rc=pressure_sync(e);
        if (e) qw3_engine_close(e);pressure_cleanup(root);
        return rc;
    }
}
