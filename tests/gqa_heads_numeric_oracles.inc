/* One session, disk oracle, fixed-size CPU chunks; no extra weight/session buffers. */
#include "optimization_acceptance.h"
#include <float.h>

static int grid_require_exact_state;
static qw3_opt_error grid_logit_error,grid_state_error;
static uint64_t grid_argmax_changes,grid_logit_vectors;

static int grid_oracle_blob(FILE *oracle,int record,const void *data,size_t bytes) {
    if (record) return fwrite(data,1,bytes,oracle)==bytes;
    unsigned char expected[65536];
    return bytes<=sizeof(expected) && fread(expected,1,bytes,oracle)==bytes &&
           !memcmp(expected,data,bytes);
}

static int grid_oracle_logits(FILE *oracle,int record,qw3_session *s,int vocab) {
    float data[16384],expected[16384];qw3_opt_error error={0};
    float ref_best=-FLT_MAX,ref_second=-FLT_MAX,actual_best=-FLT_MAX;
    int ref_id=-1,actual_id=-1;
    for (int off=0;off<vocab;off+=16384) {
        int count=MIN(16384,vocab-off);
        if (!qw3_metal_session_read_state(s->metal,QW3_METAL_STATE_LOGITS,0,
                (uint64_t)off*4u,data,(uint64_t)count*4u)) return 0;
        if (record) {
            for (int i=0;i<count;i++) if (!isfinite(data[i])) return 0;
            if (fwrite(data,4,(size_t)count,oracle)!=(size_t)count) return 0;
            continue;
        }
        if (fread(expected,4,(size_t)count,oracle)!=(size_t)count) return 0;
        for (int i=0;i<count;i++) {
            qw3_opt_error_add(&error,expected[i],data[i]);
            if (!grid_require_exact_state) qw3_opt_error_add(&grid_logit_error,expected[i],data[i]);
            if (expected[i]>ref_best) {ref_second=ref_best;ref_best=expected[i];ref_id=off+i;}
            else if (expected[i]>ref_second) ref_second=expected[i];
            if (data[i]>actual_best) {actual_best=data[i];actual_id=off+i;}
        }
    }
    if (record) return 1;
    if (grid_require_exact_state) return !error.nonfinite && !error.different_bits;
    grid_logit_vectors++;grid_argmax_changes+=(ref_id!=actual_id);
    int valid=qw3_opt_logits_valid(&error);
    printf("logits_numeric valid=%d rmse=%.9g max_abs=%.9g nmse=%.9g different_bits=%llu argmax_match=%d baseline_margin=%.9g\n",
        valid,qw3_opt_error_rmse(&error),error.max_abs,qw3_opt_error_nmse(&error),
        (unsigned long long)error.different_bits,ref_id==actual_id,(double)ref_best-ref_second);
    return valid;
}

static int grid_oracle_numeric_state(FILE *oracle,int record,const void *data,size_t bytes,int half_values) {
    if (record) return fwrite(data,1,bytes,oracle)==bytes;
    unsigned char expected[65536];qw3_opt_error error={0};
    if (bytes>sizeof(expected) || fread(expected,1,bytes,oracle)!=bytes) return 0;
    size_t stride=half_values ? 2u : 4u;
    if (bytes%stride) return 0;
    for (size_t off=0;off<bytes;off+=stride) {
        float a,b;
        if (half_values) {
            __fp16 ha,hb;memcpy(&ha,expected+off,2);memcpy(&hb,(const unsigned char *)data+off,2);
            a=(float)ha;b=(float)hb;
        } else {memcpy(&a,expected+off,4);memcpy(&b,(const unsigned char *)data+off,4);}
        qw3_opt_error_add(&error,a,b);
        if (!grid_require_exact_state) qw3_opt_error_add(&grid_state_error,a,b);
    }
    if (error.nonfinite) return 0;
    if (grid_require_exact_state) return !memcmp(expected,data,bytes);
    return qw3_opt_error_nmse(&error)<=QW3_OPT_STATE_NMSE_MAX;
}

static int grid_oracle_state(FILE *oracle,int record,qw3_session *s) {
    qw3_metal_state_info info={0};unsigned char data[65536];
    if (!qw3_metal_session_get_state_info(s->metal,&info) || info.kv_type!=1u) return 0;
    int32_t header[2]={s->tokens.len,s->kv.pos};
    if (!grid_oracle_blob(oracle,record,header,sizeof(header)) ||
        s->tokens.len>256 || s->kv.pos!=s->tokens.len ||
        !grid_oracle_blob(oracle,record,s->tokens.v,(size_t)s->tokens.len*sizeof(int))) return 0;
    for (int kind=QW3_METAL_STATE_GQA_K;kind<=QW3_METAL_STATE_CONV;kind++) {
        if (kind==QW3_METAL_STATE_GQA_FLASH_TAIL) continue;
        uint64_t bytes=kind==QW3_METAL_STATE_DELTANET ? info.deltanet_bytes :
            kind==QW3_METAL_STATE_CONV ? info.conv_bytes : info.kv_row_bytes*s->tokens.len;
        uint32_t layers=kind<=QW3_METAL_STATE_GQA_V ? info.n_full_layers : 1u;
        for (uint32_t l=0;l<layers;l++) for (uint64_t off=0;off<bytes;off+=sizeof(data)) {
            size_t n=(size_t)MIN((uint64_t)sizeof(data),bytes-off);
            if (!qw3_metal_session_read_state(s->metal,(qw3_metal_state_buffer)kind,l,off,data,n) ||
                !grid_oracle_numeric_state(oracle,record,data,n,kind<=QW3_METAL_STATE_GQA_V)) {
                fprintf(stderr,"GQA numeric state failed kind=%d layer=%u offset=%llu\n",
                    kind,l,(unsigned long long)off);return 0;
            }
        }
    }
    return 1;
}

static void grid_numeric_report(void) {
    printf("numeric_policy=%d logits_vectors=%llu logits_rmse=%.9g logits_max_abs=%.9g logits_different_bits=%llu argmax_changes=%llu state_nmse=%.9g state_different_bits=%llu history=exact checkpoint_policy=self_roundtrip_bit_exact agent_behavior=not_tested\n",
        QW3_OPT_ACCEPTANCE_VERSION,(unsigned long long)grid_logit_vectors,
        qw3_opt_error_rmse(&grid_logit_error),grid_logit_error.max_abs,
        (unsigned long long)grid_logit_error.different_bits,(unsigned long long)grid_argmax_changes,
        qw3_opt_error_nmse(&grid_state_error),(unsigned long long)grid_state_error.different_bits);
}
