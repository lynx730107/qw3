/* Private native-layout IQ4_XS prefetch. Preserve block/lane accumulation
 * order; move only the following block's packed loads ahead of current dots. */
static id<MTLComputePipelineState> prefetch_iq4_pipeline[3];
static int prefetch_iq4_initialize(void) {
    NSString *all=qw3_metal_kernel_source();
    NSString *table=lut_extract(all,@"constant float qw3_iq4nl_table[16]",@"\n};");
    NSString *value=lut_extract(all,@"inline float qw3_iq4nl_val(",@"\n}\n");
    NSMutableString *dot=[lut_extract(all,@"inline float qw3_iq4xs_dot8_vector(",@"\n}\n") mutableCopy];
    if (!table || !value || !dot) return 0;
    NSString *payload=@"struct iq4_payload { half d; ushort scales_h; uchar scales_l; uint2 q; };\n"
        "inline iq4_payload iq4_load(device const uchar *blk, uint ib, uint il) {\n"
        "    iq4_payload p;\n"
        "    p.d = *((device const half *)blk);\n"
        "    p.scales_h = *((device const ushort *)(blk + 2));\n"
        "    p.scales_l = blk[4 + ib / 2u];\n"
        "    device const uint *q = (device const uint *)(blk + 8 + ib * 16u + il * 8u);\n"
        "    p.q = uint2(q[0], q[1]);\n"
        "    return p;\n"
        "}\n";
    NSString *loads=@"const half d = *((device const half *)blk);\n    const ushort scales_h = *((device const ushort *)(blk + 2));\n    device const uchar *scales_l = blk + 4;\n    device const uint *qs = (device const uint *)(blk + 8 + ib * 16u + il * 8u);";
    if (!lut_replace(dot,@"device const uchar *blk,",@"iq4_payload payload,",1) ||
        !lut_replace(dot,loads,@"const half d = payload.d;\n    const ushort scales_h = payload.scales_h;",1) ||
        !lut_replace(dot,@"scales_l[ib / 2u]",@"payload.scales_l",1) ||
        !lut_replace(dot,@"qs[j]",@"payload.q[j]",1)) return 0;
    NSMutableString *source=[@"#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n" mutableCopy];
    [source appendString:table];[source appendString:@"\n"];[source appendString:value];
    [source appendString:payload];[source appendString:dot];
    const char *names[]={"qw3_matvec_iq4_xs_fast","qw3_matvec_iq4_xs_pair_fast","qw3_matvec_iq4_xs_add_x0_fast"};
    for (int kind=0;kind<3;kind++) {
        NSMutableString *body=[lut_extract(all,[@"kernel void " stringByAppendingString:[NSString stringWithUTF8String:names[kind]]],@"\n}\n") mutableCopy];
        if (!body) return 0;
        NSMutableString *initial=[@"uint n_blocks = args.n_in / 256u;\n" mutableCopy];
        NSMutableString *ahead=[@"for (uint b = ix; b < n_blocks; b += 2u) {\n" mutableCopy];
        NSMutableString *advance=[NSMutableString new];
        const int count=kind==1 ? 4 : 2;
        for (int p=0;p<count;p++) {
            const int row=kind==1 ? p/2 : p;
            const char *weights=kind==1 ? (p%2 ? "up_weights" : "gate_weights") : "weights";
            [initial appendFormat:@"    iq4_payload current%d = {};\n    if (ix < n_blocks && first_row + %d < args.n_out) current%d = iq4_load(%s + uint64_t(first_row + %d) * args.row_bytes + uint64_t(ix) * 136ull, ib, il);\n",p,row,p,weights,row];
            [ahead appendFormat:@"        iq4_payload next%d = {};\n        if (b + 2u < n_blocks && first_row + %d < args.n_out) next%d = iq4_load(%s + uint64_t(first_row + %d) * args.row_bytes + uint64_t(b + 2u) * 136ull, ib, il);\n",p,row,p,weights,row];
            [advance appendFormat:@"        current%d = next%d;\n",p,p];
        }
        if (!lut_replace(body,@"uint n_blocks = args.n_in / 256u;",initial,1) ||
            !lut_replace(body,@"for (uint b = ix; b < n_blocks; b += 2u) {",ahead,1)) return 0;
        if (kind==0) {
            NSString *call=@"qw3_iq4xs_dot8_vector(weights +\n                uint64_t(row) * uint64_t(args.row_bytes) + block_off,";
            NSRange first=[body rangeOfString:call];
            if (first.location==NSNotFound) return 0;
            [body replaceCharactersInRange:first withString:@"qw3_iq4xs_dot8_vector(current0,"];
            if (!lut_replace(body,call,@"qw3_iq4xs_dot8_vector(current1,",1)) return 0;
        } else if (kind==1) {
            for (int row=0;row<2;row++) for (int side=0;side<2;side++) {
                NSString *call=[NSString stringWithFormat:@"%s%d += qw3_iq4xs_dot8_vector(%s + off,",side ? "up" : "gate",row,side ? "up_weights" : "gate_weights"];
                NSString *replacement=[NSString stringWithFormat:@"%s%d += qw3_iq4xs_dot8_vector(current%d,",side ? "up" : "gate",row,row*2+side];
                if (!lut_replace(body,call,replacement,1)) return 0;
            }
        } else for (int row=0;row<2;row++) {
            NSString *call=[NSString stringWithFormat:@"sum%d += qw3_iq4xs_dot8_vector(blk,",row];
            NSString *replacement=[NSString stringWithFormat:@"sum%d += qw3_iq4xs_dot8_vector(current%d,",row,row];
            if (!lut_replace(body,call,replacement,1)) return 0;
        }
        NSString *epilogue=kind==1 ? @"    }\n    gate0 = simd_sum(gate0);" : @"    }\n    sum0 = simd_sum(sum0);";
        if (!lut_replace(body,epilogue,[advance stringByAppendingString:epilogue],1)) return 0;
        [source appendString:body];
    }
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
    for (int kind=0;kind<3;kind++) {
        id<MTLFunction> function=[library newFunctionWithName:[NSString stringWithUTF8String:names[kind]]];
        prefetch_iq4_pipeline[kind]=function ? [g_device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!prefetch_iq4_pipeline[kind] || prefetch_iq4_pipeline[kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ4 prefetch pipeline failed kind=%d: %s\n",kind,error ? error.localizedDescription.UTF8String : "");return 0;
        }
    }
    return 1;
}
