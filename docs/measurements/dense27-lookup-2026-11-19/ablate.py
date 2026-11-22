"""Generate an isolated width-eight ablation runner from the checked-out core."""
from pathlib import Path
import sys
dst=Path(sys.argv[1])
source=Path('qw3.c').read_text()
start=source.rindex('static int qw3_metal_session_eval_dense_prefill_batch(')
prefix,body=source[:start],source[start:]
fields={'gate_batch':'1','up_batch':'1','down_batch':'1','batch_linear_qkv':'2','batch_linear_z':'2'}
for field,bit in fields.items():
    key=f'const int {field} = '
    assert body.count(key)==1
    body=body.replace(key,key+f'!(probe_mask & {bit}) && ')
(dst/'ablation_core.c').write_text(prefix+'static int probe_mask;\n'+body)
probe=Path(__file__).with_name('block.c').read_text().replace('#include "qw3.c"','#include "ablation_core.c"')
probe=probe.replace('for (int rep=-1; rep<3; rep++) {','for (int rep=0; rep<4; rep++) {\n        probe_mask=rep;')
(dst/'ablation.c').write_text(probe)
