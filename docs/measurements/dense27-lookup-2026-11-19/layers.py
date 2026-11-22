"""Extend the isolated ablation to layer ranges during the eight-token block."""
from pathlib import Path
import subprocess
import sys
dst=Path(sys.argv[1])
subprocess.run([sys.executable,str(Path(__file__).with_name('ablate.py')),str(dst)],check=True)
p=dst/'ablation_core.c'
s=p.read_text().replace('static int probe_mask;', 'static int probe_mask, probe_lo, probe_hi;')
for bit in (1,2):
 s=s.replace(f'!(probe_mask & {bit})',f'!((probe_mask & {bit}) && il>=probe_lo && il<probe_hi)')
p.write_text(s)
p=dst/'ablation.c';s=p.read_text().replace(
    'for (int rep=0; rep<4; rep++) {\n        probe_mask=rep;',
    '''const int lo[]={0,0,0,0,0,16,32,48};
    const int hi[]={0,8,16,32,48,64,64,64};
    for (int rep=0; rep<8; rep++) {
        probe_mask=3; probe_lo=lo[rep]; probe_hi=hi[rep];''')
p.write_text(s)
