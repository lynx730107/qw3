from pathlib import Path
import tempfile, subprocess, shutil, re, os
root=Path.cwd()
work=Path(tempfile.mkdtemp(prefix='qw3-projection-'))
arc=root/'docs/measurements/decode_projection_barriers_2026-10-14/source_snapshot/tests/.decode_projection_barriers'
private=work/'tests/.decode_projection_barriers'
shutil.copytree(arc,private)
for p in [*root.glob('qw3*.h'),root/'qw3_bench.c']:
 shutil.copy2(p,work/p.name)
shutil.copytree(root/'metal',work/'metal')
old=subprocess.check_output(['git','show','8444157:qw3.c'])
(work/'base.c').write_bytes(old)
merged=subprocess.check_output(['git','merge-file','-p',str(root/'qw3.c'),str(work/'base.c'),str(private/'qw3_private.c')])
(private/'qw3_private.c').write_bytes(merged)
def instrument(s):
 return re.sub(r'\[(\w+) dispatch(Threadgroups|Threads):',r'[(id<QW3FFNBarrierHooks>)\1 privateDispatch\2:',s)
assert instrument(subprocess.check_output(['git','show','8444157:qw3_metal.m']).decode())==(private/'qw3_private_metal.m').read_text()
(private/'qw3_private_metal.m').write_text(instrument((root/'qw3_metal.m').read_text()))
for suffix,src in [('bench','bench_decode_projection_barriers_native.m'),('quality','test_decode_projection_barriers_model.m')]:
 shutil.copy2(root/'tests'/src,work/'tests'/src)
 subprocess.run(['cc','-O3','-ffast-math','-fno-finite-math-only','-mcpu=native','-fobjc-arc','-I.', '-I'+str(root), 'tests/'+src,'-framework','Foundation','-framework','Metal','-o',suffix],cwd=work,check=True)
print(work)
