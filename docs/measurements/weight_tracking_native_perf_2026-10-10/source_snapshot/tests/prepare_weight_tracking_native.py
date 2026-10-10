"""Private native benchmark: one GGUF buffer set, no alias-view switch."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'tests/.weight_tracking';out.mkdir(exist_ok=True)
source=(root/'qw3_metal.m').read_text()
needle='newBufferWithBytesNoCopy:'
position=0;count=0
while True:
    start=source.find(needle,position)
    if start<0:break
    end=source.index('deallocator:nil]',start)
    part=source[start:end]
    assert part.count('options:MTLResourceStorageModeShared')==1
    part=part.replace('options:MTLResourceStorageModeShared','options:weight_native_options()')
    source=source[:start]+part+source[end:]
    position=start+len(part);count+=1
assert count==2
(out/'native_core.inc').write_text(source)
