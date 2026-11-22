from array import array
import math
import sys
from pathlib import Path
def read(path):
    a=array('f')
    a.frombytes(Path(path).read_bytes())
    assert len(a)==4*248320
    return a
a,b=map(read,sys.argv[1:3])
for i in range(4):
    lo,hi=i*248320,(i+1)*248320
    x,y=a[lo:hi],b[lo:hi]
    delta=[abs(v-w) for v,w in zip(x,y)]
    assert all(map(math.isfinite,delta))
    print(i, 'rmse',math.sqrt(sum(z*z for z in delta)/len(x)),'max',max(delta),'argmax',x.index(max(x)),y.index(max(y)))
