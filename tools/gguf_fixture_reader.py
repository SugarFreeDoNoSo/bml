"""Independent strict GGUF v3 subset reader for bounded single-F32-tensor tests.

No imports from BML or the fixture writer. Not a general model/inference loader.
"""
import math
import struct

def read_fixture(data, max_dimension=128, decode_values=True):
    pos=0
    def take(n):
        nonlocal pos
        if n<0 or pos+n>len(data):
            raise ValueError('truncated GGUF')
        b=data[pos:pos+n];pos+=n;return b
    def number(fmt):
        return struct.unpack('<'+fmt,take(struct.calcsize('<'+fmt)))[0]
    def string():
        n=number('Q')
        if n>65536: raise ValueError('oversized test metadata string')
        return take(n).decode('utf-8')
    if take(4)!=b'GGUF' or number('I')!=3: raise ValueError('not GGUF v3 LE')
    nt,nkv=number('Q'),number('Q')
    if nt!=1 or nkv>32: raise ValueError('expected one bounded test tensor')
    metadata={}
    for _ in range(nkv):
        k=string(); t=number('I')
        if t!=8: raise ValueError('test subset requires string metadata')
        if k in metadata: raise ValueError('duplicate metadata')
        metadata[k]=string()
    descriptor_offset=pos
    name=string(); rank=number('I')
    if rank!=2: raise ValueError('rank must be 2')
    ne0,ne1=number('Q'),number('Q')
    if not 1<=ne0<=max_dimension or not 1<=ne1<=max_dimension: raise ValueError('bounded test reader dimension limit')
    if number('I')!=0: raise ValueError('test subset supports F32 only')
    relative=number('Q')
    if relative%32: raise ValueError('unaligned tensor offset')
    start=(pos+31)//32*32+relative
    if start+4*ne0*ne1!=len(data): raise ValueError('incorrect tensor range/file size')
    values=list(struct.unpack_from('<'+str(ne0*ne1)+'f',data,start)) if decode_values else []
    if not all(math.isfinite(v) for v in values): raise ValueError('non-finite weights')
    return {'name':name,'dims':[ne0,ne1],'dtype':'F32','alignment':32,'data_offset':start,'descriptor_offset':descriptor_offset,'metadata':metadata,'values':values}
