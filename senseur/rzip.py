import sys, io, zipfile, urllib.request
class R(io.RawIOBase):
    def __init__(s,u):
        s.u=u; s.p=0
        req=urllib.request.Request(u,method='HEAD'); s.n=int(urllib.request.urlopen(req).headers['Content-Length'])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.p
    def seek(s,o,w=0):
        s.p = o if w==0 else (s.p+o if w==1 else s.n+o); return s.p
    def readinto(s,b):
        if s.p>=s.n: return 0
        e=min(s.p+len(b),s.n)-1
        d=urllib.request.urlopen(urllib.request.Request(s.u,headers={'Range':f'bytes={s.p}-{e}'})).read()
        b[:len(d)]=d; s.p+=len(d); return len(d)
z=zipfile.ZipFile(io.BufferedReader(R(sys.argv[1]),buffer_size=1<<20))
if len(sys.argv)>2:
    for name in sys.argv[3:]:
        z.extract(name, sys.argv[2])
else:
    for i in z.infolist(): print(i.file_size, i.compress_size, i.filename)
