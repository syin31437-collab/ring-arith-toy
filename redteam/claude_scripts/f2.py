from rtc import *
import os,re,collections
os.chdir(REPO+"/bipm")
import bipm as B, events as E
cur=None;mis=collections.Counter();tot=collections.Counter();ex={}
for line in open("data/notes.utc",errors="ignore"):
    m=re.match(r"\s{3,6}([A-Z][A-Z0-9]{1,4})\s*\.",line)
    if m: cur=m.group(1).lower()
    if cur is None: continue
    mj=re.findall(r"MJD[^0-9]{0,8}(\d{5})",line)[:1]
    if mj and int(mj[0])>=B.SANDBOX_END:
        tot[cur]+=1
        o=re.findall(r"UTC\(([A-Z0-9]+)\)",line)
        if o and o[0].lower()!=cur: mis[(cur,o[0].lower())]+=1; ex[(cur,o[0].lower())]=line.strip()[:100]
print("测试段事件总数",sum(tot.values()),"; 段落标签≠行内 UTC(X) 的:",sum(mis.values()),mis.most_common(12))
for k,v in list(ex.items())[:6]: print(k,v)
# 段落内只含“次行”的 MJD（无 UTC(X)）在 za/dmdm
print("za 事件的 MJD 年范围",tot["za"]);
