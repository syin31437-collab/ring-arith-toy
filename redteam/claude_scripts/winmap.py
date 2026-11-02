from rtc import *
import json, slqa as S, validate_bipm as V
E, items = S.load(); cut=int(.3*len(items)); sb,te=items[:cut],items[cut:]
labs=sorted(l for l in {l for i in items for l in E[i]} if sum(l in E[i] for i in items)>=0.6*len(items) and sum(l in E[i] for i in sb)>=0.5*len(sb))
print("SLQA 合格实验室",len(labs),"测试题",len(te), "例题", te[:6], te[-2:])
rg=random.Random("slqa-ladder"); M=[]
for g in range(12):
    pick=rg.sample(labs,9); nodes,a,b=pick[:7],pick[7],pick[8]
    rows=[i for i in te if a in E[i] and b in E[i] and sum(l in E[i] for l in nodes)>=5]
    for s in (0,70):
        if len(rows)>=s+70:
            M.append(dict(g=g,s=s,nodes=nodes,a=a,b=b,items=rows[s:s+70]))
print("SLQA 窗",[(m["g"],m["s"]) for m in M])
print("窗内 study 数(校准,评估)",[(len({i.split('|')[0] for i in m["items"][:40]}),len({i.split('|')[0] for i in m["items"][40:]})) for m in M])
import collections
c=collections.Counter(); 
for m in M:
    for l in m["nodes"]+[m["a"],m["b"]]: c[l]+=1
print("实验室出现次数",c.most_common())
print("时间范围",[(m["items"][0].split('|')[0],m["items"][39].split('|')[0],m["items"][69].split('|')[0]) for m in M][:6])
ov=[]
for x in range(len(M)):
    for y in range(x+1,len(M)):
        a=set(M[x]["items"]);b=set(M[y]["items"]); ov.append(len(a&b)/70)
print("窗两两题目重叠: 均值 %.2f, 中位 %.2f, 最大 %.2f"%(np.mean(ov),np.median(ov),max(ov)))
json.dump(M,open(W+"/map_slqa.json","w"))
labs_,G=V.groups(); print("BIPM 组",G)
Mb=[dict(g=gi,w=w,nodes=G[gi],lo=cal[0][0],hi=ev[-1][0],cut=ev[0][0],nc=len(cal),ne=len(ev)) for gi,w,cal,ev in V.windows("test")]
for m in Mb: print(m["g"],m["w"],m["lo"],m["cut"],m["hi"],m["nc"],m["ne"])
json.dump(Mb,open(W+"/map_bipm.json","w"))
