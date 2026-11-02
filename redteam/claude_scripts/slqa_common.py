from rtc import *
W_=windows_cached("SLQA")
com=[];idi=[];A_s=[]
rows=[]
for cal,ev in W_:
    Rc,Ac,Bc=unpack(cal);Re,Ae,Be=unpack(ev)
    sh=np.nanmedian(Re,0)-np.nanmedian(Rc,0)
    c=np.median(sh); com.append(c); idi.append(np.median(np.abs(sh-c))); sa=np.median(Ae)-np.median(Ac); A_s.append(sa)
    Sc,Se=series_np(Rc,Ac,Bc),series_np(Re,Ae,Be)
    for j in range(Rc.shape[1]):
        c_,e_=Sc[("nodeA",j)],Se[("nodeA",j)]
        rows.append((abs(fstat(c_,e_)[1]),abs(sh[j]),abs(sa),abs(sh[j]-sa),abs(sh[j]-c)))
com=np.array(com);idi=np.array(idi);A_s=np.array(A_s);rows=np.array(rows)
print("SLQA 窗内节点真实偏移：共同分量 |中位| 中位",round(float(np.median(np.abs(com))),4),"；个体分量 中位",round(float(np.median(idi)),4),"；共同>个体的窗",np.mean(np.abs(com)>idi))
print("秤 A 真实偏移 |中位|",round(float(np.median(np.abs(A_s))),4))
al=rows[:,0]>3
def auc(p,n): return sum((a>b)+.5*(a==b) for a in p for b in n)/(len(p)*len(n))
print("报警数",al.sum(),"/",len(al))
for nm,i in(("|节点真实偏移|",1),("|秤A真实偏移|",2),("|节点−秤 真实偏移差|",3),("|节点个体分量|",4)):
    print(f"  AUC({nm}→报警) {auc(rows[al,i],rows[~al,i]):.3f}")
