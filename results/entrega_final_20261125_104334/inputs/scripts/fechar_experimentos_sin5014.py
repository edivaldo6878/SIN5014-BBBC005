#!/usr/bin/env python3
# SIN5014 - ETAPA 06.04 / 08 / 09 - PASSO 211
# Run once; no new parameter selection or neural inference.
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import executar_watershed_bstar_v1 as b

def check(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verified(p,h):
    check(sha(p)==h,f"SHA256 mismatch: {p}")
    return p
def read(p,h=None):
    if h: verified(p,h)
    return pd.read_csv(p)
def save(d,p):
    tmp=p.with_suffix(".tmp")
    d.to_csv(tmp,index=False)
    tmp.replace(p)
def keyed(d,col="contagem_esperada"):
    return set(zip(d[col].astype(int),d.amostra.astype(int)))
def bootstrap(d, value):
    rng=np.random.default_rng(0)
    res=np.zeros(2000)
    for _,v in d.groupby("n",sort=True):
        a=v[value].to_numpy()
        res+=a[rng.integers(0,len(a),(2000,len(a)))].sum(axis=1)
    return np.quantile(res/len(d),[.025,.975])

def main():
    ap=argparse.ArgumentParser()
    for k in ("repo","base","cal","out"): ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args()
    r,o,c=a.repo,a.out,a.cal
    manifest=r/"data/manifestos/manifest_f1_w2.csv"
    base=r/"results/reproducao_v04_20261124_201858/por_imagem.csv"
    cp=r/"results/cellpose_completo_20261124_214400/por_imagem_cellpose.csv"
    cs0=r/"results/cellpose_cstar_calibracao_20261125_081650/por_imagem_cellpose.csv"
    cs1=r/"results/cellpose_cstar_validacao_20261125_083155/por_imagem_cellpose.csv"
    sel=json.loads((c/"selecao_bstar.json").read_text(encoding="utf-8"))
    check(sel["selected"]=="B1" and (sel["alpha"],sel["beta"])==(.35,.2),"Selection not B1")
    check(sel["baseline_controls"]==120,"Missing calibration controls")
    check(sha(Path(b.__file__))==sel["executor_sha256"],"Executor changed")
    check(sha(base)==sel["baseline_sha256"],"Baseline changed")
    verified(c/"protocolo.json",sel["protocol_sha256"])
    proto=json.loads((c/"protocolo.json").read_text(encoding="utf-8"))
    for name,actual in [("numpy",np.__version__),("scipy",b.scipy.__version__),("skimage",b.skimage.__version__)]:
        check(proto[name]==actual,f"Version mismatch {name}")
    verified(r/"docs/protocolo_watershed_bstar.md",proto["methodological_protocol_sha256"])
    cal=read(c/"por_imagem_bstar.csv",sel["calibration_csv_sha256"])
    lookup=b.load_manifest(manifest)
    old=b.load_baseline(base,lookup)
    ck={(n,s) for n,s in lookup if s<=5}
    vk={(n,s) for n,s in lookup if s>5}
    check(len(cal)==1080 and set(cal.config)==set(b.GRID),"Calibration incomplete")
    for tag,g in cal.groupby("config"):
        check(len(g)==120 and keyed(g)==ck,f"Calibration keys {tag}")
        check((g.ws_n-g.contagem_esperada==g.ws_erro).all(),"Calibration error")
        check((g.alpha==b.GRID[tag][0]).all() and (g.beta==b.GRID[tag][1]).all(),"Grid mismatch")
    for row in cal[cal.config=="B5"].to_dict("records"): b.check_baseline(row,old)
    rank=cal.assign(ae=cal.ws_erro.abs()).groupby("config").ae.sum()
    winner=min(rank.index,key=lambda t:(rank[t],0 if t=="B5" else int(t[1:])))
    check(winner=="B1" and int(rank[winner])==1634,"Selection mismatch")
    neural=read(cp,"d945f06a4e9b84bde6e83813801fcec8723a85f882f2e0c8d793644b42e6aa62")
    star=pd.concat([read(cs0,"98519c486553207ea274462a44c05cd686acd84429d97145124f76a004ece8b7"),read(cs1,"82bd60b874b45da60561d05ee1f1acb48a19322e62d00f2215f8ec5a205cd19d")],ignore_index=True)
    for tag,g in [*neural.groupby("config"),("Cstar_norm",star)]:
        check(len(g)==600 and keyed(g)==set(lookup),f"Neural keys {tag}")
        check((g.cp_n-g.contagem_esperada==g.cp_erro).all(),f"Neural error {tag}")
    check(set(neural.config)=={"C0_fixed17","C1_calibrated"} and set(star.config)=={"Cstar_norm"},"Neural configs")
    b.morphological_test()
    o.mkdir(parents=True,exist_ok=False)
    (o/"protocolo.json").write_text(json.dumps(dict(partition="validacao",selected=sel,selection_sha256=sha(c/"selecao_bstar.json"),script_sha256=sha(__file__),manifest_sha256=sha(manifest),seed=0,bootstrap=2000,source_hashes={str(x.name)+":"+x.parent.name:sha(x) for x in [base,cp,cs0,cs1]},note="Retrospective internal validation, no new tuning"),indent=2),encoding="utf-8")
    rows=[]
    for i,(n,s) in enumerate(sorted(vk),1):
        im=b.read_verified(a.base,lookup[n,s],"imagem")
        ref=b.read_verified(a.base,lookup[n,s],"mascara")>0
        check(im.shape==ref.shape and ref.any(),"Invalid reference")
        prep=b.prepare(im)
        for tag in ("B5","B1"):
            v=b.run_ws(prep,*b.GRID[tag],ref)
            v.update(config=tag,contagem_esperada=n,amostra=s,grupo=b.group(n),particao="validacao",ws_erro=v["ws_n"]-n)
            if tag=="B5": b.check_baseline(v,old)
            rows.append(v)
        if i%20==0:
            save(pd.DataFrame(rows),o/"checkpoint.csv")
            print(f"{i}/480 images; {2*i}/960 segmentations",flush=True)
        if (n,s) in {(10,6),(48,6),(91,6)}:
            fig,ax=plt.subplots(1,5,figsize=(15,3.6))
            for j,img,title in [(0,im,"Original"),(1,ref,"Reference"),(2,prep["dist"],"Distance")]:
                ax[j].imshow(img,cmap="gray" if j<2 else "viridis"); ax[j].set_title(title)
            for j,tag in [(3,"B5"),(4,"B1")]:
                alpha,beta=b.GRID[tag]
                pk=peak_local_max(prep["dist"],min_distance=max(1,round(alpha*prep["rhat"])),threshold_abs=beta*prep["dmax"],labels=prep["lab"],exclude_border=False)
                mk=np.zeros(im.shape,int)
                if len(pk): mk[pk[:,0],pk[:,1]]=np.arange(1,len(pk)+1)
                ws=watershed(-prep["dist"],mk,mask=prep["bw"])
                count=int(np.count_nonzero(np.unique(ws)>0))
                check(count==next(v["ws_n"] for v in rows[-2:] if v["config"]==tag),"Visual count mismatch")
                ax[j].imshow(ws,cmap="nipy_spectral",interpolation="nearest")
                if len(pk): ax[j].scatter(pk[:,1],pk[:,0],s=3,c="white")
                ax[j].set_title(f"{tag}: {count}/{n}")
            for x in ax: x.axis("off")
            fig.suptitle(f"BBBC005 C{n}_s{s:02}");fig.tight_layout()
            fig.savefig(o/f"exemplo_C{n}_s{s:02}.png",dpi=220);plt.close(fig)
    val=pd.DataFrame(rows)
    check(len(val)==960 and not val.duplicated(["config","contagem_esperada","amostra"]).any(),"Validation incomplete")
    for _,g in val.groupby("config"): check(keyed(g)==vk,"Validation keys")
    save(val,o/"por_imagem_bstar.csv")
    save(b.summarize(val),o/"resumo_bstar.csv")
    full=pd.concat([cal[cal.config.isin(["B5","B1"])],val],ignore_index=True)
    records=[]
    for (n,s),v in sorted(old.items()):
        records.append(dict(n=n,amostra=s,grupo=b.group(n),particao="calibracao" if s<=5 else "validacao",metodo="A",contagem=int(v.n8_ref),erro=int(v.n8_ref)-n,dice=np.nan,iou=np.nan))
    for _,v in full.iterrows():
        records.append(dict(n=int(v.contagem_esperada),amostra=int(v.amostra),grupo=v.grupo,particao=v.particao,metodo={"B5":"B","B1":"Bstar"}[v.config],contagem=int(v.ws_n),erro=int(v.ws_erro),dice=v.final_dice,iou=v.final_iou))
    for _,v in pd.concat([neural,star],ignore_index=True).iterrows():
        n,s=int(v.contagem_esperada),int(v.amostra)
        records.append(dict(n=n,amostra=s,grupo=b.group(n),particao="calibracao" if s<=5 else "validacao",metodo={"C0_fixed17":"C0","C1_calibrated":"C1","Cstar_norm":"Cstar"}[v.config],contagem=int(v.cp_n),erro=int(v.cp_erro),dice=v.cp_dice,iou=v.cp_iou))
    d=pd.DataFrame(records);d["ae"]=d.erro.abs()
    check(len(d)==3600 and not d.duplicated(["metodo","n","amostra"]).any(),"Combined keys")
    save(d,o/"resultados_seis_metodos.csv")
    sm=[]
    for part in ("calibracao","validacao","Todas"):
        for g in ("Baixa","Media","Alta","Todas"):
            t=d if part=="Todas" else d[d.particao==part]
            if g!="Todas": t=t[t.grupo==g]
            for method,x in t.groupby("metodo"):
                sm.append(dict(particao=part,grupo=g,metodo=method,n_imagens=len(x),mae=x.ae.mean(),vies=x.erro.mean(),rmse=np.sqrt((x.erro**2).mean()),mape=100*(x.ae/x.n).mean(),exatas=int((x.erro==0).sum()),subcontagens=int((x.erro<0).sum()),supercontagens=int((x.erro>0).sum()),zeros=int((x.contagem==0).sum()),dice_final=x.dice.mean(),iou_final=x.iou.mean()))
    sm=pd.DataFrame(sm);save(sm,o/"resumo_seis_metodos.csv")
    v=d[d.particao=="validacao"].pivot(index=["n","amostra","grupo"],columns="metodo",values="ae").reset_index()
    ci=[]
    for left,right in [("B","A"),("Bstar","B"),("C0","B"),("C1","B"),("C1","C0"),("Cstar","C1"),("Cstar","B"),("C1","Bstar")]:
        for g in ("Baixa","Media","Alta","Todas"):
            t=(v if g=="Todas" else v[v.grupo==g]).copy()
            t["delta"]=t[left]-t[right]
            lo,hi=bootstrap(t,"delta")
            ci.append(dict(contraste=f"{left}-{right}",grupo=g,n_imagens=len(t),delta_mae=t.delta.mean(),ic95_inf=lo,ic95_sup=hi,melhor=int((t.delta<0).sum()),empate=int((t.delta==0).sum()),pior=int((t.delta>0).sum())))
    save(pd.DataFrame(ci),o/"comparacoes_pareadas.csv")
    methods=["A","B","Bstar","C0","C1","Cstar"]
    vv=sm[(sm.particao=="validacao")&(sm.grupo=="Todas")].set_index("metodo").loc[methods]
    fig,ax=plt.subplots(figsize=(8,4)); bars=ax.bar(methods,vv.mae);ax.bar_label(bars,fmt="%.2f");ax.set_ylabel("MAE (nuclei/image)");ax.set_title("Internal validation (n=480)");fig.tight_layout();fig.savefig(o/"mae_validacao.png",dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for method,g in d[d.particao=="validacao"].groupby("metodo"):
        k=g.groupby("n").contagem.mean();ax.plot(k.index,k.values,"o-",ms=3,label=method)
    ax.plot(b.LEVELS,b.LEVELS,"k--",label="Ideal");ax.set(xlabel="Expected count",ylabel="Mean predicted count");ax.legend(ncol=2);fig.tight_layout();fig.savefig(o/"contagem_por_nivel.png",dpi=240);plt.close(fig)
    grid=cal.groupby(["alpha","beta"]).ws_erro.apply(lambda x:x.abs().mean()).unstack()
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(grid,cmap="viridis_r")
    ax.set_xticks(range(3),grid.columns);ax.set_yticks(range(3),grid.index)
    ax.set(xlabel="beta",ylabel="alpha",title="Calibration MAE")
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{grid.iloc[i,j]:.2f}",ha="center",va="center",color="white" if grid.iloc[i,j]>17 else "black")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(o/"grade_calibracao.png",dpi=240);plt.close(fig)
    (o/"checkpoint.csv").unlink(missing_ok=True)
    files=sorted(x for x in o.iterdir() if x.is_file() and x.name!="SHA256SUMS.txt")
    (o/"SHA256SUMS.txt").write_text("".join(f"{sha(x)}  {x.name}\n" for x in files),encoding="ascii")
    archive=o.parent/(o.name+".zip")
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for folder in (o,c):
            for x in sorted(folder.iterdir()):
                if x.is_file(): z.write(x,f"{folder.name}/{x.name}")
        for x in [Path(__file__),Path(b.__file__),r/"docs/protocolo_watershed_bstar.md",manifest,base,cp,cs0,cs1]:
            z.write(x,f"inputs/{x.parent.name}/{x.name}")
    print(vv[["mae","vies","rmse","dice_final"]].to_string())
    print(pd.DataFrame(ci).query("grupo=='Todas'").to_string(index=False))
    print("FINAL COMPLETE: B5 controls 480/480; Bstar 480/480; 3600 consolidated rows")
    print(f"ZIP: {archive}\nSHA256: {sha(archive)}")

if __name__=="__main__": main()
