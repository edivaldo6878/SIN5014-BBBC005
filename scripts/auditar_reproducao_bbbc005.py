#!/usr/bin/env python3
"""Audit the reproduced BBBC005 F1/w2 experiment without modifying input files."""
import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon, spearmanr

LEVELS = [1,5,10,14,18,23,27,31,35,40,44,48,53,57,61,66,70,74,78,83,87,91,96,100]
HISTORICAL_SHA = "0616288f3ef62a0b1fbb814b97d483daf02998eb39409b62d5c5aa8afe7c704a"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def group(n):
    return "Baixa" if n <= 31 else "Media" if n <= 66 else "Alta"

def stats(g):
    a, b = g.n8_ref_erro, g.ws_erro
    n = g.contagem_esperada
    return dict(n_imagens=len(g), N8_referencia_MAE=abs(a).mean(),
        N8_referencia_vies=a.mean(), N8_referencia_RMSE=np.sqrt((a*a).mean()),
        Watershed_MAE=abs(b).mean(), Watershed_vies=b.mean(),
        Watershed_RMSE=np.sqrt((b*b).mean()), N8_MAPE_pct=(abs(a)/n).mean()*100,
        Watershed_MAPE_pct=(abs(b)/n).mean()*100,
        Watershed_Dice=g.ws_dice.mean(), Watershed_IoU=g.ws_iou.mean(),
        Watershed_tempo_s=g.ws_tempo_s.mean(), Watershed_memMB=g.ws_mem_MB.mean(),
        subcontagem=int((b<0).sum()), exata=int((b==0).sum()),
        supercontagem=int((b>0).sum()), melhor=int((abs(b)<abs(a)).sum()),
        empate=int((abs(b)==abs(a)).sum()), pior=int((abs(b)>abs(a)).sum()))

def bootstrap(x, B):
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x,len(x),replace=True).mean()
                          for _ in range(B)],[2.5,97.5]).tolist()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--legacy",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    require(not args.out.exists(),"Output directory already exists")
    paths = dict(manifest=args.manifest, legacy=args.legacy,
                 individual=args.results/"por_imagem.csv",
                 summary=args.results/"resumo_por_grupo.csv",
                 environment=args.results/"ambiente_e_parametros.json")
    hashes = {k:sha(v) for k,v in paths.items()}
    require(hashes["legacy"]==HISTORICAL_SHA,"Historical code SHA-256 differs")
    d = pd.read_csv(paths["individual"])
    m = pd.read_csv(paths["manifest"])
    s = pd.read_csv(paths["summary"])
    env = json.loads(paths["environment"].read_text(encoding="utf-8-sig"))
    keys = {(n,i) for n in LEVELS for i in range(1,26)}
    require(len(d)==len(m)==600,"Expected 600 rows in each table")
    require(set(zip(d.contagem_esperada,d.amostra))==set(zip(m.n,m.amostra))==keys,
            "Missing, extra or duplicate keys")
    require(not d.isna().any().any() and not m.isna().any().any(),"Missing values")
    require((m.foco==1).all() and (m.canal==2).all(),"Focus/channel mismatch")
    require((m.densidade==m.n.map(group)).all(),"Manifest group mismatch")
    require((d.grupo.str.replace("é","e")==d.contagem_esperada.map(group)).all(),
            "Result group mismatch")
    all_hashes = list(m.sha256_imagem)+list(m.sha256_mascara)
    require(all(re.fullmatch(r"[a-f0-9]{64}",h) for h in all_hashes),"Invalid SHA-256")
    require(len(set(all_hashes))==1200,"Repeated hashes in manifest")
    for r in m.itertuples():
        a,b = Path(r.imagem),Path(r.mascara)
        match = re.search(r"C(\d+)_F(\d+)_s(\d+)_w(\d+)",a.name,re.I)
        require(a.name==b.name and match is not None and tuple(map(int,match.groups()))==
                (r.n,1,r.amostra,2),"Filename mismatch")
    for col in ["n8_ref","ws_n"]:
        require((d[col]>=0).all() and (d[col]%1==0).all(),"Invalid count")
    require((d.n8_ref-d.contagem_esperada==d.n8_ref_erro).all(),"N8 error mismatch")
    require((d.ws_n-d.contagem_esperada==d.ws_erro).all(),"Watershed error mismatch")
    require(d.ws_iou.between(0,1).all() and d.ws_dice.between(0,1).all() and
            np.allclose(d.ws_dice,2*d.ws_iou/(1+d.ws_iou)),"Dice/IoU mismatch")
    require(np.isfinite(d.select_dtypes(include="number")).all().all(),"Nonfinite values")
    require((d.ws_tempo_s>=0).all() and (d.ws_mem_MB>=0).all(),"Invalid resource metric")
    groups = [(k,d[d.grupo==k]) for k in ["Baixa","Média","Alta"]]+[("Todas",d)]
    grouped = pd.DataFrame([dict(grupo=k,**stats(g)) for k,g in groups])
    require(len(s)==4 and list(s.grupo)==list(grouped.grupo),"Summary group mismatch")
    for c in s.columns.drop("grupo"):
        require(np.allclose(s[c],grouped[c],rtol=0,atol=0.00050001),
                f"Summary mismatch: {c}")
    a,b = abs(d.n8_ref_erro.to_numpy()),abs(d.ws_erro.to_numpy())
    ci1000,ci2000 = bootstrap(b,1000),bootstrap(b,2000)
    w = wilcoxon(a,b)
    require(np.allclose(ci1000,env["bootstrap_IC95_MAE_watershed"]),
            "Historical bootstrap mismatch")
    require(np.isclose(w.pvalue,env["wilcoxon_N8_vs_watershed_p"],rtol=1e-6,atol=0),
            "Historical Wilcoxon mismatch")
    rho = spearmanr(d.contagem_esperada,b).statistic
    bylevel = pd.DataFrame([dict(contagem_esperada=k,**stats(g))
                            for k,g in d.groupby("contagem_esperada")])
    split = pd.DataFrame([dict(particao=k,**stats(g)) for k,g in
                         [("calibracao_s01_s05",d[d.amostra<=5]),
                          ("validacao_s06_s25",d[d.amostra>5])]])
    linked = d.merge(m,left_on=["contagem_esperada","amostra"],right_on=["n","amostra"],
                     validate="one_to_one")
    linked["erro_absoluto_ws"] = abs(linked.ws_erro)
    extremes = linked.sort_values(["erro_absoluto_ws","contagem_esperada","amostra"],
                                   ascending=[False,True,True]).head(20)
    report = dict(status="AUDITORIA OK", pares=600, hashes_manifesto_distintos=1200,
        sha256_entradas=hashes, bootstrap_seed=0, bootstrap_1000=ci1000,
        bootstrap_2000=ci2000, wilcoxon_estatistica=float(w.statistic),
        wilcoxon_p=float(w.pvalue), spearman_contagem_erro_abs=float(rho),
        reducao_mae_pct=float(100*(1-b.mean()/a.mean())),
        ambiente_auditoria=dict(python=platform.python_version(),numpy=np.__version__,
                                 pandas=pd.__version__,scipy=scipy.__version__),
        limites=["No TIFF bytes audited here; manifest entries only.",
                 "N8 uses privileged reference mask.",
                 "Dice and IoU measure binary foreground before Watershed.",
                 "No historical per-image CSV available for comparison.",
                 "Retrospective internal split; not external validation.",
                 "Bootstrap resamples individual images; no population-level generalization."])
    args.out.mkdir(parents=True,exist_ok=False)
    for name,table in [("estatisticas_grupo",grouped),("estatisticas_nivel",bylevel),
                       ("baseline_particoes",split),("casos_extremos",extremes),
                       ("supercontagens",linked[linked.ws_erro>0])]:
        table.to_csv(args.out/f"{name}.csv",index=False,encoding="utf-8",lineterminator="\n")
    (args.out/"auditoria.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                                          encoding="utf-8")
    print(grouped[["grupo","n_imagens","N8_referencia_MAE","Watershed_MAE",
                   "subcontagem","exata","supercontagem"]].to_string(index=False))
    print("Bootstrap 2000:",ci2000)
    print("Wilcoxon p:",w.pvalue)
    print("AUDITORIA OK")

if __name__=="__main__":
    main()
