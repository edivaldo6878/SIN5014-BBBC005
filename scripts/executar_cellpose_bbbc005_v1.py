#!/usr/bin/env python3
"""SIN5014: execute fixed Cellpose C0/C1 protocol, with atomic CSV checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import tifffile

LEVELS = (1,5,10,14,18,23,27,31,35,40,44,48,53,57,61,66,70,74,78,83,87,91,96,100)
CONFIGS = {"C0_fixed17":17.0, "C1_calibrated":60.52363315435602}
PARAMS = dict(channels=[0,0], normalize=True, do_3D=False,
              flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
              batch_size=8, resample=True, augment=False)
WEIGHTS = "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303"
MANIFEST = "7f8a8f2605fbcf161db426743d7dccc6f125a51b8e33f3c200c03b5bdeec2932"
FIELDS = ["config","contagem_esperada","amostra","grupo","particao","diametro_px",
          "cp_n","cp_erro","cp_dice","cp_iou","cp_tempo_s",
          "rss_antes_MiB","rss_depois_MiB","peak_wset_processo_MiB"]

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def atomic_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)

def safe_path(base, rel):
    p = (base / rel).resolve()
    require(p.is_relative_to(base.resolve()), "Path outside dataset")
    return p

def read_verified(base,r,field):
    p = safe_path(base,r[field])
    require(sha(p)==r["sha256_"+field],f"SHA256 mismatch: {p}")
    a = tifffile.imread(p)
    require(a.ndim==2 and np.isfinite(a).all(),f"Invalid 2D TIFF: {p}")
    return a

def group(n):
    return "Baixa" if n<=31 else "Media" if n<=66 else "Alta"

def partition(s):
    return "calibracao" if s<=5 else "validacao"

def scores(mask,ref):
    require(mask.shape==ref.shape and np.issubdtype(mask.dtype,np.integer)
            and np.all(mask>=0),"Invalid instance labels")
    k = int(np.count_nonzero(np.unique(mask)>0))
    pred = mask>0
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    return k, 2*tp/max(1,2*tp+fp+fn), tp/max(1,tp+fp+fn)

def load_manifest(path):
    require(sha(path)==MANIFEST,"Manifest byte hash differs from pilot; verify before proceeding")
    rows=read_csv(path)
    keys={(int(r["n"]),int(r["amostra"])):r for r in rows}
    require(len(rows)==len(keys)==600 and set(keys)=={(n,s) for n in LEVELS for s in range(1,26)},
            "Invalid manifest keys")
    for (n,s),r in keys.items():
        require((int(r["foco"]),int(r["canal"]))==(1,2) and r["densidade"]==group(n),
                "Manifest focus/channel/group mismatch")
    return keys

def validate_checkpoint(rows, expected):
    done=set()
    for r in rows:
        tag,n,s=r["config"],int(r["contagem_esperada"]),int(r["amostra"])
        key=(tag,n,s)
        require(key in expected and key not in done,"Unexpected/duplicate checkpoint key")
        require(r["grupo"]==group(n) and r["particao"]==partition(s)
                and float(r["diametro_px"])==CONFIGS[tag],"Checkpoint protocol mismatch")
        k=int(r["cp_n"])
        dice,iou=float(r["cp_dice"]),float(r["cp_iou"])
        require(k>=0 and int(r["cp_erro"])==k-n and 0<=iou<=1 and 0<=dice<=1
                and math.isclose(dice,2*iou/(1+iou),abs_tol=1e-10),"Checkpoint metrics invalid")
        for col in ("cp_tempo_s","rss_antes_MiB","rss_depois_MiB"):
            v=float(r[col])
            require(math.isfinite(v) and v>=0,"Checkpoint resource metric invalid")
        if r["peak_wset_processo_MiB"]!="":
            v=float(r["peak_wset_processo_MiB"])
            require(math.isfinite(v) and v>=0,"Invalid process peak")
        done.add(key)
    return done

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--pilot",action="store_true",help="Only three calibration images, both configurations")
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    lookup=load_manifest(a.manifest)
    keys=[(10,1),(48,1),(91,1)] if a.pilot else sorted(lookup)
    expected={(tag,n,s) for n,s in keys for tag in CONFIGS}
    import torch
    from cellpose import models
    require(version("cellpose")=="3.1.1.2" and torch.__version__=="2.7.1+cpu",
            "Unexpected Cellpose/PyTorch version")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    model=models.CellposeModel(gpu=False,model_type="nuclei")
    w=Path(model.pretrained_model)
    require(w.name=="nucleitorch_0" and sha(w)==WEIGHTS and model.device.type=="cpu",
            "Unexpected model weights/device")
    require(np.isclose(float(model.diam_mean),17),"Unexpected model diameter")
    protocol=dict(model="nuclei",weights_sha256=WEIGHTS,weights_bytes=w.stat().st_size,
        manifest_sha256=MANIFEST,executor_sha256=sha(Path(__file__)),
        python=sys.version,platform=platform.platform(),
        versions={x:version(x) for x in ("cellpose","torch","numpy","scipy","tifffile",
            "opencv-python-headless","psutil","numba","llvmlite")},
        device="cpu",threads=4,interop_threads=1,seed=0,params=PARAMS,diameters=CONFIGS,
        calibration_keys=[f"C1_s{s:02}" for s in range(1,6)],
        mode="pilot" if a.pilot else "all600",expected_rows=len(expected),
        timing="model.eval wall seconds, excludes input, metrics and warmup; no tracemalloc",
        warmup="C10_s01 one inference per configuration per invocation, excluded",
        memory="RSS before/after; Windows peak_wset is cumulative process lifetime peak, not per-image",
        note="C1 calibration uses five reference masks; no retraining; retrospective internal validation")
    csv_path=a.out/"por_imagem_cellpose.csv"
    proto_path=a.out/"protocolo.json"
    if a.resume:
        require(proto_path.is_file() and csv_path.is_file(),"No complete checkpoint metadata")
        require(json.loads(proto_path.read_text(encoding="utf-8"))==protocol,
                "Protocol changed: use original environment/script or new output directory")
        rows=read_csv(csv_path)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        proto_path.write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
        rows=[]
        atomic_csv(csv_path,rows)
    done=validate_checkpoint(rows,expected)
    print(f"Completed {len(done)}/{len(expected)} inferences",flush=True)
    if done!=expected:
        warm=read_verified(a.base,lookup[(10,1)],"imagem")
        for tag,d in CONFIGS.items():
            print(f"Warmup {tag} (excluded)",flush=True)
            model.eval(warm,diameter=d,**PARAMS)
        proc=psutil.Process()
        mib=2**20
        for n,s in keys:
            if all((tag,n,s) in done for tag in CONFIGS):
                continue
            r=lookup[(n,s)]
            img=read_verified(a.base,r,"imagem")
            ref=read_verified(a.base,r,"mascara")>0
            require(img.shape==ref.shape,"Image/reference shape mismatch")
            for tag,d in CONFIGS.items():
                if (tag,n,s) in done:
                    continue
                before=proc.memory_info().rss/mib
                t=time.perf_counter()
                mask,*_=model.eval(img,diameter=d,**PARAMS)
                dt=time.perf_counter()-t
                mem=proc.memory_info()
                k,dice,iou=scores(np.asarray(mask),ref)
                peak=getattr(mem,"peak_wset",None)
                rows.append(dict(config=tag,contagem_esperada=n,amostra=s,grupo=group(n),
                    particao=partition(s),diametro_px=d,cp_n=k,cp_erro=k-n,cp_dice=dice,cp_iou=iou,
                    cp_tempo_s=dt,rss_antes_MiB=before,rss_depois_MiB=mem.rss/mib,
                    peak_wset_processo_MiB="" if peak is None else peak/mib))
                atomic_csv(csv_path,rows)
                done.add((tag,n,s))
                print(f"{len(done)}/{len(expected)} C{n}_s{s:02} {tag} count={k} error={k-n:+d} time={dt:.3f}s",flush=True)
    require(validate_checkpoint(read_csv(csv_path),expected)==expected,"Incomplete execution")
    print("CELLPOSE COMPLETE",flush=True)
    for tag in CONFIGS:
        for part in ("calibracao","validacao","todas"):
            rr=[r for r in rows if r["config"]==tag and (part=="todas" or r["particao"]==part)]
            if rr:
                print(tag,part,"n=",len(rr),"MAE=",sum(abs(int(r["cp_erro"])) for r in rr)/len(rr),
                      "mean_s=",sum(float(r["cp_tempo_s"]) for r in rr)/len(rr),flush=True)

if __name__=="__main__":
    main()
