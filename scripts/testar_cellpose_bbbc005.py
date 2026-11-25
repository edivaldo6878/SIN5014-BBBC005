#!/usr/bin/env python3
"""Pilot Cellpose v3 on BBBC005: calibration samples only."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import tifffile
from scipy import ndimage as ndi

EXPECTED_HASH = "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303"
LEVELS = (1,5,10,14,18,23,27,31,35,40,44,48,53,57,61,66,70,74,78,83,87,91,96,100)

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def read_verified(base, row, field):
    path = base / row[field]
    if sha(path) != row["sha256_" + field]:
        raise ValueError(f"SHA256 mismatch: {path}")
    a = tifffile.imread(path)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"Expected a finite 2D TIFF: {path}, {a.shape}")
    return a

def count_labels(a):
    return int(np.count_nonzero(np.unique(a) > 0))

def foreground_scores(pred, ref):
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    return 2*tp/max(1,2*tp+fp+fn), tp/max(1,tp+fp+fn)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from cellpose import models
    if version("cellpose") != "3.1.1.2" or torch.__version__ != "2.7.1+cpu":
        raise RuntimeError("Unexpected Cellpose/PyTorch version")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    with a.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(int(r["n"]), int(r["amostra"])):r for r in rows}
    if len(rows) != 600 or set(lookup) != {(n,s) for n in LEVELS for s in range(1,26)}:
        raise ValueError("Unexpected manifest keys")
    if any((int(r["foco"]),int(r["canal"])) != (1,2) for r in rows):
        raise ValueError("Unexpected focus/channel")
    a.out.mkdir(parents=True, exist_ok=False)
    diameters = []
    for s in range(1,6):
        ref = read_verified(a.base, lookup[(1,s)], "mascara") > 0
        if ndi.label(ref, structure=np.ones((3,3)))[1] != 1:
            raise ValueError(f"Expected one connected nucleus in C1_s{s:02}")
        diameters.append(float(2*np.sqrt(np.count_nonzero(ref)/np.pi)))
    diameter = float(np.median(diameters))
    print("Calibration C1_s01-s05 diameters:", diameters, flush=True)
    print(f"Fixed calibrated diameter: {diameter:.6f} px", flush=True)
    model = models.CellposeModel(gpu=False, model_type="nuclei")
    weights = Path(model.pretrained_model)
    if weights.name != "nucleitorch_0" or sha(weights) != EXPECTED_HASH or model.device.type != "cpu":
        raise RuntimeError("Unexpected model weights/device")
    if not np.isclose(float(model.diam_mean),17):
        raise RuntimeError(f"Unexpected model reference diameter: {model.diam_mean}")
    params = dict(channels=[0,0], normalize=True, do_3D=False,
                  flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
                  batch_size=8, resample=True, augment=False)
    configs = {"C0_fixed17":17.0, "C1_calibrated":diameter}
    metadata = dict(python=sys.version, platform=platform.platform(),
        versions={name:version(name) for name in ("cellpose","torch","numpy","scipy","tifffile","opencv-python-headless")},
        model="nuclei", weights_file=weights.name, weights_sha256=EXPECTED_HASH,
        weights_bytes=weights.stat().st_size, device=str(model.device),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        seed=0, manifest_sha256=sha(a.manifest), params=params, diameters=configs,
        calibration_diameters=diameters, calibration_keys=[f"C1_s{s:02}" for s in range(1,6)],
        pilot_keys=["C10_s01","C48_s01","C91_s01"],
        timing="model.eval wall time; no warmup; pilot only",
        note="C0 fixed model diameter, C1 uses reference masks for scale calibration only; no retraining")
    (a.out/"protocolo.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    fields = ["config","n","amostra","diametro_px","cp_n","cp_erro","cp_dice","cp_iou","cp_tempo_s"]
    with (a.out/"piloto.csv").open("x",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for n in (10,48,91):
            r = lookup[(n,1)]
            img = read_verified(a.base,r,"imagem")
            ref = read_verified(a.base,r,"mascara") > 0
            if img.shape != ref.shape:
                raise ValueError("Image/reference shape mismatch")
            for tag,d in configs.items():
                print(f"Processing C{n}_s01 {tag} diameter={d:.3f}",flush=True)
                t = time.perf_counter()
                mask, *_ = model.eval(img, diameter=d, **params)
                dt = time.perf_counter()-t
                mask = np.asarray(mask)
                if mask.shape != img.shape or not np.issubdtype(mask.dtype,np.integer) or np.any(mask<0):
                    raise ValueError("Invalid instance labels")
                k = count_labels(mask)
                dice,iou = foreground_scores(mask>0,ref)
                result = dict(config=tag,n=n,amostra=1,diametro_px=d,
                              cp_n=k,cp_erro=k-n,cp_dice=dice,cp_iou=iou,cp_tempo_s=dt)
                w.writerow(result)
                f.flush()
                print(f"GT={n} CP={k} error={k-n:+d} Dice={dice:.4f} IoU={iou:.4f} time={dt:.3f}s",flush=True)
                tifffile.imwrite(a.out/f"C{n}_s01_{tag}_labels.tif",mask.astype(np.uint32))
    print("CELLPOSE PILOT OK", flush=True)

if __name__ == "__main__":
    main()
