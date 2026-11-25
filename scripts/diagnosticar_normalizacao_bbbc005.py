"""Pixel-only diagnostic, no Cellpose inference. Run with --base (local) or --dataset-zip."""
import argparse, csv, hashlib, io, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import tifffile

def main():
    p=argparse.ArgumentParser(); group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--base',type=Path); group.add_argument('--dataset-zip',type=Path)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); rows=[]
    with a.manifest.open(encoding='utf-8-sig',newline='') as f: manifest={(int(r['n']),int(r['amostra'])):r for r in csv.DictReader(f)}
    z=zipfile.ZipFile(a.dataset_zip) if a.dataset_zip else None
    try:
        for n,s in [(1,1),(1,2),(5,1),(5,2)]:
            r=manifest[n,s]; arrays=[]
            for field,folder in [('imagem','imagens_F1_w2'),('mascara','mascaras_F1_w2')]:
                if z:
                    name=PurePosixPath(r[field]).name
                    matches=[v for v in z.namelist() if v.replace('\\','/')==folder+'/'+name]
                    if len(matches)!=1: raise ValueError(f'Expected one archive match: {name}')
                    data=z.read(matches[0])
                else: data=(a.base/r[field]).read_bytes()
                if hashlib.sha256(data).hexdigest()!=r['sha256_'+field]: raise ValueError(f'Hash mismatch: {field} {n} {s}')
                arrays.append(tifffile.imread(io.BytesIO(data)))
            x,y=arrays; fg=y>0
            if x.ndim!=2 or x.shape!=y.shape or not fg.any() or fg.all(): raise ValueError('Unexpected arrays')
            q=np.percentile(x,[0,.1,1,50,99,99.9,100]); median=float(np.median(x[fg]));
            row=dict(n=n,amostra=s,pixels=x.size,foreground_pixels=int(fg.sum()),foreground_fraction=float(fg.mean()),fg_median=median,bg_median=float(np.median(x[~fg])),**dict(zip(['p0','p0_1','p1','p50','p99','p99_9','p100'],map(float,q))))
            row.update(fg_median_scaled_1_99=(median-q[2])/(q[4]-q[2]),fg_median_scaled_0_1_99_9=(median-q[1])/(q[5]-q[1]),fg_fraction_above_p99=float(np.mean(x[fg]>q[4])),sha256_imagem=r['sha256_imagem'],sha256_mascara=r['sha256_mascara']); rows.append(row)
    finally:
        if z:z.close()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print('DIAGNOSTICO OK:',a.out)
if __name__=='__main__':main()
