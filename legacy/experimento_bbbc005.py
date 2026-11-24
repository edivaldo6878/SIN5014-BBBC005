#!/usr/bin/env python3
"""Experimento SIN5014 / BBBC005: contagem de núcleos por componentes conexos (N8),
watershed com marcadores e (opcional) Cellpose, estratificado por densidade.

Uso:
  python experimento_bbbc005.py --images PASTA_IMAGENS --masks PASTA_MASCARAS --out saida
  python experimento_bbbc005.py ... --cellpose --cp-model nuclei --cp-diameter 30   # opcional

Convenção de nomes da BBBC005: ..._C<contagem>_F<foco>_s<amostra>_w<canal>.TIF
Filtra foco F1 e canal w2 (núcleos). Regras do pipeline fixadas ANTES de ver os resultados.
Nada aqui ajusta parâmetros à contagem esperada.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, glob, hashlib, json, os, platform, re, sys, time, tracemalloc
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from skimage import io, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

PAT = re.compile(r'C(\d+)_F(\d+)_s(\d+)_w(\d)', re.I)

def parse(path):
    m = PAT.search(os.path.basename(path))
    return tuple(int(x) for x in m.groups()) if m else None

def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def gray(a):
    a = np.asarray(a)
    if a.ndim == 3: a = a[..., 0]
    return a

def n8_count(bw):
    return int(ndi.label(bw, structure=np.ones((3, 3)))[1])

def pipeline_ws(img, peak_frac=0.3):
    """mediana 3x3 -> Otsu -> distância -> sementes (regra a priori) -> watershed."""
    f = ndi.median_filter(img.astype(np.float32), size=3)
    bw = f > filters.threshold_otsu(f)
    bw = morphology.remove_small_objects(bw, 20)
    lab, n = ndi.label(bw, structure=np.ones((3, 3)))
    if n == 0: return bw, 0
    areas = np.array([r.area for r in measure.regionprops(lab)])
    rhat = np.sqrt(np.median(areas) / np.pi)
    md = max(1, int(round(rhat / 2)))                    # regra a priori
    dist = ndi.distance_transform_edt(bw)
    pk = peak_local_max(dist, min_distance=md, threshold_abs=peak_frac * dist.max(),
                        labels=lab, exclude_border=False)
    mk = np.zeros(bw.shape, int)
    for i, (y, x) in enumerate(pk, 1): mk[y, x] = i
    ws = segmentation.watershed(-dist, mk, mask=bw)
    return bw, int(ws.max())

def dice_iou(pred, ref):
    tp = np.logical_and(pred, ref).sum(); fp = np.logical_and(pred, ~ref).sum(); fn = np.logical_and(~pred, ref).sum()
    d = 2 * tp / max(1, 2 * tp + fp + fn); i = tp / max(1, tp + fp + fn)
    return d, i

def timed(fn, *a):
    tracemalloc.start(); t0 = time.perf_counter()
    out = fn(*a)
    dt = time.perf_counter() - t0; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return out, dt, peak / 2**20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True); ap.add_argument('--masks', required=True)
    ap.add_argument('--out', default='saida'); ap.add_argument('--cellpose', action='store_true')
    ap.add_argument('--cp-model', default='nuclei'); ap.add_argument('--cp-diameter', type=float, default=None)
    ap.add_argument('--gpu', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    imgs = {parse(p)[:3]: p for p in glob.glob(os.path.join(a.images, '**', '*'), recursive=True)
            if parse(p) and parse(p)[1] == 1 and parse(p)[3] == 2 and p.lower().endswith(('.tif', '.tiff'))}
    msks = {parse(p)[:3]: p for p in glob.glob(os.path.join(a.masks, '**', '*'), recursive=True)
            if parse(p) and parse(p)[1] == 1 and parse(p)[3] == 2 and p.lower().endswith(('.tif', '.tiff'))}
    keys = sorted(set(imgs) & set(msks))
    print(f'{len(keys)} pares imagem/máscara (F1, w2)')
    if not keys: sys.exit('Nenhum par encontrado: confira os nomes e as pastas.')
    levels = sorted({k[0] for k in keys}); third = max(1, len(levels) // 3)
    grp = {c: ('Baixa' if i < third else 'Média' if i < 2 * third else 'Alta') for i, c in enumerate(levels)}

    cp = None; cp_info = {}
    if a.cellpose:
        import importlib.metadata as md
        from cellpose import models
        cp_info = {'versao': md.version('cellpose'), 'modelo': a.cp_model, 'diametro': a.cp_diameter, 'gpu': a.gpu}
        try: cp = models.CellposeModel(gpu=a.gpu, model_type=a.cp_model)
        except Exception: cp = models.CellposeModel(gpu=a.gpu)
        print('Cellpose:', cp_info)

    rows = []
    for k in keys:
        n = k[0]
        img = gray(io.imread(imgs[k])); ref = gray(io.imread(msks[k])) > 0
        r = {'contagem_esperada': n, 'amostra': k[2], 'grupo': grp[n]}
        r['n8_ref'] = n8_count(ref); r['n8_ref_erro'] = r['n8_ref'] - n
        (bw, nws), dt, mem = timed(pipeline_ws, img)
        d, i = dice_iou(bw, ref)
        r.update(ws_n=nws, ws_erro=nws - n, ws_tempo_s=dt, ws_mem_MB=mem, ws_dice=d, ws_iou=i)
        if cp is not None:
            (mk, *_), dt, mem = timed(lambda x: cp.eval(x, diameter=a.cp_diameter), img)
            ncp = int(mk.max()); d, i = dice_iou(mk > 0, ref)
            r.update(cp_n=ncp, cp_erro=ncp - n, cp_tempo_s=dt, cp_mem_MB=mem, cp_dice=d, cp_iou=i)
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, 'por_imagem.csv'), index=False)

    def summ(g):
        o = {'n_imagens': len(g)}
        for tag, col in [('N8_referencia', 'n8_ref_erro'), ('Watershed', 'ws_erro')] + ([('Cellpose', 'cp_erro')] if cp else []):
            e = g[col].values; o[f'{tag}_MAE'] = np.abs(e).mean(); o[f'{tag}_vies'] = e.mean(); o[f'{tag}_RMSE'] = np.sqrt((e**2).mean())
        o['Watershed_Dice'] = g.ws_dice.mean(); o['Watershed_IoU'] = g.ws_iou.mean()
        o['Watershed_tempo_s'] = g.ws_tempo_s.mean(); o['Watershed_memMB'] = g.ws_mem_MB.mean()
        if cp: o.update(Cellpose_Dice=g.cp_dice.mean(), Cellpose_IoU=g.cp_iou.mean(), Cellpose_tempo_s=g.cp_tempo_s.mean(), Cellpose_memMB=g.cp_mem_MB.mean())
        return pd.Series(o)
    s = df.groupby('grupo').apply(summ).reindex(['Baixa', 'Média', 'Alta']); s.loc['Todas'] = summ(df)
    s.round(3).to_csv(os.path.join(a.out, 'resumo_por_grupo.csv'))
    print(s.round(3).T.to_string())

    rng = np.random.default_rng(0); ae = np.abs(df.ws_erro.values)
    ci = np.percentile([rng.choice(ae, len(ae)).mean() for _ in range(1000)], [2.5, 97.5])
    w = wilcoxon(np.abs(df.n8_ref_erro), np.abs(df.ws_erro)) if len(df) > 5 else None
    env = {'python': sys.version, 'plataforma': platform.platform(), 'processador': platform.processor(),
           'regra_parametros': 'min_distance=round(r/2), r=sqrt(mediana(area N8)/pi); pico=0.3*max(dist)',
           'cellpose': cp_info, 'bootstrap_IC95_MAE_watershed': ci.tolist(),
           'wilcoxon_N8_vs_watershed_p': None if w is None else float(w.pvalue),
           'bibliotecas': {m: __import__(m).__version__ for m in ('numpy', 'scipy', 'skimage', 'pandas')}}
    dup = {}
    for p in list(imgs.values()) + list(msks.values()): dup.setdefault(sha1(p), []).append(os.path.basename(p))
    env['arquivos_duplicados_por_hash'] = sum(1 for v in dup.values() if len(v) > 1)
    json.dump(env, open(os.path.join(a.out, 'ambiente_e_parametros.json'), 'w'), indent=2, ensure_ascii=False)
    print('Arquivos em', a.out)

if __name__ == '__main__':
    main()
