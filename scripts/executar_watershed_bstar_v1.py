
#!/usr/bin/env python3
# SIN5014 - ETAPA 06.02 - PASSO 197
# Watershed sensitivity analysis - independent executor

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import skimage
import tifffile

from scipy import ndimage as ndi
from skimage import filters, morphology, measure
from skimage import segmentation
from skimage.feature import peak_local_max


MANIFEST_SHA = (
    "7f8a8f2605fbcf161db426743d7dccc6f"
    "125a51b8e33f3c200c03b5bdeec2932"
)

LEVELS = (
    1, 5, 10, 14, 18, 23, 27, 31,
    35, 40, 44, 48, 53, 57, 61, 66,
    70, 74, 78, 83, 87, 91, 96, 100
)

GRID = {
    f"B{i*3+j+1}": (alpha, beta)
    for i, alpha in enumerate([0.35, 0.50, 0.65])
    for j, beta in enumerate([0.20, 0.30, 0.40])
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def group(n):
    if n <= 31:
        return "Baixa"
    if n <= 66:
        return "Media"
    return "Alta"


def load_manifest(path):
    require(
        sha(path) == MANIFEST_SHA,
        "Manifest SHA256 mismatch"
    )

    raw = Path(path).read_text(
        encoding="utf-8-sig"
    )

    rows = list(csv.DictReader(io.StringIO(raw)))

    lookup = {}

    for r in rows:
        n = int(r["n"])
        s = int(r["amostra"])

        require(
            (int(r["foco"]), int(r["canal"])) == (1, 2),
            "Unexpected focus or channel"
        )

        require(
            (n, s) not in lookup,
            "Duplicate manifest key"
        )

        lookup[(n, s)] = r

    expected = {
        (n, s)
        for n in LEVELS
        for s in range(1, 26)
    }

    require(
        len(rows) == 600
        and set(lookup) == expected,
        "Manifest key mismatch"
    )

    return lookup


def read_verified(base, row, field):
    base = Path(base).resolve()
    path = (base / row[field]).resolve()

    require(
        path.is_relative_to(base),
        "Unsafe image path"
    )

    data = path.read_bytes()

    actual = hashlib.sha256(data).hexdigest()

    require(
        actual == row["sha256_" + field].lower(),
        f"TIFF SHA256 mismatch: {path}"
    )

    arr = tifffile.imread(io.BytesIO(data))

    require(
        arr.ndim == 2
        and np.isfinite(arr).all(),
        f"Invalid TIFF array: {path}"
    )

    return arr


def morphological_test():
    a = np.zeros((7, 25), dtype=bool)

    a[0, :19] = True
    a[3, :20] = True
    a[6, :21] = True

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", FutureWarning
        )
        old = morphology.remove_small_objects(
            a, 20
        )

    new = morphology.remove_small_objects(
        a, max_size=20, connectivity=1
    )

    require(
        np.array_equal(old, new),
        "Morphological API mismatch"
    )

    lab, _ = ndi.label(new)

    areas = np.bincount(lab.ravel())[1:]

    require(
        areas.tolist() == [21],
        "Unexpected morphological semantics"
    )


def prepare(img):
    f = ndi.median_filter(
        img.astype(np.float32),
        size=3
    )

    otsu = filters.threshold_otsu(f)

    bw = f > otsu

    bw = morphology.remove_small_objects(
        bw, max_size=20, connectivity=1
    )

    lab, nc = ndi.label(
        bw, structure=np.ones((3, 3))
    )

    areas = [
        int(r.area)
        for r in measure.regionprops(lab)
    ]

    rhat = (
        float(np.sqrt(np.median(areas)/np.pi))
        if areas else 0.0
    )

    dist = ndi.distance_transform_edt(bw)

    return dict(
        bw=bw,
        lab=lab,
        nc=int(nc),
        areas=areas,
        rhat=rhat,
        dist=dist,
        dmax=float(dist.max()),
        otsu=float(otsu)
    )


def binary_metrics(pred, ref):
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        dice=2*tp/max(1, 2*tp+fp+fn),
        iou=tp/max(1, tp+fp+fn),
        precision=tp/max(1, tp+fp),
        recall=tp/max(1, tp+fn)
    )


def run_ws(prep, alpha, beta, ref):
    bw = prep["bw"]
    lab = prep["lab"]
    dist = prep["dist"]

    md = max(
        1, int(round(alpha*prep["rhat"]))
    )

    threshold = beta*prep["dmax"]

    if prep["nc"] == 0:
        pk = np.empty((0, 2), dtype=int)
        ws = np.zeros(bw.shape, dtype=int)

    else:
        pk = peak_local_max(
            dist,
            min_distance=md,
            threshold_abs=threshold,
            labels=lab,
            exclude_border=False
        )

        mk = np.zeros(bw.shape, dtype=int)

        for i, (y, x) in enumerate(pk, 1):
            mk[y, x] = i

        ws = segmentation.watershed(
            -dist, mk, mask=bw
        )

    count = int(
        np.count_nonzero(np.unique(ws) > 0)
    )

    require(
        count == int(ws.max()) == len(pk),
        "Marker/label count mismatch"
    )

    final = ws > 0

    require(
        not np.any(final & ~bw),
        "Watershed outside input mask"
    )

    seeded = set()

    if len(pk):
        seeded = set(
            lab[pk[:, 0], pk[:, 1]].tolist()
        )
        seeded.discard(0)

    pre = binary_metrics(bw, ref)
    post = binary_metrics(final, ref)

    bw_pixels = int(np.count_nonzero(bw))
    final_pixels = int(np.count_nonzero(final))

    result = dict(
        alpha=alpha,
        beta=beta,
        ws_n=count,
        otsu=prep["otsu"],
        componentes_n8=prep["nc"],
        areas_componentes=json.dumps(
            prep["areas"]
        ),
        rhat_px=prep["rhat"],
        min_distance_px=md,
        dist_max_px=prep["dmax"],
        threshold_abs=threshold,
        marcadores=len(pk),
        componentes_n8_sem_marcador=(
            prep["nc"]-len(seeded)
        ),
        pixels_pre=bw_pixels,
        pixels_final=final_pixels,
        pixels_sem_rotulo=bw_pixels-final_pixels,
        cobertura_final=(
            final_pixels/max(1, bw_pixels)
        )
    )

    for name, value in pre.items():
        result["pre_"+name] = value

    for name, value in post.items():
        result["final_"+name] = value

    return result


def load_baseline(path, lookup):
    old = pd.read_csv(path)

    baseline = {
        (int(r.contagem_esperada),
         int(r.amostra)): r
        for r in old.itertuples(index=False)
    }

    require(
        len(old) == len(baseline) == 600
        and set(baseline) == set(lookup),
        "Baseline keys mismatch"
    )

    for (n, s), r in baseline.items():

        require(
            int(r.ws_erro) == int(r.ws_n)-n,
            f"Baseline error mismatch: {n},{s}"
        )

        require(
            0 <= float(r.ws_dice) <= 1
            and 0 <= float(r.ws_iou) <= 1,
            "Invalid baseline metrics"
        )

    return baseline


def check_baseline(row, baseline):
    key = (
        row["contagem_esperada"],
        row["amostra"]
    )

    old = baseline[key]

    require(
        row["ws_n"] == int(old.ws_n),
        f"B5 count mismatch: {key}"
    )

    require(
        row["ws_erro"] == int(old.ws_erro),
        f"B5 error mismatch: {key}"
    )

    require(
        math.isclose(
            row["pre_dice"],
            float(old.ws_dice),
            rel_tol=0,
            abs_tol=1e-9
        ),
        f"B5 Dice mismatch: {key}"
    )

    require(
        math.isclose(
            row["pre_iou"],
            float(old.ws_iou),
            rel_tol=0,
            abs_tol=1e-9
        ),
        f"B5 IoU mismatch: {key}"
    )


def summarize(df):
    rows = []

    for tag in GRID:

        for density in [
            "Baixa", "Media", "Alta", "Todas"
        ]:

            rr = df[df.config == tag]

            if density != "Todas":
                rr = rr[rr.grupo == density]

            if rr.empty:
                continue

            e = rr.ws_erro.to_numpy()

            rows.append(dict(
                config=tag,
                grupo=density,
                alpha=GRID[tag][0],
                beta=GRID[tag][1],
                n_imagens=len(rr),
                soma_erro_absoluto=int(
                    np.abs(e).sum()
                ),
                mae=float(np.abs(e).mean()),
                vies=float(e.mean()),
                rmse=float(
                    np.sqrt(np.mean(e**2))
                ),
                mape=float(
                    100*np.mean(
                        np.abs(e) /
                        rr.contagem_esperada.to_numpy()
                    )
                ),
                exatas=int((e==0).sum()),
                subcontagens=int((e<0).sum()),
                supercontagens=int((e>0).sum()),
                dice_pre=float(rr.pre_dice.mean()),
                dice_final=float(rr.final_dice.mean()),
                iou_pre=float(rr.pre_iou.mean()),
                iou_final=float(rr.final_iou.mean()),
                precision_final=float(
                    rr.final_precision.mean()
                ),
                recall_final=float(
                    rr.final_recall.mean()
                ),
                raio_mediano_px=float(
                    rr.rhat_px.median()
                ),
                distancia_minima_media_px=float(
                    rr.min_distance_px.mean()
                ),
                marcadores_media=float(
                    rr.marcadores.mean()
                ),
                cobertura_final_media=float(
                    rr.cobertura_final.mean()
                )
            ))

    return pd.DataFrame(rows)


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2)+"\n",
        encoding="utf-8"
    )


def write_csv_atomic(df, path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base", type=Path, required=True
    )

    ap.add_argument(
        "--manifest", type=Path, required=True
    )

    ap.add_argument(
        "--baseline", type=Path, required=True
    )

    ap.add_argument(
        "--out", type=Path, required=True
    )

    a = ap.parse_args()

    require(
        skimage.__version__ == "0.26.0"
        and scipy.__version__ == "1.18.1"
        and np.__version__ == "2.5.4",
        "Unexpected scientific environment"
    )

    morphological_test()

    print("MORPHOLOGY TEST OK", flush=True)

    lookup = load_manifest(a.manifest)

    baseline = load_baseline(
        a.baseline, lookup
    )

    keys = sorted(
        (n, s) for n, s in lookup
        if s <= 5
    )

    require(
        len(keys) == 120,
        "Calibration size mismatch"
    )

    protocol_doc = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "protocolo_watershed_bstar.md"
    )

    require(
        protocol_doc.is_file(),
        "Missing methodological protocol"
    )

    a.out.mkdir(
        parents=True, exist_ok=False
    )

    protocol = dict(
        experiment="Watershed Bstar",
        partition="calibracao",
        samples="s01-s05",
        images=120,
        expected_rows=1080,
        grid=GRID,
        control="B5",
        selection="minimum absolute error sum",
        tie_break="B5 first, then lowest index",
        median_filter=3,
        otsu=True,
        remove_max_size=20,
        removal_connectivity=1,
        labeling_connectivity=8,
        watershed_connectivity=1,
        exclude_border=False,
        manifest_sha256=sha(a.manifest),
        baseline_sha256=sha(a.baseline),
        executor_sha256=sha(Path(__file__)),
        methodological_protocol_sha256=sha(
            protocol_doc
        ),
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        skimage=skimage.__version__,
        pandas=pd.__version__,
        tifffile=tifffile.__version__,
        note="Retrospective calibration only"
    )

    write_json(
        a.out/"protocolo.json",
        protocol
    )

    rows = []
    controls = 0

    order = ["B5"] + [
        tag for tag in GRID
        if tag != "B5"
    ]

    csv_path = a.out/"por_imagem_bstar.csv"

    partial_path = a.out/"checkpoint_parcial.csv"

    for index, (n, s) in enumerate(keys, 1):

        r = lookup[(n, s)]

        img = read_verified(
            a.base, r, "imagem"
        )

        ref = read_verified(
            a.base, r, "mascara"
        ) > 0

        require(
            img.shape == ref.shape
            and ref.any(),
            "Invalid reference mask"
        )

        prep = prepare(img)

        for tag in order:

            alpha, beta = GRID[tag]

            result = run_ws(
                prep, alpha, beta, ref
            )

            result.update(
                config=tag,
                contagem_esperada=n,
                amostra=s,
                grupo=group(n),
                particao="calibracao",
                ws_erro=result["ws_n"]-n
            )

            if tag == "B5":
                check_baseline(result, baseline)
                controls += 1

            rows.append(result)

        if index % 5 == 0:

            write_csv_atomic(
                pd.DataFrame(rows),
                partial_path
            )

            print(
                f"{index}/120 images | "
                f"{len(rows)}/1080 configurations | "
                f"B5 controls={controls}",
                flush=True
            )

    require(
        len(rows) == 1080 and controls == 120,
        "Incomplete calibration"
    )

    df = pd.DataFrame(rows)

    expected = {
        (tag, n, s)
        for n, s in keys
        for tag in GRID
    }

    actual = set(zip(
        df.config,
        df.contagem_esperada,
        df.amostra
    ))

    require(
        actual == expected
        and len(actual) == len(df),
        "Calibration key mismatch"
    )

    write_csv_atomic(df, csv_path)

    summary = summarize(df)

    require(
        len(summary) == 36,
        "Incomplete summary"
    )

    write_csv_atomic(
        summary,
        a.out/"resumo_por_grupo.csv"
    )

    ranking = summary[
        summary.grupo == "Todas"
    ].copy()

    ranking["tie_priority"] = (
        ranking.config.map(
            lambda tag: (
                0 if tag == "B5"
                else int(tag[1:])
            )
        )
    )

    ranking = ranking.sort_values(
        ["soma_erro_absoluto", "tie_priority"]
    )

    require(
        len(ranking) == 9,
        "Incomplete ranking"
    )

    write_csv_atomic(
        ranking,
        a.out/"ranking_calibracao.csv"
    )

    winner = ranking.iloc[0]
    tag = str(winner.config)

    selection = dict(
        selected=tag,
        alpha=GRID[tag][0],
        beta=GRID[tag][1],
        mae=float(winner.mae),
        absolute_error_sum=int(
            winner.soma_erro_absoluto
        ),
        calibration_images=120,
        calibration_csv_sha256=sha(csv_path),
        protocol_sha256=sha(
            a.out/"protocolo.json"
        ),
        executor_sha256=sha(Path(__file__)),
        baseline_sha256=sha(a.baseline),
        baseline_controls=controls,
        note="Selected exclusively on calibration"
    )

    write_json(
        a.out/"selecao_bstar.json",
        selection
    )

    partial_path.unlink(missing_ok=True)

    print("\nB5 CONTROLS OK: 120/120")

    print(
        ranking[[
            "config", "alpha", "beta",
            "mae", "vies", "rmse",
            "exatas", "supercontagens"
        ]].to_string(index=False)
    )

    print(
        "\nSELECTED:",
        tag,
        "alpha=", GRID[tag][0],
        "beta=", GRID[tag][1]
    )

    print("BSTAR CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
