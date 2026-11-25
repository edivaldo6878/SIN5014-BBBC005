
#!/usr/bin/env python3
# SIN5014 - ETAPA 05.14 - PASSO 182.8

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import executar_cellpose_bbbc005_v1 as original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    lookup = original.load_manifest(a.manifest)
    rows = []

    for (n, s), r in sorted(lookup.items()):

        img = original.read_verified(
            a.base, r, "imagem"
        )

        ref = original.read_verified(
            a.base, r, "mascara"
        ) > 0

        assert img.shape == ref.shape
        assert ref.any()

        lo, l, h, hi = np.percentile(
            img, [0.1, 1, 99, 99.9]
        )

        med = float(np.median(img[ref]))

        v0 = (med-l)/(h-l) if h>l else np.nan
        v1 = (med-lo)/(hi-lo) if hi>lo else np.nan

        rows.append(dict(
            n=n,
            amostra=s,
            grupo=original.group(n),
            particao=original.partition(s),
            foreground_fraction=ref.mean(),
            p0_1=lo,
            p1=l,
            p99=h,
            p99_9=hi,
            fg_median=med,
            norm_original=v0,
            norm_alternativa=v1,
            ratio_p99_p999=(
                (h-l)/(hi-l) if hi>l else np.nan
            ),
            ratio_norm_alt_original=(
                v1/v0
                if np.isfinite(v0) and v0!=0
                else np.nan
            )
        ))

        if len(rows) % 25 == 0:
            print(f"Verified {len(rows)}/600",
                  flush=True)

    assert len(rows) == 600

    a.out.mkdir(parents=True, exist_ok=False)

    d = pd.DataFrame(rows)

    d.to_csv(
        a.out/"percentis_por_imagem.csv",
        index=False
    )

    columns = [
        "foreground_fraction",
        "p0_1", "p1", "p99", "p99_9",
        "fg_median",
        "norm_original",
        "norm_alternativa",
        "ratio_p99_p999",
        "ratio_norm_alt_original"
    ]

    g = d.groupby("n")[columns].median()

    g.insert(
        0, "n_imagens", d.groupby("n").size()
    )

    g.to_csv(a.out/"percentis_por_nivel.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        d.foreground_fraction*100,
        d.ratio_norm_alt_original,
        s=12, alpha=.45,
        label="Images (n=600)"
    )

    ax.plot(
        g.foreground_fraction*100,
        g.ratio_norm_alt_original,
        "o-", markersize=4,
        label="Median by count level"
    )

    ax.set(
        xlabel="Reference foreground (%)",
        ylabel="Normalized intensity: alternative / original",
        title="BBBC005 - Foreground and intensity scaling"
    )

    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()

    fig.savefig(
        a.out/"fracao_vs_razao.png",
        dpi=240
    )

    plt.close(fig)

    protocol = dict(
        manifest_sha256=original.sha(a.manifest),
        script_sha256=original.sha(Path(__file__)),
        original_executor_sha256=original.sha(
            Path(original.__file__)
        ),
        images=len(d),
        numpy=np.__version__,
        pandas=pd.__version__,
        tifffile=tifffile.__version__,
        matplotlib=matplotlib.__version__,
        percentiles=[0.1, 1, 99, 99.9],
        note="Descriptive post-hoc raw pixel analysis"
    )

    (a.out/"protocolo.json").write_text(
        json.dumps(protocol, indent=2)+"\n",
        encoding="utf-8"
    )

    print(g.round(4).to_string())

    print(
        "Lower percentiles both zero:",
        int(((d.p0_1==0)&(d.p1==0)).sum())
    )

    print(
        "N=1 foreground below 1 percent:",
        int(((d.n==1)&(d.foreground_fraction<0.01)).sum())
    )

    print("DIAGNOSTICO PERCENTIS OK: 600 pares verificados")


if __name__ == "__main__":
    main()
