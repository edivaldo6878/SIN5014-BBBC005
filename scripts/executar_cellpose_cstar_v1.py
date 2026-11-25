
#!/usr/bin/env python3
# SIN5014 - ETAPA 05.11 - PASSO 151
# Cellpose Cstar: fixed percentile normalization

import argparse
import json
import sys
import time
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np
import psutil
import torch
from cellpose import models

import executar_cellpose_bbbc005_v1 as original

TAG = "Cstar_norm"
DIAMETER = original.CONFIGS["C1_calibrated"]

BASELINE_SHA = (
    "d945f06a4e9b84bde6e83813801fcec872"
    "3a85f882f2e0c8d793644b42e6aa62"
)

PROTOCOL_SHA = (
    "0214ba786c5d4b89ee713832b6085735e"
    "04b43ca21a690a5d0d787066eed3161"
)

require = original.require


def parameters():
    p = dict(original.PARAMS)
    p["normalize"] = {
        "percentile": [0.1, 99.9]
    }
    return p


def validate_rows(rows, keys):
    require(
        all(r["config"] == TAG for r in rows),
        "Unexpected configuration"
    )

    converted = [
        dict(r, config="C1_calibrated")
        for r in rows
    ]

    expected = {
        ("C1_calibrated", n, s)
        for n, s in keys
    }

    checked = original.validate_checkpoint(
        converted, expected
    )

    return {(n, s) for _, n, s in checked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--partition",
        choices=["calibracao", "validacao"],
        required=True
    )
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    baseline_file = (
        a.baseline_dir / "por_imagem_cellpose.csv"
    )

    protocol_file = (
        a.baseline_dir / "protocolo.json"
    )

    require(
        original.sha(baseline_file) == BASELINE_SHA,
        "Baseline CSV hash mismatch"
    )

    require(
        original.sha(protocol_file) == PROTOCOL_SHA,
        "Baseline protocol hash mismatch"
    )

    old = json.loads(
        protocol_file.read_text(encoding="utf-8")
    )

    require(
        original.sha(Path(original.__file__))
        == old["executor_sha256"],
        "Original executor hash mismatch"
    )

    for package, expected in old["versions"].items():
        require(
            version(package) == expected,
            f"Version mismatch: {package}"
        )

    require(
        sys.version == old["python"],
        "Python version mismatch"
    )

    require(
        platform.platform() == old["platform"],
        "Platform mismatch"
    )

    lookup = original.load_manifest(a.manifest)

    baseline_rows = original.read_csv(baseline_file)

    all_expected = {
        (tag, n, s)
        for n, s in lookup
        for tag in original.CONFIGS
    }

    require(
        original.validate_checkpoint(
            baseline_rows, all_expected
        ) == all_expected,
        "Incomplete baseline"
    )

    baseline = {
        (int(r["contagem_esperada"]),
         int(r["amostra"])): r
        for r in baseline_rows
        if r["config"] == "C1_calibrated"
    }

    keys = sorted(
        (n, s) for n, s in lookup
        if original.partition(s) == a.partition
    )

    require(
        len(keys) ==
        (120 if a.partition == "calibracao" else 480),
        "Unexpected partition size"
    )

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)

    model = models.CellposeModel(
        gpu=False, model_type="nuclei"
    )

    weights = Path(model.pretrained_model)

    require(
        weights.name == "nucleitorch_0"
        and original.sha(weights) == original.WEIGHTS
        and model.device.type == "cpu",
        "Invalid model or device"
    )

    require(
        np.isclose(float(model.diam_mean), 17),
        "Unexpected model diameter"
    )

    protocol = {
        "config": TAG,
        "model": "nuclei",
        "diameter": DIAMETER,
        "params": parameters(),
        "partition": a.partition,
        "expected_rows": len(keys),
        "manifest_sha256": original.sha(a.manifest),
        "weights_sha256": original.sha(weights),
        "executor_sha256": original.sha(Path(__file__)),
        "original_executor_sha256": original.sha(
            Path(original.__file__)
        ),
        "baseline_sha256": BASELINE_SHA,
        "baseline_protocol_sha256": PROTOCOL_SHA,
        "versions": old["versions"],
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": 4,
        "interop_threads": 1,
        "seed": 0,
        "warmup": "C10_s01 excluded",
        "timing": "model.eval wall seconds",
        "memory": "RSS and cumulative process peak",
        "note": "Post-hoc normalization sensitivity"
    }

    csv_path = a.out / "por_imagem_cellpose.csv"
    proto_path = a.out / "protocolo.json"

    if a.resume:
        require(
            json.loads(
                proto_path.read_text(encoding="utf-8")
            ) == protocol,
            "Protocol changed"
        )
        rows = original.read_csv(csv_path)
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        proto_path.write_text(
            json.dumps(protocol, indent=2) + "\n",
            encoding="utf-8"
        )
        rows = []
        original.atomic_csv(csv_path, rows)

    done = validate_rows(rows, keys)

    print(
        f"Completed {len(done)}/{len(keys)}",
        flush=True
    )

    if done != set(keys):

        warm = original.read_verified(
            a.base, lookup[(10, 1)], "imagem"
        )

        print("Warmup excluded", flush=True)

        model.eval(
            warm,
            diameter=DIAMETER,
            **parameters()
        )

        proc = psutil.Process()
        mib = 2**20

        for n, s in keys:

            if (n, s) in done:
                continue

            r = lookup[(n, s)]

            img = original.read_verified(
                a.base, r, "imagem"
            )

            ref = original.read_verified(
                a.base, r, "mascara"
            ) > 0

            require(
                img.shape == ref.shape,
                "Image/reference shape mismatch"
            )

            before = proc.memory_info().rss / mib

            t = time.perf_counter()

            mask, *_ = model.eval(
                img,
                diameter=DIAMETER,
                **parameters()
            )

            dt = time.perf_counter() - t
            mem = proc.memory_info()

            k, dice, iou = original.scores(
                np.asarray(mask), ref
            )

            peak = getattr(mem, "peak_wset", None)

            rows.append(dict(
                config=TAG,
                contagem_esperada=n,
                amostra=s,
                grupo=original.group(n),
                particao=a.partition,
                diametro_px=DIAMETER,
                cp_n=k,
                cp_erro=k-n,
                cp_dice=dice,
                cp_iou=iou,
                cp_tempo_s=dt,
                rss_antes_MiB=before,
                rss_depois_MiB=mem.rss/mib,
                peak_wset_processo_MiB=(
                    "" if peak is None else peak/mib
                )
            ))

            original.atomic_csv(csv_path, rows)
            done.add((n, s))

            print(
                f"{len(done)}/{len(keys)} "
                f"C{n}_s{s:02} "
                f"GT={n} CP={k} "
                f"Dice={dice:.4f} "
                f"time={dt:.3f}s",
                flush=True
            )

    rows = original.read_csv(csv_path)

    require(
        validate_rows(rows, keys) == set(keys),
        "Incomplete execution"
    )

    for group in [
        "Baixa", "Media", "Alta", "N=1", "Todas"
    ]:

        selected = [
            r for r in rows
            if group == "Todas"
            or r["grupo"] == group
            or (
                group == "N=1"
                and int(r["contagem_esperada"]) == 1
            )
        ]

        errors = np.array([
            int(r["cp_erro"]) for r in selected
        ])

        old_rows = [
            baseline[(
                int(r["contagem_esperada"]),
                int(r["amostra"])
            )]
            for r in selected
        ]

        old_errors = np.array([
            int(r["cp_erro"]) for r in old_rows
        ])

        print(
            group,
            "n=", len(selected),
            "MAE_C1=", np.abs(old_errors).mean(),
            "MAE_Cstar=", np.abs(errors).mean(),
            "zeros_C1=", sum(
                int(r["cp_n"]) == 0
                for r in old_rows
            ),
            "zeros_Cstar=", sum(
                int(r["cp_n"]) == 0
                for r in selected
            ),
            "exatas_Cstar=", int(
                np.count_nonzero(errors == 0)
            ),
            "Dice_Cstar=", np.mean([
                float(r["cp_dice"])
                for r in selected
            ]),
            flush=True
        )

    print("CSTAR COMPLETE", flush=True)


if __name__ == "__main__":
    main()
