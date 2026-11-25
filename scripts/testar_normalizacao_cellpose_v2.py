
#!/usr/bin/env python3
# SIN5014 - ETAPA 05.10 - PASSO 136.6
# Teste controlado da normalizacao Cellpose

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
from cellpose import models

import executar_cellpose_bbbc005_v1 as original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    require = original.require

    require(
        version("cellpose") == "3.1.1.2",
        "Unexpected Cellpose version"
    )

    require(
        torch.__version__ == "2.7.1+cpu",
        "Unexpected PyTorch version"
    )

    proto_file = a.baseline_dir / "protocolo.json"
    csv_file = a.baseline_dir / "por_imagem_cellpose.csv"

    require(
        original.sha(proto_file) ==
        "0214ba786c5d4b89ee713832b6085735e04b43ca21a690a5d0d787066eed3161",
        "Baseline protocol hash mismatch"
    )

    require(
        original.sha(csv_file) ==
        "d945f06a4e9b84bde6e83813801fcec8723a85f882f2e0c8d793644b42e6aa62",
        "Baseline CSV hash mismatch"
    )

    proto = json.loads(
        proto_file.read_text(encoding="utf-8")
    )

    require(
        original.sha(Path(original.__file__)) ==
        proto["executor_sha256"],
        "Original executor hash mismatch"
    )

    for package, expected in proto["versions"].items():
        require(
            version(package) == expected,
            f"Version mismatch: {package}"
        )

    lookup = original.load_manifest(a.manifest)

    baseline_rows = original.read_csv(csv_file)

    expected = {
        (tag, n, s)
        for n, s in lookup
        for tag in original.CONFIGS
    }

    require(
        original.validate_checkpoint(
            baseline_rows, expected
        ) == expected,
        "Invalid baseline"
    )

    baseline = {
        (int(r["contagem_esperada"]),
         int(r["amostra"])): r
        for r in baseline_rows
        if r["config"] == "C1_calibrated"
    }

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
        "Unexpected model"
    )

    keys = [(1, 1), (1, 2), (5, 1), (5, 2)]

    diameter = original.CONFIGS["C1_calibrated"]

    configs = {
        "N0_control": True,
        "N1_percentile": {
            "percentile": [0.1, 99.9]
        }
    }

    a.out.mkdir(parents=True, exist_ok=False)

    protocol = {
        "purpose": "Post-hoc normalization pilot",
        "partition": "calibracao",
        "keys": keys,
        "diameter": diameter,
        "configs": configs,
        "original_params": original.PARAMS,
        "weights_sha256": original.sha(weights),
        "manifest_sha256": original.sha(a.manifest),
        "baseline_sha256": original.sha(csv_file),
        "script_sha256": original.sha(Path(__file__)),
        "versions": proto["versions"],
        "python": sys.version,
        "seed": 0,
        "threads": 4,
        "interop_threads": 1,
        "expected_rows": 8
    }

    (a.out / "protocolo.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8"
    )

    rows = []

    for config, normalization in configs.items():

        params = dict(original.PARAMS)
        params["normalize"] = normalization

        for n, s in keys:
            r = lookup[(n, s)]

            img = original.read_verified(
                a.base, r, "imagem"
            )

            ref = original.read_verified(
                a.base, r, "mascara"
            ) > 0

            mask, flows, styles = model.eval(
                img.copy(),
                diameter=diameter,
                **params
            )

            mask = np.asarray(mask)

            k, dice, iou = original.scores(mask, ref)

            cellprob = np.asarray(flows[2])

            require(
                cellprob.shape == ref.shape
                and np.isfinite(cellprob).all(),
                "Invalid cellprob map"
            )

            control_ok = "NA"

            if config == "N0_control":
                b = baseline[(n, s)]

                control_ok = (
                    k == int(b["cp_n"])
                    and math.isclose(
                        dice, float(b["cp_dice"]),
                        rel_tol=0, abs_tol=1e-10
                    )
                    and math.isclose(
                        iou, float(b["cp_iou"]),
                        rel_tol=0, abs_tol=1e-10
                    )
                )

                require(
                    control_ok,
                    f"Control mismatch: C{n}_s{s:02}"
                )

            name = f"C{n}_s{s:02}_{config}"

            np.save(
                a.out / f"{name}_labels.npy", mask
            )

            np.save(
                a.out / f"{name}_cellprob.npy",
                cellprob
            )

            rows.append({
                "config": config,
                "n": n,
                "amostra": s,
                "cp_n": k,
                "cp_erro": k - n,
                "cp_dice": dice,
                "cp_iou": iou,
                "controle_ok": control_ok,
                "fg_cellprob_median": float(
                    np.median(cellprob[ref])
                ),
                "bg_cellprob_median": float(
                    np.median(cellprob[~ref])
                ),
                "cellprob_positive_fraction": float(
                    np.mean(cellprob > 0)
                )
            })

            print(
                f"{len(rows)}/8 {name} "
                f"GT={n} CP={k} "
                f"Dice={dice:.6f} "
                f"IoU={iou:.6f} "
                f"Control={control_ok}",
                flush=True
            )

    require(len(rows) == 8, "Incomplete experiment")

    with (a.out / "teste_normalizacao.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        "TESTE NORMALIZACAO OK: "
        "8 inferencias, 4 imagens de calibracao"
    )


if __name__ == "__main__":
    main()
