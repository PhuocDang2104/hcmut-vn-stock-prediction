from __future__ import annotations


def build_model_spec(config: dict) -> dict[str, object]:
    return {
        "family": "patchtst",
        "seq_len": config["seq_len"],
        "pred_len": config["pred_len"],
        "patch_len": config["patch_len"],
        "patch_stride": config["patch_stride"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "target": config["target"],
        "note": "Attach the specific PatchTST implementation or wrapper here.",
    }
