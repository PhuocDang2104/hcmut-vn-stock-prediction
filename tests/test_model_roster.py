from __future__ import annotations

from vnstock.utils.io import read_yaml
from vnstock.utils.paths import repo_root


EXPECTED_MODEL_CONFIGS = {
    "itransformer.yaml",
    "kronos.yaml",
    "patchtst.yaml",
    "xlstm_ts.yaml",
}

EXPECTED_MODEL_NOTEBOOKS = {
    "10_xlstm_ts_block.ipynb",
    "11_itransformer_block.ipynb",
    "12_patchtst_block.ipynb",
    "13_kronos_block.ipynb",
}

OBSOLETE_REFERENCES = (
    "10_cnn_block.ipynb",
    "11_lstm_block.ipynb",
    "13_tft_block.ipynb",
    "14_kronos_block.ipynb",
    "configs/models/cnn.yaml",
    "configs/models/lstm.yaml",
    "configs/models/tft.yaml",
    "data/processed/cnn/",
    "data/processed/lstm/",
    "data/processed/tft/",
    "run_cnn.py",
    "run_lstm.py",
    "run_tft.py",
)


def test_model_config_roster_matches_expected() -> None:
    model_config_dir = repo_root() / "configs" / "models"
    assert {path.name for path in model_config_dir.glob("*.yaml")} == EXPECTED_MODEL_CONFIGS


def test_dataset_builder_config_exports_only_current_models() -> None:
    config = read_yaml(repo_root() / "configs" / "data" / "dataset_daily.yaml")
    assert set(config["model_exports"]) == {"xlstm_ts", "itransformer", "patchtst", "kronos"}


def test_model_notebooks_match_expected_roster() -> None:
    notebooks_dir = repo_root() / "notebooks"
    assert {path.name for path in notebooks_dir.glob("*_block.ipynb")} == EXPECTED_MODEL_NOTEBOOKS


def test_obsolete_model_paths_are_removed() -> None:
    root = repo_root()
    obsolete_paths = [
        root / "src" / "vnstock" / "pipelines" / "run_cnn.py",
        root / "src" / "vnstock" / "pipelines" / "run_lstm.py",
        root / "src" / "vnstock" / "pipelines" / "run_tft.py",
        root / "configs" / "models" / "cnn.yaml",
        root / "configs" / "models" / "lstm.yaml",
        root / "configs" / "models" / "tft.yaml",
        root / "notebooks" / "10_cnn_block.ipynb",
        root / "notebooks" / "11_lstm_block.ipynb",
        root / "notebooks" / "12_itransformer_block.ipynb",
        root / "notebooks" / "13_tft_block.ipynb",
        root / "notebooks" / "14_kronos_block.ipynb",
    ]
    assert all(not path.exists() for path in obsolete_paths)


def test_docs_and_readme_reference_current_model_roster() -> None:
    text_files = [
        repo_root() / "README.md",
        repo_root() / "docs" / "data_contract.md",
        repo_root() / "docs" / "model_blocks.md",
    ]
    combined_text = "\n".join(path.read_text(encoding="utf-8") for path in text_files)

    assert "xLSTM-TS" in combined_text
    assert "PatchTST" in combined_text
    assert "iTransformer" in combined_text
    assert "Kronos" in combined_text

    for obsolete_reference in OBSOLETE_REFERENCES:
        assert obsolete_reference not in combined_text
