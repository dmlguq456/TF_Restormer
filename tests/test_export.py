import json
import sys
import types
from pathlib import Path

import pytest
import torch

import tf_restormer.export as export


def test_select_checkpoint_exact_epoch(tmp_path):
    wanted = tmp_path / "epoch.0019.pth"
    wanted.touch()
    (tmp_path / "epoch.0025.pth").touch()

    path, epoch = export._select_checkpoint(tmp_path, epoch=19)

    assert Path(path) == wanted
    assert epoch == 19


def test_select_checkpoint_missing_exact_epoch_does_not_fall_back(tmp_path):
    (tmp_path / "epoch.0025.pth").touch()

    with pytest.raises(FileNotFoundError, match="epoch 0019"):
        export._select_checkpoint(tmp_path, epoch=19)


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_select_checkpoint_rejects_invalid_epoch(tmp_path, epoch):
    with pytest.raises(ValueError, match="non-negative integer"):
        export._select_checkpoint(tmp_path, epoch=epoch)


def test_export_checkpoint_records_exact_provenance(tmp_path, monkeypatch):
    source = tmp_path / "epoch.0019.pth"
    torch.save(
        {
            "model_state_dict": {
                "layer.weight": torch.arange(4, dtype=torch.float32),
                "layer.total_ops": torch.tensor(0.0, dtype=torch.float64),
            }
        },
        source,
    )
    config = tmp_path / "baseline.yaml"
    config.write_text("config:\n  train_phase: adversarial\n", encoding="utf-8")
    output = tmp_path / "release"

    monkeypatch.setattr(
        export,
        "_resolve_source_dir",
        lambda config_name, variant, epoch: source,
    )
    monkeypatch.setattr(export, "resolve_config", lambda variant, name: config)

    model_path = export.export_checkpoint(
        "baseline.yaml",
        output_dir=output,
        epoch=19,
    )

    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    assert set(payload["model_state_dict"]) == {"layer.weight"}
    metadata = json.loads((output / "export_metadata.json").read_text())
    assert metadata["source_epoch"] == 19
    assert metadata["source_checkpoint"] == "epoch.0019.pth"
    assert metadata["source_checkpoint_sha256"] == export._file_hash(source)
    assert metadata["model_sha256"] == export._file_hash(model_path)
    assert metadata["config_sha256"] == export._file_hash(output / "config.yaml")


def test_upload_rejects_provenance_epoch_mismatch(tmp_path):
    torch.save({"model_state_dict": {}}, tmp_path / "model.pt")
    (tmp_path / "config.yaml").write_text("config: {}\n", encoding="utf-8")
    (tmp_path / "export_metadata.json").write_text(
        json.dumps({"source_epoch": 25}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected epoch 19"):
        export.upload_to_hub(
            "baseline.yaml",
            repo_id="owner/repo",
            checkpoint_dir=tmp_path,
            expected_epoch=19,
            force=True,
        )


def test_upload_commits_exported_bundle_atomically(tmp_path, monkeypatch):
    model = tmp_path / "model.pt"
    config = tmp_path / "config.yaml"
    metadata = tmp_path / "export_metadata.json"
    model_card = tmp_path / "MODEL_CARD.md"
    torch.save({"model_state_dict": {}}, model)
    config.write_text("config: {}\n", encoding="utf-8")
    metadata.write_text(json.dumps({"source_epoch": 19}), encoding="utf-8")
    model_card.write_text("# Model card\n", encoding="utf-8")
    calls = {}

    class FakeOperation:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, private, exist_ok):
            calls["repo"] = (repo_id, private, exist_ok)

        def create_commit(self, *, repo_id, operations, commit_message):
            calls["commit"] = (repo_id, operations, commit_message)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(CommitOperationAdd=FakeOperation, HfApi=FakeApi),
    )

    url = export.upload_to_hub(
        "baseline.yaml",
        repo_id="owner/repo",
        checkpoint_dir=tmp_path,
        expected_epoch=19,
        force=True,
        private=False,
        model_card=model_card,
    )

    assert url == "https://huggingface.co/owner/repo"
    assert calls["repo"] == ("owner/repo", False, True)
    repo_id, operations, message = calls["commit"]
    assert repo_id == "owner/repo"
    assert message == "Release baseline checkpoint from epoch 19"
    assert [operation.path_in_repo for operation in operations] == [
        "model.pt",
        "config.yaml",
        "export_metadata.json",
        "README.md",
    ]
    assert Path(operations[1].path_or_fileobj) == config
    assert Path(operations[3].path_or_fileobj) == model_card
    assert (tmp_path / ".upload_hash").read_text() == export._bundle_hash(
        [model, config, metadata, model_card]
    )
