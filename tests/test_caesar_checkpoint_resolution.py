from __future__ import annotations

from compression_pipeline.caesar_runner import _resolve_caesar_checkpoint


def test_resolve_caesar_checkpoint_prefers_exact_pth(tmp_path):
    exact = tmp_path / "caesar_v.pth"
    exact.write_bytes(b"exact")
    (tmp_path / "caesar_v_tuning_Turb-Rot (1).pt").write_bytes(b"tuned")

    assert _resolve_caesar_checkpoint(tmp_path, "caesar_v") == exact


def test_resolve_caesar_checkpoint_accepts_tuned_pt_names(tmp_path):
    tuned = tmp_path / "caesar_d_tuning_Turb-Rot (1).pt"
    tuned.write_bytes(b"tuned")

    assert _resolve_caesar_checkpoint(tmp_path, "caesar_d") == tuned
