from pathlib import Path

from scripts import run_objective_video


def test_ensure_uvg_frames_exports_missing_frames(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        frames = tmp_path / run_objective_video.DATASET_ID / "frames"
        frames.mkdir(parents=True)
        for index in range(30):
            (frames / f"im{index + 1:05d}.png").touch()
        (frames / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_objective_video.subprocess, "run", fake_run)
    frames = run_objective_video.ensure_uvg_frames(tmp_path)

    assert frames == tmp_path / run_objective_video.DATASET_ID / "frames"
    assert len(calls) == 1
    assert calls[0][0][-2:] == ["--root", str(tmp_path)]
    assert calls[0][1]["check"] is True


def test_ensure_uvg_frames_reuses_complete_export(tmp_path, monkeypatch):
    frames = tmp_path / run_objective_video.DATASET_ID / "frames"
    frames.mkdir(parents=True)
    for index in range(30):
        (frames / f"im{index + 1:05d}.png").touch()
    (frames / "manifest.json").write_text("{}", encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("complete export should be reused")

    monkeypatch.setattr(run_objective_video.subprocess, "run", fail_run)
    assert run_objective_video.ensure_uvg_frames(tmp_path) == frames
