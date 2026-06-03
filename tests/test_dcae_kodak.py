import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DCAE_DIR = ROOT / "models" / "DCAE"
sys.path.insert(0, str(DCAE_DIR))

test_kodak = pytest.importorskip("test_kodak")


def test_find_checkpoints_prefers_known_dcae_order():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        for name in ("mse_0.025.pth.tar", "mse_0.0018.pth.tar", "mse_0.05.pth.tar"):
            (ckpt_dir / name).write_bytes(b"ckpt")

        results = test_kodak.find_checkpoints(ckpt_dir)

    assert [item["model_id"] for item in results] == [
        "DCAE_mse_lmbda0.05",
        "DCAE_mse_lmbda0.025",
        "DCAE_mse_lmbda0.0018",
    ]


def test_find_images_returns_sorted_kodak_images_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        Image.new("RGB", (4, 4), color=(1, 2, 3)).save(root / "kodim02.png")
        Image.new("RGB", (4, 4), color=(4, 5, 6)).save(root / "kodim01.jpg")
        (root / "notes.txt").write_text("ignore", encoding="utf-8")

        images = test_kodak.find_images(root)

    assert [path.name for path in images] == ["kodim01.jpg", "kodim02.png"]
