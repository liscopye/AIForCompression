import torch

from models.CAESAR.CAESAR.models.run_gae_cuda import PCA


def test_single_residual_vector_uses_identity_basis():
    pca = PCA(device="cpu").fit(torch.tensor([[1.0, -2.0, 3.0]]))

    torch.testing.assert_close(pca.components_, torch.eye(3))


def test_eigh_failure_recomputes_covariance_in_float64(monkeypatch):
    original_eigh = torch.linalg.eigh
    calls = []

    def fail_float32_once(matrix):
        calls.append(matrix.dtype)
        if matrix.dtype == torch.float32:
            raise RuntimeError("simulated ill-conditioned covariance")
        return original_eigh(matrix)

    monkeypatch.setattr(torch.linalg, "eigh", fail_float32_once)
    samples = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ],
        dtype=torch.float32,
    )
    pca = PCA(device="cpu").fit(samples)

    assert calls == [torch.float32, torch.float64]
    assert pca.components_.dtype == torch.float32
    torch.testing.assert_close(
        pca.components_ @ pca.components_.T,
        torch.eye(3),
        atol=1e-5,
        rtol=1e-5,
    )
