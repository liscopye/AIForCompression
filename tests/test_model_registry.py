from compression_pipeline.model_registry import image_model_jobs


def test_lictcm_large_checkpoint_uses_large_model_id_and_loader_default(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints" / "lictcm"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "mse_lambda_0.05.pth.tar").touch()

    jobs = list(image_model_jobs(tmp_path, {"LIC_TCM"}))
    large_jobs = [job for job in jobs if job.checkpoint and job.checkpoint.endswith("mse_lambda_0.05.pth.tar")]

    assert len(large_jobs) == 1
    assert large_jobs[0].model_id == "LICTCM_mse_lambda_0.05_large"
