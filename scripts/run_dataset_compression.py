import argparse
import fcntl
import gc
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from compression_pipeline.adapters.era5 import ERA5Adapter
from compression_pipeline.adapters.hurricane import HurricaneAdapter
from compression_pipeline.adapters.isotropic1024 import Isotropic1024Adapter
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.adapters.lysozyme import LysozymeAdapter
from compression_pipeline.adapters.nyx import NYXAdapter
from compression_pipeline.adapters.s2c import S2CAdapter
from compression_pipeline.adapters.shanghai_xray import ShanghaiXrayAdapter
from compression_pipeline.adapters.tomo_h5 import TomoH5Adapter
from compression_pipeline.adapters.turb_rot_npz import TurbRotNPZAdapter
from compression_pipeline.adapters.uvg import UVGAdapter
from compression_pipeline.caesar_runner import CAESAR_N_FRAMES, run_caesar_sequence
from compression_pipeline.cra5_runner import run_cra5_sample
from compression_pipeline.metrics import make_lpips_fn, reset_torch_peak_memory, torch_memory_usage_mb
from compression_pipeline.model_registry import image_model_jobs
from compression_pipeline.runner import run_image_grouped_sample
from compression_pipeline.torch_codecs import CompressAILikeCodec, DCVCRTCodec, DCMVCCodec, ForwardLikelihoodCodec


def parse_args():
    parser = argparse.ArgumentParser(description="Run supported codecs on ERA5 or Kodak through the shared data adapter.")
    parser.add_argument("--dataset", choices=["era5", "kodak", "tomo", "uvg", "hurricane", "s2c", "nyx", "shanghai_xray", "isot1024", "lysozyme", "turb_rot_npz"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=None, help="Subset: CRA5 DCAE WeConvene LIC_TCM LIC-HPCM RwkvCompress DCVC-RT DCMVC caesar_v caesar_d")
    parser.add_argument("--image_eval_mode", choices=["real", "forward"], default="real",
                        help="For image models, use true compress/decompress or forward+likelihood evaluation.")
    parser.add_argument("--max_model_jobs", type=int, default=-1, help="Limit number of checkpoint/model jobs after filtering; useful for smoke tests.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm to use the allocated GPU.")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--caesar_root", default=str(PROJECT_ROOT / "models" / "CAESAR"))
    parser.add_argument("--caesar_ckpt_dir", default=str(PROJECT_ROOT / "checkpoints" / "caesar"))
    parser.add_argument("--caesar_start_index", type=int, default=0)
    parser.add_argument("--caesar_eb", type=float, nargs="+", default=[1e-4],
                        help="CAESAR error bound(s); pass multiple to sweep, e.g. --caesar_eb 1e-4 5e-4 1e-3")
    parser.add_argument("--allow_cra5_adapted", action="store_true",
                        help="Allow experimental non-ERA5 CRA5 runs by resizing/replicating samples to 268x721x1440. Disabled by default for fair benchmarks.")
    parser.add_argument("--tomo_group_frames", type=int, default=1,
                        help="Stack N consecutive tomo frames as N-channel input (e.g. 3 for pseudo-RGB DCVC-RT/DCMVC evaluation).")
    parser.add_argument("--tile_size", type=int, default=None,
                        help="Tile large images into tile_size x tile_size blocks (for s2c dataset).")
    parser.add_argument("--turb_rot_section_index", type=int, default=0,
                        help="Section index used for Turb_Rot CAESAR sequence view.")
    parser.add_argument("--turb_rot_section_start", type=int, default=0,
                        help="First section index used for Turb_Rot 3-channel image samples.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_lpips", action="store_true", help="Disable optional LPIPS calculation.")
    return parser.parse_args()


def iter_dataset_samples(args):
    if args.dataset == "kodak":
        yield from KodakAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "tomo":
        yield from TomoH5Adapter(args.data_root, group_frames=args.tomo_group_frames).iter_samples(
            max_samples=args.max_samples,
            resolution=tuple(args.resolution) if args.resolution else None,
        )
        return
    if args.dataset == "uvg":
        yield from UVGAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "hurricane":
        yield from HurricaneAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "s2c":
        yield from S2CAdapter(args.data_root, tile_size=args.tile_size).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "nyx":
        yield from NYXAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "shanghai_xray":
        yield from ShanghaiXrayAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "isot1024":
        yield from Isotropic1024Adapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "lysozyme":
        yield from LysozymeAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    if args.dataset == "turb_rot_npz":
        yield from TurbRotNPZAdapter(
            args.data_root,
            section_index=args.turb_rot_section_index,
            section_start=args.turb_rot_section_start,
        ).iter_samples(max_samples=args.max_samples)
        return
    yield from ERA5Adapter(args.data_root).iter_samples(
        max_samples=args.max_samples,
        max_channels=args.max_channels,
        resolution=tuple(args.resolution) if args.resolution else None,
    )


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()} device={device}", flush=True)
    lpips_fn = None if args.no_lpips else make_lpips_fn(device)

    # Load existing results (resume-safe) so concurrent jobs don't overwrite each other
    existing_keys: set[tuple] = set()
    summary = []
    if summary_file.exists():
        try:
            existing = json.loads(summary_file.read_text(encoding="utf-8"))
            for r in existing:
                if "error" not in r:
                    key = (r.get("model_id", ""), r.get("sample_id", ""), r.get("eb", 0), r.get("start_index", 0))
                    existing_keys.add(key)
            summary = existing
        except Exception:
            pass

    def _safe_write_summary():
        """Atomically write summary with file locking to prevent concurrent overwrites."""
        with open(summary_file, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(summary, f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    requested_models = set(args.models) if args.models else None
    caesar_models = requested_caesar_models(requested_models)
    non_caesar_models = None if requested_models is None else requested_models - {"CAESAR", "caesar_v", "caesar_d", "CAESAR-V", "CAESAR-D"}

    if caesar_models:
        if args.dataset == "era5":
            sequence, timestamps = ERA5Adapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                max_channels=args.max_channels,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "kodak":
            sequence, timestamps = KodakAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "tomo":
            sequence, timestamps = TomoH5Adapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "hurricane":
            sequence, timestamps = HurricaneAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "nyx":
            sequence, timestamps = NYXAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "isot1024":
            sequence, timestamps = Isotropic1024Adapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "uvg":
            sequence, timestamps = UVGAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "s2c":
            from compression_pipeline.adapters.s2c import S2CAdapter
            adapter = S2CAdapter(args.data_root, tile_size=args.tile_size)
            sequence, timestamps = adapter.load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "lysozyme":
            sequence, timestamps = LysozymeAdapter(args.data_root).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        elif args.dataset == "turb_rot_npz":
            sequence, timestamps = TurbRotNPZAdapter(
                args.data_root,
                section_index=args.turb_rot_section_index,
                section_start=args.turb_rot_section_start,
            ).load_sequence(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        else:
            raise SystemExit("CAESAR requires a dataset with sequential structure")
        max_n_frame = max(CAESAR_N_FRAMES[name] for name in caesar_models)
        if args.max_samples > 0 and args.max_samples < args.caesar_start_index + max_n_frame:
            raise SystemExit(
                f"CAESAR requires at least {args.caesar_start_index + max_n_frame} contiguous samples, "
                f"got --max_samples {args.max_samples}"
            )
        for model_name in caesar_models:
            for eb in args.caesar_eb:
                # Skip if already completed (resume-safe, cross-job dedup)
                sample_id = f"{args.dataset}_{model_name}"
                key = (model_name, sample_id, eb, args.caesar_start_index)
                if key in existing_keys:
                    print(f"[model] skip {model_name} eb={eb} — already in {summary_file.name}", flush=True)
                    continue
                try:
                    print(f"[model] running {model_name} eb={eb} on {args.dataset} sequence", flush=True)
                    result = run_caesar_sequence(
                        sequence,
                        timestamps,
                        model_name=model_name,
                        caesar_root=args.caesar_root,
                        ckpt_dir=args.caesar_ckpt_dir,
                        output_dir=output_dir,
                        device=device,
                        batch_size=args.batch_size,
                        eb=eb,
                        start_index=args.caesar_start_index,
                        sample_id=sample_id,
                        collect_lpips=not args.no_lpips,
                    )
                    result["eb"] = eb
                    summary.append(result)
                    existing_keys.add(key)
                    print(json.dumps(result, indent=2), flush=True)
                except Exception as exc:
                    summary.append({"model_name": "CAESAR", "model_id": model_name, "metric": "mse", "eb": eb, "error": str(exc)})
                    print(f"[error] {model_name} eb={eb}: {exc}", flush=True)
                finally:
                    _safe_write_summary()

    jobs = list(image_model_jobs(args.project_root, non_caesar_models))
    if args.max_model_jobs > 0:
        jobs = jobs[:args.max_model_jobs]
    if non_caesar_models != set() and jobs:
        samples = list(iter_dataset_samples(args))
        if not samples:
            raise SystemExit(f"No samples found in {args.data_root}")
    else:
        samples = []

    for job in jobs:
        model = None
        try:
            print(f"[model] loading {job.model_id}", flush=True)
            model = job.loader(device)
            params = sum(p.numel() for p in model.parameters())
            codec = None
            if job.model_name != "CRA5":
                if job.codec_cls is not None:
                    codec_cls = job.codec_cls
                else:
                    codec_cls = CompressAILikeCodec if args.image_eval_mode == "real" else ForwardLikelihoodCodec
                codec = codec_cls(model, device=device, divisor=job.divisor, **job.codec_kwargs)
            for sample in samples:
                print(f"[sample] {job.model_id} {sample.sample_id}", flush=True)
                if job.model_name == "CRA5":
                    reset_torch_peak_memory(device)
                    result = run_cra5_sample(sample, model, device=device, allow_adapted=args.allow_cra5_adapted)
                    result.update(torch_memory_usage_mb(device))
                else:
                    reset_torch_peak_memory(device)
                    result = run_image_grouped_sample(
                        sample,
                        codec,
                        lpips_fn=lpips_fn,
                        memory_fn=lambda d=device: torch_memory_usage_mb(d),
                    )
                result.update({
                    "model_name": job.model_name,
                    "model_id": job.model_id,
                    "metric": job.metric,
                    "checkpoint": job.checkpoint,
                    "params": params,
                    "image_eval_mode": args.image_eval_mode if job.model_name != "CRA5" else "native",
                })
                summary.append(result)
                print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:
            summary.append({
                "model_name": job.model_name,
                "model_id": job.model_id,
                "metric": job.metric,
                "checkpoint": job.checkpoint,
                "error": str(exc),
            })
            print(f"[error] {job.model_id}: {exc}", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {summary_file}", flush=True)


def requested_caesar_models(requested_models):
    if requested_models is None:
        return []
    if "CAESAR" in requested_models:
        return ["caesar_v", "caesar_d"]
    models = []
    aliases = {
        "caesar_v": "caesar_v",
        "CAESAR-V": "caesar_v",
        "caesar_d": "caesar_d",
        "CAESAR-D": "caesar_d",
    }
    for requested in requested_models:
        if requested in aliases:
            models.append(aliases[requested])
    return models


if __name__ == "__main__":
    main()
