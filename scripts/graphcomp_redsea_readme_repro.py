#!/usr/bin/env python3
"""Run the GraphComp README workflow on the public RedSea GAN data.

This keeps the four README stages explicit:
1. graph initialization
2. graph representation learning
3. graph-to-grid reconstruction
4. error-bounded residual compression

The upstream scripts are path-hardcoded research scripts. This wrapper uses the
same public functions/model structure but makes the RedSea input and outputs
reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from torch_geometric.data import Data


REPO = Path(__file__).resolve().parents[1]
GRAPHCOMP_ROOT = REPO / "models" / "GraphComp"
sys.path.insert(0, str(GRAPHCOMP_ROOT))

from error_bounded import decompress, my_compress  # noqa: E402
from graph_representation_learning_gnn import AutoEncoder  # noqa: E402
from graph_representation_learning_gnn_cnn import ConvAutoEncoder  # noqa: E402


class WandbLogger:
    def __init__(self, enabled: bool, project: str, name: str, config: dict) -> None:
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self._wandb.init(project=project, name=name, config=config)
        except Exception as exc:
            print(f"wandb init failed ({exc}); continuing without wandb.")

    def log(self, metrics: dict) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def read_redsea(zip_path: Path, member: str, frames: int, height: int, width: int, frame_offset: int) -> np.ndarray:
    frame_bytes = height * width * np.dtype(np.float32).itemsize
    n_bytes = frames * frame_bytes
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            if frame_offset:
                fh.read(frame_offset * frame_bytes)
            raw = fh.read(n_bytes)
    if len(raw) != n_bytes:
        raise ValueError(f"Requested {n_bytes} bytes from {member}, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.float32).copy().reshape(frames, height, width)


def read_era5_npy(npy_path: Path, channel: int, frames: int, y_stride: int, x_stride: int, frame_offset: int) -> np.ndarray:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected ERA5 [C,T,H,W] array, got {arr.shape}: {npy_path}")
    c, t, h, w = map(int, arr.shape)
    if channel < 0 or channel >= c:
        raise ValueError(f"channel={channel} out of range for C={c}")
    if frame_offset < 0 or frame_offset >= t:
        raise ValueError(f"frame_offset={frame_offset} out of range for T={t}")
    n = min(frames, t - frame_offset)
    data = np.asarray(arr[channel, frame_offset : frame_offset + n, ::y_stride, ::x_stride], dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected selected ERA5 [T,H,W], got {data.shape}")
    return np.array(data, dtype=np.float32, copy=True)


def read_lysozyme_npy(
    npy_path: Path,
    frames: int,
    frame_offset: int,
    y_stride: int,
    x_stride: int,
    invalid_threshold: float,
    invalid_policy: str,
) -> tuple[np.ndarray, dict]:
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 5:
        raise ValueError(f"Expected Lysozyme [V,N,T,H,W] array, got {arr.shape}: {npy_path}")
    v, chunks, t_inner, h, w = map(int, arr.shape)
    if v < 1:
        raise ValueError(f"Lysozyme V dimension must be positive, got {arr.shape}")
    total = chunks * t_inner
    if frame_offset < 0 or frame_offset >= total:
        raise ValueError(f"frame_offset={frame_offset} out of range for flattened T={total}")
    n = min(frames, total - frame_offset)
    start_chunk = frame_offset // t_inner
    start_inner = frame_offset % t_inner
    end = frame_offset + n
    end_chunk = math.ceil(end / t_inner)
    data = np.asarray(arr[0, start_chunk:end_chunk, :, ::y_stride, ::x_stride], dtype=np.float32)
    data = np.array(data.reshape((end_chunk - start_chunk) * t_inner, data.shape[-2], data.shape[-1])[start_inner : start_inner + n], dtype=np.float32, copy=True)
    invalid = data >= float(invalid_threshold)
    invalid_count = int(invalid.sum())
    if invalid_count:
        if invalid_policy == "zero":
            data[invalid] = 0.0
        elif invalid_policy == "median":
            for i in range(data.shape[0]):
                mask = invalid[i]
                if not mask.any():
                    continue
                valid = data[i][~mask]
                fill = float(np.median(valid)) if valid.size else 0.0
                data[i][mask] = fill
        else:
            raise ValueError(f"Unsupported lysozyme invalid policy: {invalid_policy}")
    meta = {
        "source_layout": "V,N,T,H,W",
        "flattened_time_axis": "chunk_then_frame",
        "frame_offset": int(frame_offset),
        "invalid_threshold": float(invalid_threshold),
        "invalid_policy": invalid_policy,
        "invalid_count": invalid_count,
        "invalid_fraction": float(invalid_count / data.size),
    }
    return data, meta


def densify_segments(segments: np.ndarray) -> np.ndarray:
    _, dense = np.unique(segments, return_inverse=True)
    return dense.reshape(segments.shape).astype(np.int32)


def segment_edges(segments: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in ((segments[:, :-1], segments[:, 1:]), (segments[:-1, :], segments[1:, :])):
        mask = a != b
        if not np.any(mask):
            continue
        pairs = np.stack((a[mask], b[mask]), axis=1)
        for u, v in np.unique(pairs, axis=0):
            uu, vv = int(u), int(v)
            if uu > vv:
                uu, vv = vv, uu
            edges.add((uu, vv))
    return sorted(edges)


def build_graph_from_edges(frame: np.ndarray, segments: np.ndarray, edges: list[tuple[int, int]], n_segments: int) -> nx.Graph:
    flat_labels = segments.reshape(-1)
    flat_values = frame.reshape(-1).astype(np.float64)
    sums = np.bincount(flat_labels, weights=flat_values, minlength=n_segments)
    counts = np.bincount(flat_labels, minlength=n_segments)
    means = sums / np.maximum(counts, 1)
    g = nx.Graph()
    g.add_nodes_from(range(n_segments))
    g.add_edges_from(edges)
    for region, mean in enumerate(means):
        g.nodes[region]["mean temperature"] = float(mean)
    return g


def graph_to_grid(rag: nx.Graph, segments: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.empty(shape, dtype=np.float32)
    for region in np.unique(segments):
        out[segments == region] = rag.nodes[int(region)]["mean temperature"]
    return out


def graph_to_data(g: nx.Graph, device: torch.device) -> tuple[Data, torch.Tensor]:
    nodes = list(g.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    x = torch.tensor([g.nodes[n]["mean temperature"] for n in nodes], dtype=torch.float32).view(-1, 1)
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in g.edges()]
    if edges:
        edge_index = torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    adj = nx.to_numpy_array(g, nodelist=nodes, dtype=np.float32)
    adj = torch.from_numpy(adj)
    return Data(x=x, edge_index=edge_index).to(device), adj.to(device)


def normalize_graphs(graphs: list[nx.Graph]) -> tuple[list[nx.Graph], float, float]:
    vals = np.array([d["mean temperature"] for g in graphs for _, d in g.nodes(data=True)], dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std() if vals.std() > 0 else 1.0)
    out = []
    for g in graphs:
        h = g.copy()
        for _, d in h.nodes(data=True):
            d["mean temperature"] = (d["mean temperature"] - mean) / std
        out.append(h)
    return out, mean, std


def psnr(data: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((data.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    data_range = float(np.ptp(data))
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def train_autoencoder(
    graphs: list[nx.Graph],
    epochs: int,
    lr: float,
    device: torch.device,
    logger: WandbLogger,
) -> tuple[AutoEncoder, list[float]]:
    model = AutoEncoder(input_size=1, hidden_size=1, output_size=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total = 0.0
        for g in graphs:
            data, adj = graph_to_data(g, device)
            opt.zero_grad(set_to_none=True)
            z = model.encoder(data.x, data.edge_index)
            pred_adj = model.decoder(z)
            loss = F.mse_loss(pred_adj, adj)
            loss.backward()
            opt.step()
            total += float(loss.item())
        epoch_loss = total / max(len(graphs), 1)
        losses.append(epoch_loss)
        logger.log({"gnn/epoch": len(losses), "gnn/loss": epoch_loss, "gnn/lr": lr})
    return model, losses


@torch.no_grad()
def encode_latents(model: AutoEncoder, graphs: list[nx.Graph], device: torch.device) -> np.ndarray:
    model.eval()
    latents = []
    for g in graphs:
        data, _ = graph_to_data(g, device)
        latents.append(model.encoder(data.x, data.edge_index).cpu().numpy())
    return np.stack(latents, axis=0).astype(np.float32)


def normalize_latents(latents: np.ndarray) -> tuple[torch.Tensor, float, float]:
    # Upstream code treats nodes as batch and time as the 1D signal length.
    latent_tensor = torch.from_numpy(latents).permute(1, 0, 2).float()
    min_val = float(latent_tensor.min())
    max_val = float(latent_tensor.max())
    denom = max(max_val - min_val, 1e-12)
    return (latent_tensor - min_val) / denom, min_val, max_val


def train_cnn_latent_autoencoder(
    normalized_latents: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    logger: WandbLogger,
) -> tuple[ConvAutoEncoder, list[float], np.ndarray, np.ndarray]:
    model = ConvAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    data = normalized_latents.permute(0, 2, 1).to(device)  # [nodes, 1, frames]
    target_len = data.shape[-1]
    model.train()
    for _ in range(epochs):
        total = 0.0
        seen = 0
        perm = torch.randperm(data.shape[0], device=device)
        for start in range(0, data.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            batch = data.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            # The upstream ConvTranspose stack is length-sensitive; align it so
            # short smoke tests and longer series use the same model.
            if recon.shape[-1] < batch.shape[-1]:
                recon = F.pad(recon, (0, batch.shape[-1] - recon.shape[-1]))
            recon = recon[..., : batch.shape[-1]]
            loss = F.mse_loss(recon, batch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.shape[0]
            seen += int(batch.shape[0])
        epoch_loss = total / max(seen, 1)
        losses.append(epoch_loss)
        logger.log({"cnn/epoch": len(losses), "cnn/loss": epoch_loss, "cnn/lr": lr})
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(data).cpu().numpy().astype(np.float32)
        decoded_t = model(data)
        if decoded_t.shape[-1] < target_len:
            decoded_t = F.pad(decoded_t, (0, target_len - decoded_t.shape[-1]))
        decoded = decoded_t[..., :target_len].cpu().numpy().astype(np.float32)
    return model, losses, encoded, decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_kind", choices=["redsea", "era5", "lysozyme"], default="redsea")
    parser.add_argument("--zip_path", type=Path, default=Path("/workspace/Redsea_t2_gan.zip"))
    parser.add_argument("--member", default="Redsea_t2_500_gan.dat")
    parser.add_argument("--era5_npy", type=Path, default=None)
    parser.add_argument("--era5_channel", type=int, default=0)
    parser.add_argument("--lysozyme_npy", type=Path, default=None)
    parser.add_argument("--lysozyme_invalid_threshold", type=float, default=4e9)
    parser.add_argument("--lysozyme_invalid_policy", choices=["median", "zero"], default="median")
    parser.add_argument("--y_stride", type=int, default=1)
    parser.add_argument("--x_stride", type=int, default=1)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--height", type=int, default=855)
    parser.add_argument("--width", type=int, default=1215)
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cnn_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--cnn_batch_size", type=int, default=16)
    parser.add_argument("--ebs", default="1e-2,1e-3,1e-4")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--skip_residual", action="store_true")
    parser.add_argument("--skip_reconstruction_save", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="graphcomp-era5")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--output_dir", type=Path, default=REPO / "unified_results" / "graphcomp_redsea_readme_repro")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    wandb_name = args.wandb_name or f"graphcomp_{args.input_kind}_ch{args.era5_channel}_f{args.frames}_s{args.scale:g}_m{args.min_size}"
    logger = WandbLogger(
        args.wandb,
        args.wandb_project,
        wandb_name,
        {
            "input_kind": args.input_kind,
            "era5_npy": str(args.era5_npy) if args.era5_npy else None,
            "era5_channel": args.era5_channel,
            "lysozyme_npy": str(args.lysozyme_npy) if args.lysozyme_npy else None,
            "lysozyme_invalid_threshold": args.lysozyme_invalid_threshold,
            "lysozyme_invalid_policy": args.lysozyme_invalid_policy,
            "frames": args.frames,
            "frame_offset": args.frame_offset,
            "scale": args.scale,
            "sigma": args.sigma,
            "min_size": args.min_size,
            "epochs": args.epochs,
            "cnn_epochs": args.cnn_epochs,
            "lr": args.lr,
            "cnn_lr": args.cnn_lr,
            "cnn_batch_size": args.cnn_batch_size,
            "device": str(device),
            "skip_residual": args.skip_residual,
            "skip_reconstruction_save": args.skip_reconstruction_save,
        },
    )

    timings = {}
    t0 = time.time()
    input_extra_meta = {}
    if args.input_kind == "redsea":
        frames = read_redsea(args.zip_path, args.member, args.frames, args.height, args.width, args.frame_offset)
    elif args.input_kind == "era5":
        if args.era5_npy is None:
            raise ValueError("--era5_npy is required when --input_kind era5")
        frames = read_era5_npy(args.era5_npy, args.era5_channel, args.frames, args.y_stride, args.x_stride, args.frame_offset)
        args.height, args.width = int(frames.shape[1]), int(frames.shape[2])
    else:
        if args.lysozyme_npy is None:
            raise ValueError("--lysozyme_npy is required when --input_kind lysozyme")
        frames, input_extra_meta = read_lysozyme_npy(
            args.lysozyme_npy,
            args.frames,
            args.frame_offset,
            args.y_stride,
            args.x_stride,
            args.lysozyme_invalid_threshold,
            args.lysozyme_invalid_policy,
        )
        args.height, args.width = int(frames.shape[1]), int(frames.shape[2])
    timings["read"] = time.time() - t0

    # README step 1: graph initialization.
    t1 = time.time()
    segments = densify_segments(felzenszwalb(img_as_float(frames[0]), scale=args.scale, sigma=args.sigma, min_size=args.min_size))
    n_segments = int(segments.max()) + 1
    edges = segment_edges(segments)
    graphs = [build_graph_from_edges(frame, segments, edges, n_segments) for frame in frames]
    timings["graph_initialization"] = time.time() - t1
    joblib.dump(graphs, args.output_dir / "graph_scale.pkl")
    np.save(args.output_dir / "segments.npy", segments.astype(np.int32))

    # README step 2: graph representation learning.
    t2 = time.time()
    norm_graphs, node_mean, node_std = normalize_graphs(graphs)
    model, losses = train_autoencoder(norm_graphs, args.epochs, args.lr, device, logger)
    timings["graph_representation_learning"] = time.time() - t2
    torch.save(
        {
            "model": model.state_dict(),
            "node_mean": node_mean,
            "node_std": node_std,
            "scale": args.scale,
            "sigma": args.sigma,
            "min_size": args.min_size,
            "losses": losses,
        },
        args.output_dir / "auto_ggl.pt",
    )
    latents = encode_latents(model, norm_graphs, device)
    latents.tofile(args.output_dir / "latent_representations.dat")

    # Upstream gnn_cnn step: train a 1D CNN autoencoder over per-node temporal
    # latent series and save its encoded representation.
    t2b = time.time()
    normalized_latents, latent_min, latent_max = normalize_latents(latents)
    cnn_model, cnn_losses, cnn_encoded, cnn_decoded = train_cnn_latent_autoencoder(
        normalized_latents,
        args.cnn_epochs,
        args.cnn_lr,
        args.cnn_batch_size,
        device,
        logger,
    )
    timings["cnn_latent_autoencoder"] = time.time() - t2b
    torch.save(
        {
            "model": cnn_model.state_dict(),
            "latent_min": latent_min,
            "latent_max": latent_max,
            "losses": cnn_losses,
        },
        args.output_dir / "auto_cnn.pt",
    )
    cnn_encoded.tofile(args.output_dir / "auto_cnn_latent.dat")
    np.save(args.output_dir / "auto_cnn_decoded_latents.npy", cnn_decoded)

    # README step 3: graph2grid reconstruction. Upstream graph2grid uses node
    # mean temperature from the graph, so this is the exact reconstruction used
    # for residual prediction.
    t3 = time.time()
    pred = np.stack([graph_to_grid(g, segments, (args.height, args.width)) for g in graphs], axis=0)
    timings["graph2grid"] = time.time() - t3
    if not args.skip_reconstruction_save:
        np.save(args.output_dir / "reconstructed_grid.npy", pred)
    predictor_psnr = psnr(frames, pred)

    flat_data = np.ascontiguousarray(frames.astype(np.float32).reshape(-1))
    flat_pred = np.ascontiguousarray(pred.astype(np.float32).reshape(-1))
    data_range = float(np.ptp(flat_data))
    results = []
    if not args.skip_residual:
        # README step 4: error-bounded residual compression.
        t4 = time.time()
        with tempfile.TemporaryDirectory(prefix="graphcomp_readme_repro_") as tmpdir:
            for eb_text in args.ebs.split(","):
                eb = float(eb_text)
                cmp_path = Path(tmpdir) / f"residual_eb{eb:g}.sz"
                c0 = time.time()
                cmp_size, _ = my_compress(flat_data.copy(), flat_pred.copy(), eb, str(cmp_path))
                c_sec = time.time() - c0
                d0 = time.time()
                recon = decompress(str(cmp_path), flat_data.copy(), flat_pred.copy(), eb)
                d_sec = time.time() - d0
                saved_cmp = args.output_dir / cmp_path.name
                saved_cmp.write_bytes(cmp_path.read_bytes())
                mse = float(np.mean((flat_data.astype(np.float64) - recon.astype(np.float64)) ** 2))
                result = {
                    "eb": eb,
                    "relative_bound": eb * data_range,
                    "residual_size_bytes": int(cmp_size),
                    "residual_bpp": float(cmp_size * 8 / flat_data.size),
                    "psnr_db": psnr(flat_data, recon),
                    "mse": mse,
                    "max_abs_error": float(np.max(np.abs(flat_data - recon))),
                    "compress_sec": c_sec,
                    "decompress_sec": d_sec,
                }
                results.append(result)
                logger.log({f"residual/eb_{eb:g}_bpp": result["residual_bpp"], f"residual/eb_{eb:g}_psnr": result["psnr_db"]})
        timings["error_bounded"] = time.time() - t4

    side_files = [
        args.output_dir / "segments.npy",
        args.output_dir / "graph_scale.pkl",
        args.output_dir / "auto_ggl.pt",
        args.output_dir / "latent_representations.dat",
        args.output_dir / "auto_cnn.pt",
        args.output_dir / "auto_cnn_latent.dat",
    ]
    side_bytes = sum(p.stat().st_size for p in side_files)
    side_bpp = float(side_bytes * 8 / flat_data.size)
    for result in results:
        result["side_bpp"] = side_bpp
        result["total_bpp_with_side"] = float(result["residual_bpp"] + side_bpp)
        result["total_size_bytes_with_side"] = int(result["residual_size_bytes"] + side_bytes)
    summary = {
        "workflow": [
            "graph_initialization",
            "graph_representation_learning",
            "graph2grid",
            "error_bounded",
        ],
        "input": {
            "kind": args.input_kind,
            "zip_path": str(args.zip_path),
            "member": args.member,
            "era5_npy": str(args.era5_npy) if args.era5_npy else None,
            "era5_channel": args.era5_channel if args.input_kind == "era5" else None,
            "lysozyme_npy": str(args.lysozyme_npy) if args.lysozyme_npy else None,
            "extra": input_extra_meta,
            "spatial_stride": [args.y_stride, args.x_stride] if args.input_kind in {"era5", "lysozyme"} else None,
            "shape": list(frames.shape),
            "frame_offset": int(args.frame_offset),
            "dtype": str(frames.dtype),
            "data_min": float(np.min(frames)),
            "data_max": float(np.max(frames)),
            "data_range": data_range,
        },
        "segmentation": {
            "scale": args.scale,
            "sigma": args.sigma,
            "min_size": args.min_size,
            "n_segments": int(n_segments),
            "n_edges": int(len(edges)),
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "device": str(device),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "cnn_epochs": args.cnn_epochs,
            "cnn_lr": args.cnn_lr,
            "cnn_initial_loss": cnn_losses[0] if cnn_losses else None,
            "cnn_final_loss": cnn_losses[-1] if cnn_losses else None,
            "cnn_encoded_shape": list(cnn_encoded.shape),
        },
        "predictor_psnr_db": predictor_psnr,
        "side_information": {
            "files": {p.name: p.stat().st_size for p in side_files},
            "bytes": int(side_bytes),
            "bpp": side_bpp,
            "note": "This includes implementation artifacts for reproducibility, not an optimized entropy-coded bitstream.",
        },
        "results": results,
        "timing_sec": timings,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.log(
        {
            "summary/predictor_psnr_db": predictor_psnr,
            "summary/side_bpp": summary["side_information"]["bpp"],
            "summary/n_segments": summary["segmentation"]["n_segments"],
        }
    )
    logger.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
