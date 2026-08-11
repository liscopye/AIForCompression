#!/usr/bin/env python3
"""Minimal single-process P-frame test for DCVC-RT and DCMVC on UVG PNG frames."""
import argparse, json, os, struct, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
sys.path.insert(0, str(PROJECT_ROOT))
from compression_pipeline.metrics import make_lpips_fn, process_memory_usage_mb, reset_torch_peak_memory, torch_memory_usage_mb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dcvc", "dcmvc"], required=True)
    p.add_argument("--data_dir", default=str(PROJECT_ROOT.parent / "Data" / "UVG_png" / "Twilight"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_frames", type=int, default=30)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument(
        "--dcmvc_timing_mode",
        choices=["bitstream", "compute", "memory_bitstream"],
        default="bitstream",
        help="For DCMVC only: use file bitstreams, forward compute, or in-memory entropy bitstreams.",
    )
    return p.parse_args()


def load_png_rgb_cpu(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def stage_pngs(pngs):
    """Read files before measurement; measured adapters start at host tensors."""
    return [load_png_rgb_cpu(path) for path in pngs]


def tensor_to_chw_np(tensor):
    return tensor.detach().float().clamp(0, 1).squeeze(0).cpu().numpy()


def pad_to_multiple(tensor, divisor=64):
    _, _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0, 0)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (0, pad_w, 0, pad_h)


def calc_psnr(orig, recon):
    mse = float(torch.mean((orig - recon) ** 2))
    if mse < 1e-30:
        return float("inf"), mse
    data_range = float(orig.max() - orig.min())
    if data_range < 1e-8:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


def bitstream_size(bit_stream):
    if isinstance(bit_stream, (bytes, bytearray)):
        return len(bit_stream)
    if isinstance(bit_stream, (list, tuple)):
        return sum(bitstream_size(x) for x in bit_stream)
    return len(bit_stream)


def pack_dcmvc_i(height, width, q_index, payload):
    flag = (1 << 7) + (q_index << 1)
    return struct.pack(">2IBI", height, width, flag, len(payload)) + payload


def unpack_dcmvc_i(stream):
    height, width, flag, length = struct.unpack(">2IBI", stream[:13])
    return height, width, (flag >> 7) > 0, (flag & 0x7F) >> 1, stream[13:13 + length]


def pack_dcmvc_p(q_index, frame_idx, payload):
    flag = (1 << 7) + (q_index << 1)
    return struct.pack(">BBI", flag, frame_idx, len(payload)) + payload


def unpack_dcmvc_p(stream):
    flag, frame_idx, length = struct.unpack(">BBI", stream[:6])
    return (flag >> 7) > 0, (flag & 0x7F) >> 1, frame_idx, stream[6:6 + length]


def test_dcvc(args, device, lpips_fn):
    """DCVC-RT P-frame test using DMC video model."""
    dcvc_root = PROJECT_ROOT / "models" / "DCVC"
    sys.path.insert(0, str(dcvc_root))
    sys.path.insert(0, str(dcvc_root / "src" / "cpp"))
    from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
    from src.models.video_model import DMC
    from src.models.image_model import DMCI
    from src.layers.cuda_inference import replicate_pad
    from src.utils.common import get_state_dict
    from src.utils.stream_helper import (SPSHelper, write_sps, write_ip, read_header,
                                         read_sps_remaining, read_ip_remaining, NalType)

    # Load I-frame model
    i_ckpt = str(CHECKPOINTS / "dcvc-rt" / "cvpr2025_image.pth.tar")
    i_net = DMCI().to(device).eval()
    i_net.load_state_dict(get_state_dict(i_ckpt))
    i_net.update(None)
    i_net.half()
    i_net.set_use_two_entropy_coders(True)  # > 720p

    # Load P-frame model
    p_ckpt = str(CHECKPOINTS / "dcvc-rt" / "cvpr2025_video.pth.tar")
    p_net = DMC().to(device).eval()
    p_net.load_state_dict(get_state_dict(p_ckpt))
    p_net.update(None)
    p_net.half()
    p_net.set_use_two_entropy_coders(True)

    png_dir = Path(args.data_dir)
    pngs = sorted(png_dir.glob("*.png"))[:args.max_frames]
    if not pngs:
        raise FileNotFoundError(f"No PNGs in {png_dir}")
    host_frames = stage_pngs(pngs)
    first = host_frames[0]
    _, _, h, w = first.shape
    print(f"DCVC-RT P-frame: {len(pngs)} frames, {w}x{h}", flush=True)

    results = []
    for qp in [0, 21, 42, 63]:
        print(f"  QP={qp}...", flush=True)
        bits = []
        psnrs = []
        mses = []
        lpips_values = []
        enc_times = []
        dec_times = []
        reset_torch_peak_memory(device)

        io_buf = __import__('io').BytesIO()
        sps_helper = SPSHelper()
        p_net.set_curr_poc(0)

        with torch.no_grad():
            for fi, host_frame in enumerate(host_frames):
                torch.cuda.synchronize()
                t0 = time.time()
                x_rgb = host_frame.to(device)
                x = rgb2ycbcr(x_rgb).half()
                x_padded, _ = pad_to_multiple(x, 64)
                # replicate_pad handles 64-pixel boundary pads
                padding_r, padding_b = DMCI.get_padding_size(h, w, 16)
                x_padded = replicate_pad(x, padding_b, padding_r)

                if fi == 0:  # I-frame
                    sps = {'sps_id': -1, 'height': h, 'width': w, 'ec_part': 1, 'use_ada_i': 0}
                    encoded = i_net.compress(x_padded, qp)
                    p_net.clear_dpb()
                    p_net.add_ref_frame(None, encoded['x_hat'])
                else:  # P-frame
                    fa_idx = [0, 1, 0, 2, 0, 2, 0, 2][fi % 8]
                    curr_qp = p_net.shift_qp(qp, fa_idx)
                    sps = {'sps_id': -1, 'height': h, 'width': w, 'ec_part': 1, 'use_ada_i': 0}
                    encoded = p_net.compress(x_padded, curr_qp)

                sps_id, _ = sps_helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                write_sps(io_buf, sps)
                stream_bytes = write_ip(io_buf, fi == 0, sps_id,
                                        qp if fi == 0 else curr_qp,
                                        encoded['bit_stream'])

                torch.cuda.synchronize()
                enc_times.append(time.time() - t0)
                bits.append(stream_bytes * 8)

        # Decode pass
        io_buf.seek(0)
        input_buf = __import__('io').BytesIO(io_buf.read())
        sps_helper2 = SPSHelper()
        p_net.set_curr_poc(0)

        with torch.no_grad():
            for fi, host_frame in enumerate(host_frames):
                x_rgb = host_frame.to(device)
                x = rgb2ycbcr(x_rgb).half()
                x_padded, _ = pad_to_multiple(x, 64)
                padding_r, padding_b = DMCI.get_padding_size(h, w, 16)
                x_padded = replicate_pad(x, padding_b, padding_r)

                torch.cuda.synchronize()
                t0 = time.time()

                header = read_header(input_buf)
                while header['nal_type'] == NalType.NAL_SPS:
                    sps = read_sps_remaining(input_buf, header['sps_id'])
                    sps_helper2.add_sps_by_id(sps)
                    header = read_header(input_buf)
                sps_id = header['sps_id']
                sps = sps_helper2.get_sps_by_id(sps_id)
                qp_val, bit_stream = read_ip_remaining(input_buf)

                if header['nal_type'] == NalType.NAL_I:
                    decoded = i_net.decompress(bit_stream, sps, qp_val)
                    p_net.clear_dpb()
                    p_net.add_ref_frame(None, decoded['x_hat'])
                else:
                    decoded = p_net.decompress(bit_stream, sps, qp_val)

                x_hat_ycbcr = decoded['x_hat'][:, :, :h, :w]
                # Convert YCbCr output back to RGB for PSNR
                x_hat_rgb = ycbcr2rgb(x_hat_ycbcr.float()).clamp(0, 1)
                _ = x_hat_rgb.detach().cpu()
                torch.cuda.synchronize()
                dec_times.append(time.time() - t0)

                psnr, mse = calc_psnr(x_rgb[:, :, :h, :w], x_hat_rgb)
                psnrs.append(psnr)
                mses.append(mse)
                if lpips_fn is not None:
                    lpips_values.append(lpips_fn(tensor_to_chw_np(x_rgb[:, :, :h, :w]), tensor_to_chw_np(x_hat_rgb)))

        total_bits = sum(bits)
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_mse = sum(mses) / len(mses)
        avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else None
        avg_enc = sum(enc_times) / len(enc_times)
        avg_dec = sum(dec_times) / len(dec_times)
        bpp = total_bits / (h * w * len(pngs))
        orig_bytes = h * w * 3 * len(pngs)
        encode_time_total = avg_enc * len(pngs)
        decode_time_total = avg_dec * len(pngs)
        memory = torch_memory_usage_mb(device)

        result = {
            "model_name": "DCVC-RT",
            "model_id": f"DCVC_RT_Pframe_q{qp}",
            "metric": "mse",
            "checkpoint": f"{i_ckpt}+{p_ckpt}",
            "bpp": bpp,
            "mse": avg_mse,
            "psnr": avg_psnr,
            "average_frame_psnr": avg_psnr,
            "bitstream_bytes": int(total_bits // 8),
            "original_bytes": orig_bytes,
            "compression_ratio": orig_bytes / (total_bits // 8) if total_bits > 0 else float("inf"),
            "encode_time_total": encode_time_total,
            "decode_time_total": decode_time_total,
            "encode_time_avg": avg_enc,
            "decode_time_avg": avg_dec,
            "encode_throughput": orig_bytes / encode_time_total if encode_time_total > 0 else None,
            "decode_throughput": orig_bytes / decode_time_total if decode_time_total > 0 else None,
            "encode_throughput_MBps": (orig_bytes / encode_time_total / 1e6) if encode_time_total > 0 else None,
            "decode_throughput_MBps": (orig_bytes / decode_time_total / 1e6) if decode_time_total > 0 else None,
            "groups": 1,
            "samples": len(pngs),
            "shape": [3, h, w],
            "image_eval_mode": "real",
            "video_coding_mode": "pframe",
            "lpips": avg_lpips,
            "memory_usage_MB": process_memory_usage_mb(),
            "memory_reserved_MB": memory.get("memory_reserved_MB"),
            "sample_wall_time_total": encode_time_total + decode_time_total,
        }
        results.append(result)
        print(f"    bpp={bpp:.4f} psnr={avg_psnr:.1f}dB cr={result['compression_ratio']:.0f} enc={avg_enc*1000:.1f}ms dec={avg_dec*1000:.1f}ms", flush=True)

    return results


def test_dcmvc(args, device, lpips_fn):
    """DCMVC P-frame test using DMC video model."""
    dcmvc_root = PROJECT_ROOT / "models" / "DCMVC"
    sys.path.insert(0, str(dcmvc_root.parent))  # add models/ to path
    # Ensure DCMVC is a package so relative imports work
    if not (dcmvc_root / "__init__.py").exists():
        (dcmvc_root / "__init__.py").touch()
    sys.modules.pop("DCMVC", None)
    sys.modules.pop("DCMVC.models", None)
    sys.modules.pop("DCMVC.utils", None)
    sys.modules.pop("DCMVC.transforms", None)

    from DCMVC.models.DCMVC_model import DMC
    from DCMVC.models.image_model import IntraNoAR
    from DCMVC.utils.stream_helper import get_state_dict, get_padding_size, encode_i, decode_i, filesize

    # Load I-frame model (MUST load state dict on CPU first, then to GPU)
    i_ckpt_path = str(CHECKPOINTS / "dcmvc" / "cvpr2023_image_psnr.pth.tar")
    i_state_dict = get_state_dict(i_ckpt_path)
    i_net = IntraNoAR(ec_thread=False, stream_part=1, inplace=True)
    i_net.load_state_dict(i_state_dict)
    i_net = i_net.to(device).eval()
    i_net.update(force=True)

    # Load P-frame model
    p_ckpt_path = str(CHECKPOINTS / "dcmvc" / "dcmvc_p_frame.pth.tar")
    p_state_dict = get_state_dict(p_ckpt_path)
    p_net = DMC(ec_thread=False, stream_part=1, inplace=True)
    p_net.load_state_dict(p_state_dict)
    p_net = p_net.to(device).eval()
    p_net.update(force=True)

    png_dir = Path(args.data_dir)
    pngs = sorted(png_dir.glob("*.png"))[:args.max_frames]
    host_frames = stage_pngs(pngs)
    first = host_frames[0]
    _, _, h, w = first.shape
    print(f"DCMVC P-frame: {len(pngs)} frames, {w}x{h}", flush=True)
    stream_dir = Path(args.output_dir) / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for q_idx in [0, 1, 2, 3]:
        print(f"  q_idx={q_idx}...", flush=True)
        bits = []
        psnrs = []
        mses = []
        lpips_values = []
        enc_times = []
        dec_times = []
        roundtrip_times = []
        reset_torch_peak_memory(device)

        dpb = {"ref_frame": None, "ref_feature": None, "ref_mv_feature": None,
               "ref_y": None, "ref_mv_y": None}

        with torch.no_grad():
            for fi, host_frame in enumerate(host_frames):
                torch.cuda.synchronize()
                adapter_start = time.time()
                x = host_frame.to(device)
                x_padded, _ = pad_to_multiple(x, 64)

                padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w, 16)
                x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

                torch.cuda.synchronize()
                t0 = time.time()

                gop_size = 32
                if fi % gop_size == 0:
                    if args.dcmvc_timing_mode == "compute":
                        torch.cuda.synchronize()
                        t0 = time.time()
                        decoded = i_net.encode_decode(x_padded, q_in_ckpt=True, q_index=q_idx)
                        torch.cuda.synchronize()
                        t1 = time.time()
                        bit = decoded["bit"]
                        t2 = t1
                    elif args.dcmvc_timing_mode == "memory_bitstream":
                        torch.cuda.synchronize()
                        t0 = time.time()
                        compressed = i_net.compress(x_padded, q_in_ckpt=True, q_index=q_idx)
                        stream = pack_dcmvc_i(h, w, q_idx, compressed["bit_stream"])
                        bit = len(stream) * 8
                        torch.cuda.synchronize()
                        t1 = time.time()
                        dec_h, dec_w, dec_q_in_ckpt, dec_q_index, payload = unpack_dcmvc_i(stream)
                        decoded = i_net.decompress(payload, dec_h, dec_w, dec_q_in_ckpt, dec_q_index)
                        torch.cuda.synchronize()
                        t2 = time.time()
                    else:
                        stream_path = stream_dir / f"dcmvc_q{q_idx}_frame{fi:04d}_i.bin"
                        torch.cuda.synchronize()
                        t0 = time.time()
                        compressed = i_net.compress(x_padded, q_in_ckpt=True, q_index=q_idx)
                        encode_i(h, w, True, q_idx, compressed["bit_stream"], stream_path)
                        bit = filesize(stream_path) * 8
                        torch.cuda.synchronize()
                        t1 = time.time()
                        dec_h, dec_w, dec_q_in_ckpt, dec_q_index, bit_stream = decode_i(stream_path)
                        decoded = i_net.decompress(bit_stream, dec_h, dec_w, dec_q_in_ckpt, dec_q_index)
                        torch.cuda.synchronize()
                        t2 = time.time()
                    dpb = {
                        "ref_frame": decoded["x_hat"],
                        "ref_feature": None,
                        "ref_mv_feature": None,
                        "ref_y": None,
                        "ref_mv_y": None,
                    }
                    recon = decoded["x_hat"]
                    bits.append(bit)
                    enc_times.append(t1 - t0)
                    dec_times.append(t2 - t1)
                else:
                    if args.dcmvc_timing_mode == "memory_bitstream":
                        torch.cuda.synchronize()
                        t0 = time.time()
                        compressed = p_net.compress(x_padded, dpb, q_in_ckpt=True, q_index=q_idx, frame_idx=fi % 4)
                        stream = pack_dcmvc_p(q_idx, fi % 4, compressed["bit_stream"])
                        bit = len(stream) * 8
                        torch.cuda.synchronize()
                        t1 = time.time()
                        dec_q_in_ckpt, dec_q_index, dec_frame_idx, payload = unpack_dcmvc_p(stream)
                        decoded = p_net.decompress(dpb, payload, h, w, dec_q_in_ckpt, dec_q_index, dec_frame_idx)
                        torch.cuda.synchronize()
                        t2 = time.time()
                        dpb = decoded["dpb"]
                        recon = dpb["ref_frame"]
                        bits.append(bit)
                        enc_times.append(t1 - t0)
                        dec_times.append(t2 - t1)
                    else:
                        stream_path = None
                        if args.dcmvc_timing_mode == "bitstream":
                            stream_path = str(stream_dir / f"dcmvc_q{q_idx}_frame{fi:04d}_p.bin")
                        if args.dcmvc_timing_mode == "compute":
                            torch.cuda.synchronize()
                            t0 = time.time()
                        result = p_net.encode_decode(x_padded, dpb, q_in_ckpt=True, q_index=q_idx, output_path=stream_path,
                                                      pic_height=h, pic_width=w, frame_idx=fi % 4)
                        if args.dcmvc_timing_mode == "compute":
                            torch.cuda.synchronize()
                            result["encoding_time"] = time.time() - t0
                            result["decoding_time"] = 0.0
                        dpb = result["dpb"]
                        recon = dpb["ref_frame"]
                        bits.append(result["bit"])
                        enc_times.append(result["encoding_time"])
                        dec_times.append(result["decoding_time"])

                x_hat = recon.float().clamp_(0, 1)
                x_hat = F.pad(x_hat, (-padding_l, -padding_r, -padding_t, -padding_b))
                x_hat = x_hat[:, :, :h, :w]
                _ = x_hat.detach().cpu()
                torch.cuda.synchronize()
                roundtrip_times.append(time.time() - adapter_start)

                psnr, mse = calc_psnr(x[:, :, :h, :w], x_hat)
                psnrs.append(psnr)
                mses.append(mse)
                if lpips_fn is not None:
                    lpips_values.append(lpips_fn(tensor_to_chw_np(x[:, :, :h, :w]), tensor_to_chw_np(x_hat)))

        total_bits = sum(bits)
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_mse = sum(mses) / len(mses)
        avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else None
        avg_enc = sum(enc_times) / len(enc_times) if enc_times else 0
        avg_dec = sum(dec_times) / len(dec_times) if dec_times else 0
        bpp = total_bits / (h * w * len(pngs))
        orig_bytes = h * w * 3 * len(pngs)
        encode_time_total = avg_enc * len(pngs)
        decode_time_total = avg_dec * len(pngs)
        memory = torch_memory_usage_mb(device)

        result = {
            "model_name": "DCMVC",
            "model_id": f"DCMVC_Pframe_q{q_idx}",
            "metric": "mse",
            "checkpoint": f"{i_ckpt_path}+{p_ckpt_path}",
            "bpp": bpp,
            "mse": avg_mse,
            "psnr": avg_psnr,
            "average_frame_psnr": avg_psnr,
            "bitstream_bytes": int(total_bits // 8),
            "original_bytes": orig_bytes,
            "compression_ratio": orig_bytes / (total_bits // 8) if total_bits > 0 else float("inf"),
            "encode_time_total": encode_time_total,
            "decode_time_total": decode_time_total,
            "encode_time_avg": avg_enc,
            "decode_time_avg": avg_dec,
            "encode_throughput": orig_bytes / encode_time_total if encode_time_total > 0 else None,
            "decode_throughput": orig_bytes / decode_time_total if decode_time_total > 0 else None,
            "encode_throughput_MBps": (orig_bytes / encode_time_total / 1e6) if encode_time_total > 0 else None,
            "decode_throughput_MBps": (orig_bytes / decode_time_total / 1e6) if decode_time_total > 0 else None,
            "groups": 1,
            "samples": len(pngs),
            "shape": [3, h, w],
            "image_eval_mode": "real",
            "video_coding_mode": "pframe",
            "timing_mode": args.dcmvc_timing_mode,
            "lpips": avg_lpips,
            "memory_usage_MB": process_memory_usage_mb(),
            "memory_reserved_MB": memory.get("memory_reserved_MB"),
            "sample_wall_time_total": sum(roundtrip_times),
        }
        results.append(result)
        print(f"    bpp={bpp:.4f} psnr={avg_psnr:.1f}dB cr={result['compression_ratio']:.0f} enc={avg_enc*1000:.1f}ms dec={avg_dec*1000:.1f}ms", flush=True)

    return results


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    lpips_fn = None if args.no_lpips else make_lpips_fn(device)

    if args.model == "dcvc":
        results = test_dcvc(args, device, lpips_fn)
    else:
        results = test_dcmvc(args, device, lpips_fn)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf = out / "summary.json"
    json.dump(results, open(sf, "w"), indent=2)
    print(f"Results: {sf}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
