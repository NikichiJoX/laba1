import json
import os
import struct
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from tensorflow import keras

import CNNImageCodec as ae
from RangeCoder import ArithmeticDecoder, ArithmeticEncoder


ROOT = Path(os.environ.get("CODEC_DRIVE_ROOT", Path(__file__).parent)).resolve()
PROJECT = Path(os.environ.get("CODEC_PROJECT", Path(__file__).parent)).resolve()
RESULTS = ROOT / "cnnimagecodec_files_results"
MODELS = ROOT / "entropy_models"

HEADER = struct.Struct("<4sBBHHHe")


def load_images(files):
    images = [np.asarray(Image.open(path).convert("RGB")) for path in files]
    return np.stack(images).astype(np.float32) / 255


def quantize(enc, images, b):
    z = enc.predict(images, batch_size=2, verbose=0)
    scale = z.max(axis=(1, 2, 3)).astype(np.float16)
    divisor = np.where(scale > 0, scale, 1)
    z = z / divisor[:, None, None, None]
    z = np.clip(z, 0, 0.9999999)
    return (z * 2**b).astype(np.uint8), scale


def dequantize(q, scale, b):
    return (q.astype(np.float32) + 0.5) / 2**b * scale

def contexts(q):
    left = np.zeros_like(q)
    left[:, :, 1:, :] = q[:, :, :-1, :]
    up = np.zeros_like(q)
    up[:, 1:, :, :] = q[:, :-1, :, :]

    flat = q.reshape(len(q), -1)
    prev = np.zeros_like(flat)
    prev[:, 1:] = flat[:, :-1]
    channels = np.arange(q.shape[-1])
    return channels, left, up, prev.reshape(q.shape)


def histogram(q, k):
    counts = np.zeros((q.shape[-1], k, k, k, k), dtype=np.int64)
    c, left, up, prev = contexts(q)
    np.add.at(counts, (c, left, up, prev, q), 1)
    return counts


def frequencies(prob):
    k = prob.shape[-1]
    freq = np.floor(prob * (4096 - k)).astype(np.int64) + 1
    total = freq.sum(axis=-1, keepdims=True)
    return np.concatenate((freq, total), axis=-1)


def smooth(counts, prior, weight):
    total = counts.sum(axis=-1, keepdims=True)
    return (counts + weight * prior) / (total + weight)


def mixed_table(counts, alpha, mix):
    spatial = counts.sum(axis=3)
    cross = counts.sum(axis=(1, 2))
    marginal = cross.sum(axis=1) + 1
    prior = marginal / marginal.sum(axis=-1, keepdims=True)

    space_prob = smooth(spatial, prior[:, None, None, :], alpha)
    prev_prob = smooth(cross, prior[:, None, :], alpha)
    prob = (1 - mix) * space_prob[:, :, :, None, :]
    prob = prob + mix * prev_prob[:, None, None, :, :]
    return frequencies(prob)


def joint_table(counts, backoff, beta):
    prior = backoff[..., :-1] / backoff[..., -1:]
    return frequencies(smooth(counts, prior, beta))


def validation_bits(table, q, ctx):
    c, left, up, prev = ctx
    freq = table[c, left, up, prev, q]
    total = table[c, left, up, prev, -1]
    return -np.log2(freq / total).sum() / len(q)


def choose_params(counts, qval):
    ctx = contexts(qval)
    best_bits = np.inf
    for alpha in (1, 8, 32, 128):
        for mix in (0, 0.25, 0.5, 0.75, 1):
            table = mixed_table(counts, alpha, mix)
            bits = validation_bits(table, qval, ctx)
            if bits < best_bits:
                best_bits = bits
                best_alpha, best_mix = alpha, mix
                backoff = table

    best_beta = None
    for beta in (1, 8, 32, 128, 512):
        table = joint_table(counts, backoff, beta)
        bits = validation_bits(table, qval, ctx)
        if bits < best_bits:
            best_bits = bits
            best_beta = beta
    return best_alpha, best_mix, best_beta


def fit_entropy_model(enc, b):
    path = MODELS / f"context_joint_b{b}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["freqs"].astype(np.int64)

    files = sorted((PROJECT / "train").glob("*.png"))
    q, _ = quantize(enc, load_images(files), b)
    order = np.random.default_rng(42).permutation(len(q))
    nval = max(1, round(0.2 * len(q)))
    counts = histogram(q[order[nval:]], 2**b)
    qval = q[order[:nval]]
    alpha, mix, beta = choose_params(counts, qval)

    counts += histogram(qval, 2**b)
    table = mixed_table(counts, alpha, mix)
    if beta is not None:
        table = joint_table(counts, table, beta)
    np.savez_compressed(path, freqs=table.astype(np.uint16))
    return table


def setup_ae(b):
    k = 2**b
    ae.ALPHABET_SIZE = k
    ae.CONTEX_SIZE = ae.n3M1 * k
    ae.freqs = np.ones((ae.CONTEX_SIZE, k + 1), dtype=np.int64)
    ae.cumfreqs = np.zeros(k + 1, dtype=np.int64)


def context(q, y, x, c, prev):
    left = int(q[y, x - 1, c]) if x else 0
    up = int(q[y - 1, x, c]) if y else 0
    return c, left, up, prev


def encode_learned(path, q, scale, b, table):
    with path.open("wb") as stream:
        stream.write(HEADER.pack(b"NIC1", 2, b, *q.shape, scale))
        coder = ArithmeticEncoder(16, stream)
        prev = 0
        for y, x, c in np.ndindex(q.shape):
            freq = table[context(q, y, x, c, prev)]
            symbol = int(q[y, x, c])
            coder.write(freq, symbol, 2**b)
            prev = symbol
        coder.finish()


def decode_learned(path, table):
    with path.open("rb") as stream:
        _, _, b, h, w, channels, scale = HEADER.unpack(stream.read(HEADER.size))
        q = np.zeros((h, w, channels), dtype=np.uint8)
        coder = ArithmeticDecoder(16, stream)
        prev = 0
        for y, x, c in np.ndindex(q.shape):
            freq = table[context(q, y, x, c, prev)]
            cumulative = np.r_[0, np.cumsum(freq[:-1])]
            symbol = coder.read(freq, cumulative, 2**b)
            q[y, x, c] = symbol
            prev = symbol
    return dequantize(q, scale, b)


def quality(image, recon):
    return ae.ssim(image, recon, channel_axis=-1, data_range=1.0)


def compare_images(b, files, images):
    folder = RESULTS / f"b{b}"
    folder.mkdir(exist_ok=True)
    enc = keras.models.load_model(PROJECT / "models" / f"encoder_1_{b}.h5", compile=False)
    dec = keras.models.load_model(PROJECT / "models" / f"decoder_1_{b}.h5", compile=False)
    table = fit_entropy_model(enc, b)
    quantized, scales = quantize(enc, images, b)
    setup_ae(b)
    pixels = ae.w * ae.h
    rows = []
    base_latents, new_latents, jpeg_images = [], [], []

    for i, q in enumerate(quantized):
        prefix = f"{i:02d}"
        scale = scales[i]
        base_path = folder / f"{prefix}_AE.bin"
        new_path = folder / f"{prefix}_Learned.nic"

        ae.EntropyEncoder(base_path, q, *q.shape)
        base_q = ae.EntropyDecoder(base_path, *q.shape)
        base_latents.append(dequantize(base_q, scale, b))
        header = HEADER.pack(b"NIC1", 0, b, *q.shape, scale)
        (folder / f"{prefix}_AE.nic").write_bytes(header + base_path.read_bytes())

        encode_learned(new_path, q, scale, b, table)
        new_latents.append(decode_learned(new_path, table))

        bpp = base_path.stat().st_size * 8 / pixels
        qf, jpeg_bpp, jpeg_ssim = ae.JPEGRDSingleImage(images[i], bpp, i)
        jpeg_path = folder / f"{prefix}_JPEG.jpg"
        copyfile(Path("jpg") / f"image{i}.jpeg", jpeg_path)
        with Image.open(jpeg_path) as jpg:
            jpeg_images.append(np.asarray(jpg, dtype=np.float32) / 255)

        rows.append(dict(
            b=b, i=i, filename=files[i].name,
            AE_bpp=bpp, AE_full_bpp=bpp + HEADER.size * 8 / pixels,
            Learned_bpp=new_path.stat().st_size * 8 / pixels,
            JPEG_bpp=jpeg_bpp, JPEG_SSIM=jpeg_ssim, JPEG_QF=qf,
        ))

    base_rgb = dec.predict(np.stack(base_latents), batch_size=2, verbose=0)
    new_rgb = dec.predict(np.stack(new_latents), batch_size=2, verbose=0)
    for i, row in enumerate(rows):
        row["AE_SSIM"] = quality(images[i], base_rgb[i])
        row["Learned_SSIM"] = quality(images[i], new_rgb[i])
        rgb = np.rint(np.clip(base_rgb[i], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(folder / f"{i:02d}_AE.png")

    pics = dict(raw=images, AE=base_rgb, JPEG=np.stack(jpeg_images), Learned=new_rgb)
    np.savez_compressed(folder / "reconstructions.npz", **pics)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(folder / "metrics_20.csv", index=False)
    plot_examples(b, pics, metrics)
    keras.backend.clear_session()
    return metrics


def plot_examples(b, pics, metrics):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for row, key in enumerate(("raw", "AE", "JPEG", "Learned")):
        for i, ax in enumerate(axes[row]):
            ax.imshow(pics[key][i])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(key)
            if key == "raw":
                ax.set_title(metrics.at[i, "filename"])
            else:
                ssim = metrics.at[i, key + "_SSIM"]
                bpp = metrics.at[i, key + "_bpp"]
                ax.set_title(f"Q={ssim:.3f}  bpp={bpp:.3f}", fontsize=10)
    fig.suptitle(f"b={b}, первые пять изображений")
    fig.tight_layout()
    fig.savefig(RESULTS / f"comparison_final_b{b}.png", dpi=160)
    plt.close(fig)


def plot_rd(summary):
    fig, ax = plt.subplots(figsize=(9, 6))
    for key in ("AE", "JPEG", "Learned"):
        rate = summary[key + "_bpp"]
        quality = summary[key + "_SSIM"]
        ax.plot(rate, quality, "o-", label=key)
        for b in summary.index:
            ax.annotate(f"b={b}", (rate[b], quality[b]), xytext=(5, 4),
                        textcoords="offset points", fontsize=9)
    ax.axvspan(0.2, 0.9, color="gray", alpha=0.07)
    ax.set(xlabel="bpp", ylabel="SSIM")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "rd_final.png", dpi=180)
    plt.close(fig)


def bd_comparison(summary):

    br, nr = summary.AE_bpp, summary.Learned_bpp
    bq, nq = summary.AE_SSIM, summary.Learned_SSIM
    base = PchipInterpolator(bq, np.log(br), extrapolate=False)
    new = PchipInterpolator(nq, np.log(nr), extrapolate=False)
    lo = max(bq.min(), nq.min())
    hi = min(bq.max(), nq.max())
    if base(lo) < np.log(0.2):
        lo = brentq(lambda q: base(q) - np.log(0.2), lo, hi)
    if base(hi) > np.log(0.9):
        hi = brentq(lambda q: base(q) - np.log(0.9), lo, hi)
    delta = (new.integrate(lo, hi) - base.integrate(lo, hi)) / (hi - lo)
    bd_rate = 100 * np.expm1(delta)

    rlo = max(0.2, br.min(), nr.min())
    rhi = min(0.9, br.max(), nr.max())
    a, z = np.log([rlo, rhi])
    base_q = PchipInterpolator(np.log(br), bq, extrapolate=False)
    new_q = PchipInterpolator(np.log(nr), nq, extrapolate=False)
    bd_ssim = (new_q.integrate(a, z) - base_q.integrate(a, z)) / (z - a)
    return dict(
        BD_Rate_pct=float(bd_rate), BD_SSIM=float(bd_ssim),
        SSIM_interval=[float(lo), float(hi)], AE_bpp_interval=[float(np.exp(base(lo))), float(np.exp(base(hi)))],
        equal_bpp_interval=[float(rlo), float(rhi)], criterion_met=bool(bd_rate <= -10 or bd_ssim >= 0.05),
    )


def run():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    files = sorted((PROJECT / "test").glob("*.png"))[:20]
    images = load_images(files)

    cwd = Path.cwd()
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            Path("jpg").mkdir()
            tables = [compare_images(b, files, images) for b in range(1, 5)]
        finally:
            os.chdir(cwd)

    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(RESULTS / "metrics_all.csv", index=False)
    summary = metrics.groupby("b").mean(numeric_only=True)
    plot_rd(summary)
    result = bd_comparison(summary)
    (RESULTS / "bd_final.json").write_text(json.dumps(result, indent=2))
    print(f"BD-rate = {result['BD_Rate_pct']:.3f}%")
    return metrics
