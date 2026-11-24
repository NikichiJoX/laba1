import numpy as np
import pandas as pd

import learned_context as lc
from linear_predictor import decode_linear, encode_linear, fit_predictor
from codec_plots import plot_results


def decode_combined(path, model, context_table):
    with path.open("rb") as stream:
        header = stream.read(lc.HEADER.size)
    mode = lc.HEADER.unpack(header)[1]
    if mode == 3:
        return decode_linear(path, model)
    if mode == 2:
        return lc.decode_learned(path, context_table)
    raise ValueError(f"Неизвестный режим {mode}")


def evaluate_linear(b, baseline, train_images):
    folder = lc.RESULTS / f"b{b}"
    test_folder = lc.PROJECT / "test"
    files = [test_folder / name for name in baseline.filename]
    images = lc.load_images(files)
    pixels = images.shape[1] * images.shape[2]

    model_folder = lc.PROJECT / "models"
    encoder_path = model_folder / f"encoder_1_{b}.h5"
    decoder_path = model_folder / f"decoder_1_{b}.h5"
    encoder = lc.keras.models.load_model(encoder_path, compile=False)
    decoder = lc.keras.models.load_model(decoder_path, compile=False)

    train_q, _ = lc.quantize(encoder, train_images, b)
    model = fit_predictor(train_q, b)
    weights, limits, frequencies = model
    model_path = lc.MODELS / f"linear_b{b}.npz"
    np.savez_compressed(model_path, weights=weights, limits=limits,
                        freqs=frequencies.astype(np.uint16))

    symbols, scales = lc.quantize(encoder, images, b)
    context_table = lc.fit_entropy_model(encoder, b)
    rows = []
    latents = []
    for index, row in enumerate(baseline.itertuples()):
        q = symbols[index]
        scale = scales[index]
        prefix = f"{row.i:02d}"
        context_file = folder / f"{prefix}_Learned.nic"
        linear_file = folder / f"{prefix}_Linear.nic"
        combined_file = folder / f"{prefix}_Combined.nic"
        encode_linear(linear_file, q, scale, b, model)

        context_bytes = context_file.stat().st_size
        linear_bytes = linear_file.stat().st_size
        if linear_bytes < context_bytes:
            selected_file = linear_file
            mode = 3
        else:
            selected_file = context_file
            mode = 2

        data = selected_file.read_bytes()
        combined_file.write_bytes(data)
        latent = decode_combined(combined_file, model, context_table)
        expected = lc.dequantize(q, scale, b)
        np.testing.assert_array_equal(latent, expected)
        latents.append(latent)

        rows.append({
            "b": b,
            "filename": row.filename,
            "AE_bpp": row.AE_bpp,
            "Context_bpp": row.Learned_bpp,
            "Linear_bpp": 8 * linear_bytes / pixels,
            "Combined_bpp": 8 * combined_file.stat().st_size / pixels,
            "mode": mode,
            "AE_SSIM": row.AE_SSIM,
        })

    rgb = decoder.predict(np.stack(latents), batch_size=2, verbose=0)
    with np.load(folder / "reconstructions.npz") as saved:
        np.testing.assert_array_equal(rgb, saved["Learned"])

    for i, row in enumerate(rows):
        row["Combined_SSIM"] = lc.quality(images[i], rgb[i])
    lc.keras.backend.clear_session()
    return pd.DataFrame(rows), rgb


def run():
    baseline = lc.run()
    train_folder = lc.PROJECT / "train"
    train_files = sorted(train_folder.glob("*.png"))
    train_images = lc.load_images(train_files)

    measurements = []
    restored_images = {}
    for b, group in baseline.groupby("b"):
        group = group.sort_values("i")
        metrics, rgb = evaluate_linear(b, group, train_images)
        measurements.append(metrics)
        restored_images[b] = rgb

    combined = pd.concat(measurements, ignore_index=True)
    combined.to_csv(lc.RESULTS / "metrics_combined.csv", index=False)
    columns = ["AE_bpp", "Context_bpp", "Linear_bpp", "Combined_bpp", "AE_SSIM", "Combined_SSIM"]
    summary = combined.groupby("b")[columns].mean()
    print(summary.to_string())

    bd_columns = {"Combined_bpp": "Learned_bpp", "Combined_SSIM": "Learned_SSIM"}
    bd = lc.bd_comparison(summary.rename(columns=bd_columns))
    predictor_count = (combined["mode"] == 3).sum()
    print(f"BD-rate относительно исходного AE: {bd['BD_Rate_pct']:.3f}%")
    print(f"Выбран предиктор: {predictor_count} из {len(combined)}. Реконструкции совпали.")
    plot_results(baseline, combined, restored_images, lc.RESULTS)
