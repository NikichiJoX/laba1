import numpy as np
from matplotlib import pyplot as plt


def plot_results(baseline, combined, restored_images, results):
    columns = ["b", "filename", "Combined_bpp", "Combined_SSIM"]
    new_rates = combined[columns]
    rates = baseline.merge(new_rates, on=["b", "filename"], validate="one_to_one")
    means = rates.groupby("b").mean(numeric_only=True)

    labels = {
        "raw": "Оригинал",
        "AE": "Исходный AE",
        "JPEG": "JPEG",
        "Learned": "Прежний контекст",
        "Combined": "Объединённый кодек",
    }
    markers = {"AE": "o", "JPEG": "s", "Learned": "^", "Combined": "D"}
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    for method, marker in markers.items():
        bpp = means[method + "_bpp"]
        ssim = means[method + "_SSIM"]
        ax.plot(bpp, ssim, marker=marker, label=labels[method])

    ax.set_xlabel("Бит на пиксель (bpp)")
    ax.set_ylabel("SSIM")
    ax.set_title("Среднее по 20 тестовым изображениям · b = 1, 2, 3, 4")
    ax.grid(alpha=0.25)
    ax.legend()
    graph_path = results / "rd_combined.png"
    fig.savefig(graph_path, dpi=140)
    plt.close(fig)

    b = 2
    level = rates[rates["b"] == b]
    examples = level.sort_values("i").head(5)
    indices = examples["i"].to_numpy()
    folder = results / f"b{b}"
    pictures = {}
    with np.load(folder / "reconstructions.npz") as saved:
        for method in ["raw", "AE", "JPEG", "Learned"]:
            pictures[method] = saved[method][indices]

    pictures["Combined"] = restored_images[b][indices]

    fig, axes = plt.subplots(5, 5, figsize=(12, 10.6), layout="constrained")
    for row, method in enumerate(labels):
        axes[row, 0].set_ylabel(labels[method])
        for col in range(len(examples)):
            ax = axes[row, col]
            image = pictures[method][col]
            ax.imshow(np.clip(image, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])

            example = examples.iloc[col]
            if method == "raw":
                ax.set_title(example["filename"], fontsize=9)
            else:
                bpp = example[method + "_bpp"]
                ssim = example[method + "_SSIM"]
                ax.set_title(f"{bpp:.3f} bpp · SSIM {ssim:.3f}", fontsize=8)

    fig.suptitle(f"Пять тестовых изображений при b = {b}")
    examples_path = results / f"comparison_combined_b{b}.png"
    fig.savefig(examples_path, dpi=140)
    plt.close(fig)
