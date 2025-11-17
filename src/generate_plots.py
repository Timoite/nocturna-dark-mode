"""Generate synthetic plots for label_1 to balance the Nocturna dataset."""

import argparse
import math
import random
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MODEL_INPUT_FIGSIZE = (3.2, 3.2)  # ~320x320px @ 100dpi
WIDE_FIGSIZE = (3.6, 2.8)
SUBPLOT_FIGSIZE = (4.6, 4.6)
DEFAULT_TARGET = 9000
DEFAULT_DPI = 140
FILENAME_PADDING = 5


# Output directory
output_dir = Path("data/label_1")
output_dir.mkdir(parents=True, exist_ok=True)


# Style configurations
STYLES = [
    "default",
    "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-whitegrid",
    "ggplot",
    "bmh",
    "classic",
]
COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Blues",
    "Greens",
    "Reds",
    "Purples",
    "Oranges",
]
COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Desired total number of label_1 images. Defaults to label_0 count or 9000 if unknown.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Optional cap on how many new plots to create in this run (useful for testing).",
    )
    return parser.parse_args()


def list_image_files(directory: Path):
    if not directory.exists():
        return []
    return [
        path
        for path in directory.glob("**/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def infer_target_total(explicit_target: int | None) -> int:
    if explicit_target is not None and explicit_target > 0:
        return explicit_target

    label0_dir = Path("data/label_0")
    label0_files = list_image_files(label0_dir)
    if label0_files:
        return len(label0_files)

    print(
        f"label_0 is empty or missing, falling back to default target of {DEFAULT_TARGET} plots."
    )
    return DEFAULT_TARGET


def determine_starting_counter() -> int:
    existing_ids = []
    for path in output_dir.glob("*.png"):
        prefix = path.stem.split("_")[0]
        if prefix.isdigit():
            existing_ids.append(int(prefix))
    if existing_ids:
        return max(existing_ids) + 1
    return 0


def save_plot(filename: str, fig=None):
    global counter
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{counter:0{FILENAME_PADDING}d}_{filename}.png",
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    counter += 1


@contextmanager
def styled_figure(figsize):
    style = random.choice(STYLES)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        yield fig


def generate_line_plot():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        x = np.linspace(0, 10, 100)
        num_lines = random.randint(1, 5)
        for j in range(num_lines):
            y = np.sin(x + j) * random.uniform(0.5, 2) + random.uniform(-1, 1)
            plt.plot(x, y, label=f"Line {j+1}", linewidth=random.uniform(1, 2.5))
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Line Plot")
        if random.random() > 0.55:
            plt.legend(fontsize="small")
        if random.random() > 0.4:
            plt.grid(True, alpha=0.4)
        save_plot("line_plot")


def generate_bar_chart():
    with styled_figure(WIDE_FIGSIZE):
        categories = [f"Cat {j}" for j in range(random.randint(3, 8))]
        values = np.random.randint(10, 100, len(categories))
        colors_list = [random.choice(COLORS) for _ in categories]
        plt.bar(categories, values, color=colors_list)
        plt.xlabel("Categories")
        plt.ylabel("Values")
        plt.title("Bar Chart")
        if random.random() > 0.4:
            plt.grid(axis="y", alpha=0.3)
        save_plot("bar_chart")


def generate_scatter_plot():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        n_points = random.randint(50, 200)
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)
        colors_scatter = np.random.rand(n_points)
        sizes = np.random.randint(10, 100, n_points)
        scatter = plt.scatter(
            x,
            y,
            c=colors_scatter,
            s=sizes,
            alpha=0.6,
            cmap=random.choice(COLORMAPS),
        )
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Scatter Plot")
        if random.random() > 0.7:
            plt.colorbar(scatter, fraction=0.046, pad=0.04)
        if random.random() > 0.5:
            plt.grid(True, alpha=0.3)
        save_plot("scatter_plot")


def generate_histogram():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        data = np.random.randn(1000) * random.uniform(1, 5) + random.uniform(-3, 3)
        bins = random.randint(10, 40)
        plt.hist(
            data,
            bins=bins,
            color=random.choice(COLORS),
            alpha=0.7,
            edgecolor="black",
        )
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram")
        if random.random() > 0.5:
            plt.grid(axis="y", alpha=0.3)
        save_plot("histogram")


def generate_pie_chart():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        n_slices = random.randint(3, 8)
        sizes = np.random.randint(10, 100, n_slices)
        labels = [f"Slice {j}" for j in range(n_slices)]
        colors_pie = [random.choice(COLORS) for _ in range(n_slices)]
        explode = [0.1 if random.random() > 0.7 else 0 for _ in range(n_slices)]
        plt.pie(
            sizes,
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=random.randint(0, 360),
            explode=explode,
            pctdistance=0.8,
        )
        plt.title("Pie Chart")
        save_plot("pie_chart")


def generate_heatmap():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        data = np.random.randn(random.randint(5, 15), random.randint(5, 15))
        sns.heatmap(
            data,
            cmap=random.choice(COLORMAPS),
            annot=(random.random() > 0.55),
            fmt=".1f",
            cbar=True,
            square=False,
        )
        plt.title("Heatmap")
        save_plot("heatmap")


def generate_box_plot():
    with styled_figure(WIDE_FIGSIZE):
        n_boxes = random.randint(3, 7)
        data_boxes = [
            np.random.randn(100) * random.uniform(1, 3) + random.uniform(-2, 2)
            for _ in range(n_boxes)
        ]
        labels = [f"Group {j}" for j in range(n_boxes)]
        plt.boxplot(data_boxes, labels=labels, patch_artist=True)
        plt.ylabel("Values")
        plt.title("Box Plot")
        if random.random() > 0.4:
            plt.grid(axis="y", alpha=0.3)
        save_plot("box_plot")


def generate_area_plot():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        x = np.linspace(0, 10, 100)
        n_areas = random.randint(2, 5)
        for j in range(n_areas):
            y = np.abs(np.sin(x + j) * random.uniform(0.5, 2))
            plt.fill_between(x, y, alpha=0.3, label=f"Area {j+1}")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Area Plot")
        if random.random() > 0.5:
            plt.legend(fontsize="small")
        if random.random() > 0.5:
            plt.grid(True, alpha=0.3)
        save_plot("area_plot")


def generate_violin_plot():
    with styled_figure(WIDE_FIGSIZE):
        n_violins = random.randint(3, 6)
        data_violins = [
            np.random.randn(100) * random.uniform(1, 3)
            for _ in range(n_violins)
        ]
        positions = range(1, n_violins + 1)
        parts = plt.violinplot(
            data_violins,
            positions=positions,
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(random.choice(COLORS))
            pc.set_alpha(0.5)
        plt.xlabel("Groups")
        plt.ylabel("Values")
        plt.title("Violin Plot")
        plt.xticks(positions, [f"G{j}" for j in positions])
        save_plot("violin_plot")


def generate_step_plot():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        x = np.arange(0, 20)
        y = np.random.randint(0, 100, len(x))
        plt.step(
            x,
            y,
            where=random.choice(["pre", "post", "mid"]),
            linewidth=2,
            color=random.choice(COLORS),
        )
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Step Plot")
        if random.random() > 0.5:
            plt.grid(True, alpha=0.3)
        save_plot("step_plot")


def generate_subplot_figure():
    style = random.choice(STYLES)
    with plt.style.context(style):
        fig, axes = plt.subplots(2, 2, figsize=SUBPLOT_FIGSIZE)
        fig.suptitle("Subplot Figure")

        x = np.linspace(0, 10, 100)
        axes[0, 0].plot(x, np.sin(x))
        axes[0, 0].set_title("Sine Wave")
        axes[0, 0].grid(True)

        cats = ["A", "B", "C", "D"]
        vals = np.random.randint(10, 50, 4)
        axes[0, 1].bar(cats, vals, color=random.choice(COLORS))
        axes[0, 1].set_title("Bar Chart")

        x_scatter = np.random.randn(50)
        y_scatter = np.random.randn(50)
        axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6)
        axes[1, 0].set_title("Scatter Plot")

        data_hist = np.random.randn(500)
        axes[1, 1].hist(
            data_hist, bins=20, color=random.choice(COLORS), alpha=0.7
        )
        axes[1, 1].set_title("Histogram")

        fig.tight_layout()
        save_plot("subplot", fig)


def generate_errorbar_plot():
    with styled_figure(MODEL_INPUT_FIGSIZE):
        x = np.arange(5, 15)
        y = np.random.randint(10, 100, len(x))
        yerr = np.random.randint(5, 15, len(x))
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            capsize=5,
            color=random.choice(COLORS),
            linewidth=2,
        )
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Error Bar Plot")
        if random.random() > 0.5:
            plt.grid(True, alpha=0.3)
        save_plot("errorbar_plot")


PLOT_GENERATORS = [
    ("line plots", generate_line_plot),
    ("bar charts", generate_bar_chart),
    ("scatter plots", generate_scatter_plot),
    ("histograms", generate_histogram),
    ("pie charts", generate_pie_chart),
    ("heatmaps", generate_heatmap),
    ("box plots", generate_box_plot),
    ("area plots", generate_area_plot),
    ("violin plots", generate_violin_plot),
    ("step plots", generate_step_plot),
    ("subplot figures", generate_subplot_figure),
    ("error bar plots", generate_errorbar_plot),
]


def main():
    global counter
    args = parse_args()

    target_total = infer_target_total(args.target_total)
    current_total = len(list_image_files(output_dir))
    remaining = max(target_total - current_total, 0)

    if args.max_new is not None:
        remaining = min(remaining, max(args.max_new, 0))

    if remaining == 0:
        print(
            f"label_1 already has {current_total} images, which meets or exceeds the target of {target_total}."
        )
        return

    per_family = max(1, math.ceil(remaining / len(PLOT_GENERATORS)))
    counter = determine_starting_counter()

    print(
        f"Generating {remaining} new plots so label_1 matches the {target_total} samples available in label_0."
    )
    generated = 0
    for label, generator in PLOT_GENERATORS:
        print(f" → {label} ...")
        for _ in range(per_family):
            if generated >= remaining:
                break
            generator()
            generated += 1
        if generated >= remaining:
            break

    new_total = current_total + generated
    print(
        f"\n✓ Created {generated} plots. label_1 now holds {new_total} files (target {target_total})."
    )


if __name__ == "__main__":
    main()
