from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HEADACHE_COLORS = dict(zip(["migraine", "tension", "cluster", "sinus"], SLOTS))
MRI_COLORS = dict(zip(["glioma", "meningioma", "notumor", "pituitary"], SLOTS))
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "legend.frameon": False,
    }
)


def new_fig(nrows: int = 1, ncols: int = 1, **kw):
    kw.setdefault("figsize", (12, 6.75))
    return plt.subplots(nrows, ncols, **kw)


def headline(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.012, 0.975, title, fontsize=17, fontweight="bold", va="top", color=INK)
    if subtitle:
        fig.text(0.012, 0.915, subtitle, fontsize=11.5, va="top", color=INK_2)


def save(fig, out_dir: Path, name: str, top: float = 0.84) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def bar_labels(ax, bars, fmt: str = "{:,.0f}", horizontal: bool = False) -> None:
    for b in bars:
        if horizontal:
            ax.annotate(
                fmt.format(b.get_width()),
                (b.get_width(), b.get_y() + b.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
            )
        else:
            ax.annotate(
                fmt.format(b.get_height()),
                (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )


def draw_steps(fig_title: str, subtitle: str, steps: list[tuple[str, str, str]], out_dir: Path, name: str) -> Path:
    n = len(steps)
    fig = plt.figure(figsize=(12, 6.75))
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.84))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    headline(fig, fig_title, subtitle)
    style = {
        "done": ("#cde2fb", "#2a78d6", ""),
        "check": ("#f0efec", MUTED, "CHECK  "),
        "proposed": ("#fdf3dc", "#c98500", "PROPOSED  "),
    }
    h = 0.78
    for i, (title, detail, status) in enumerate(steps):
        y = n - 1 - i
        face, edge, tag = style[status]
        ax.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (0.02, y + (1 - h) / 2), 0.96, h, boxstyle="round,pad=0,rounding_size=0.08", fc=face, ec=edge, lw=1.2
            )
        )
        ax.text(0.045, y + 0.5, f"{i + 1}", fontsize=16, fontweight="bold", color=edge, va="center")
        ax.text(0.085, y + 0.62, title, fontsize=12.5, fontweight="bold", va="center", color=INK)
        ax.text(0.085, y + 0.32, tag + detail, fontsize=10, va="center", color=INK_2)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def table_fig(
    rows: list[list[str]],
    cols: list[str],
    title: str,
    subtitle: str,
    name: str,
    highlight: dict[int, str] | None = None,
    col_widths=None,
    *,
    out_dir: Path,
    cell_colors: dict[tuple[int, int], str] | None = None,
    fontsize: float = 10.5,
) -> Path:
    height = 0.4 * (len(rows) + 1) + 1.7
    fig = plt.figure(figsize=(13, height))
    ax = fig.add_axes((0.015, 0.02, 0.97, 1 - 1.55 / height))
    ax.set_axis_off()
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="left", colLoc="left", colWidths=col_widths, bbox=(0, 0, 1, 1))
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        if r == 0:
            cell.set_facecolor("#cde2fb")
            cell.set_text_props(weight="bold")
        elif cell_colors and (r - 1, c) in cell_colors:
            cell.set_facecolor(cell_colors[(r - 1, c)])
        elif highlight and r - 1 in highlight:
            cell.set_facecolor(highlight[r - 1])
    headline(fig, title, subtitle)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
