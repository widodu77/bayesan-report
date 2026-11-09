"""Shared style of the report figures, and saving of each figure with its LaTeX caption.

Each figure script writes figures/<name>.pdf for the report, figures/<name>.png for a quick look,
and figures/<name>.tex, a figure environment with the caption, so that the report can include it
with \\input{../code/figures/<name>.tex} after \\graphicspath{{../code/figures/}}.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent / "figures"
TEXT_WIDTH = 6.3  # inches: A4 width minus the 2.5 cm margins of the report template

# Fixed order of categorical colors, chosen to stay distinct under color-vision deficiencies.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
})


def save(fig, name, caption):
    """Write the figure as PDF and PNG, and its caption as a LaTeX figure environment."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    (FIGURES / f"{name}.tex").write_text(
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{name}.pdf}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{name}}}\n"
        "\\end{figure}\n")
