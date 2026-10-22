"""The README's speed chart: stock Prophet (process pool) vs prophet-batch, fit + forecast, per platform.

    uv run --isolated --no-project --with matplotlib python tools/readme_chart.py     # → docs/images/speed.{svg,png}

Numbers: the medians of 3 runs on prophet-batch 0.1.1 from PyPI, 1,437 M5 FOODS items, default Prophet(),
analytic history intervals, from the results pages of github.com/sumedhekaru/prophet-batch-examples
(walmart_grocery_forecasting/results/: local-macos-m4.md, docker-linux-arm64.md, cloudrun-us-east1.md,
windows-i7-1260p.md, dataproc-us-east1.md: the Spark job's execution phase). Update them here when the pages change.
Colors: prophet-batch in the dataviz reference palette's orange (contrast ≥ 3:1 on the surface), stock Prophet in a
recessive gray with value labels on every bar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

# (platform, stock pool seconds, prophet-batch seconds, speed-up): the pages' medians; the speed-up is the median of
# each run's own ratio (as published), not the ratio of the median times
ROWS = [
    ('MacBook, Apple M4 (10 cores)', 53.1, 5.2, 10.2),
    ('Linux container on the same Mac', 57.2, 6.7, 8.6),
    ('Cloud Run job (4 vCPU)', 254.4, 23.6, 10.9),
    ('Windows laptop, i7-1260P (16 threads)', 298.9, 25.8, 11.6),
    ('Spark, Dataproc 2 × 4 cores (execution)', 234.5, 24.2, 9.7),
]
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
# prophet-batch carries the emphasis (the palette's orange); stock Prophet is the recessive baseline (a light gray,
# so it reads as the "before"); every bar keeps its value label, so the light fill never carries a number alone
BATCH, STOCK = '#eb6834', '#c9c7c1'
OUT = Path(__file__).resolve().parents[1] / 'docs' / 'images'


def bar(ax, y: float, width: float, height: float, color: str, rx: float, ry: float) -> None:
    """A horizontal bar from x=0, square at the baseline, rounded at its data end (radius rx, ry in data units:
    the two axes have different scales)."""
    rx, ry = min(rx, width / 2), min(ry, height / 2)
    x0, x1, y0, y1 = 0.0, width, y - height / 2, y + height / 2
    k = 0.5523  # cubic Bézier circle constant
    verts = [(x0, y0), (x1 - rx, y0), (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
             (x1, y1 - ry), (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1), (x0, y1), (x0, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor='none'))


def main() -> int:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(ROWS)
    h, gap = 0.30, 0.03          # bar thickness and the surface gap between a row's two bars (axis units)
    xmax = 340.0
    # data-end radius ≈ 4 px: the axes are ≈ 6.0 in × 150 dpi ≈ 900 px wide for 340 s and ≈ 3.2 in ≈ 480 px tall
    # for n rows
    rx, ry = 4 * xmax / 900, 4 * n / 480
    for i, (name, stock, batch, speedup) in enumerate(ROWS):
        y = n - 1 - i
        bar(ax, y + (h + gap) / 2, stock, h, STOCK, rx, ry)
        bar(ax, y - (h + gap) / 2, batch, h, BATCH, rx, ry)
        ax.text(stock + 4, y + (h + gap) / 2, f'{stock:.1f} s', va='center', ha='left', color=INK_2, fontsize=9)
        ax.text(batch + 4, y - (h + gap) / 2, f'{batch:.1f} s', va='center', ha='left', color=INK_2, fontsize=9)
        ax.text(xmax, y - (h + gap) / 2, f'{speedup:.1f}× faster', va='center', ha='right', color=INK,
                fontsize=10, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(ROWS)], color=INK, fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xticklabels(['0', '50', '100', '150', '200', '250', '300 s'], color=INK_2, fontsize=9)
    ax.tick_params(axis='both', length=0, pad=6)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    fig.text(0.012, 0.955, 'Fit + forecast of 1,437 Walmart items: seconds, lower is better', color=INK,
             fontsize=12, fontweight='bold', ha='left')
    fig.text(0.012, 0.905, 'Stock Prophet in a process pool on every core vs prophet-batch on the same cores · '
             'default Prophet() · median of 3 runs', color=INK_2, fontsize=9, ha='left')
    # legend: colored swatch beside ink text
    for j, (label, color) in enumerate([('prophet-batch', BATCH), ('stock Prophet, process pool', STOCK)]):
        x = 0.012 + j * 0.17
        fig.patches.append(matplotlib.patches.FancyBboxPatch((x, 0.03), 0.014, 0.028, boxstyle='round,pad=0,rounding_size=0.004',
                                                               transform=fig.transFigure, facecolor=color, edgecolor='none'))
        fig.text(x + 0.02, 0.044, label, color=INK, fontsize=9, va='center', ha='left')
    fig.subplots_adjust(left=0.33, right=0.985, top=0.86, bottom=0.14)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'speed.svg', facecolor=SURFACE)
    fig.savefig(OUT / 'speed.png', facecolor=SURFACE, dpi=200)
    print(f'wrote {OUT / "speed.svg"} and speed.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
