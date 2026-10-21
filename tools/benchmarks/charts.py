#!/usr/bin/env python3
"""Draw one bar chart per benchmark suite (benchmark-<suite>.png) from a comparison, and charts.md showing them.
Needs matplotlib."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import common

# One colour per value type: a type keeps its colour whichever others are present.
PALETTE = dict(zip(common.VALUE_TYPES, ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']))
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True})


def speedups(results):
    """{suite: {(op, operands, value_type, N): (median speedup, standard error of the log speedup)}}"""
    suites = {}
    for case in results['cases']:
        if 'speedup' not in case:  # incomplete, added or removed: nothing to draw
            continue
        f = common.case_factors(case['suite'], case['name'])
        suites.setdefault(f['suite'], {})[(f['op'], f['operands'], f['value_type'], f['N'])] = (
            case['speedup'], case['speedup_log_se'])
    return suites


def log_ticks(ax, lo, hi):
    ticks = [t for t in (0.25, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}x' for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


def draw_suite(path, suite, cases, title, floor):
    """One panel per operation and operand kind; bars = median speedup (baseline time / candidate time, log scale)
    per value type and size N, whiskers = ± standard error of the per-round log ratio, grey band = no-change floor."""
    panels = sorted({(op, operands) for op, operands, _, _ in cases})
    sizes = sorted({size for *_, size in cases})
    value_types = [vt for vt in common.VALUE_TYPES if any(key[2] == vt for key in cases)]
    step = 0.76 / len(value_types)  # the bars of one size share 0.76 of the slot
    values = [speedup for speedup, _ in cases.values()]
    lo, hi = min(0.6, min(values) / 1.2), max(24, max(values) * 1.6)  # one scale for the suite, room for the labels
    widen = max(1, len(value_types) / 2)  # more value types: wider panels, fewer per row, same room per bar label
    ncol = max(1, min(len(panels), round(4 / widen)))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * widen * ncol + 0.8, 3.3 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis('off')
    for ax, (op, operands) in zip(axes.flat, panels):
        for j, vt in enumerate(value_types):
            x = [i + (j - (len(value_types) - 1) / 2) * step for i in range(len(sizes))]
            for xi, size in zip(x, sizes):
                if (op, operands, vt, size) not in cases:
                    continue
                speedup, se = cases[op, operands, vt, size]
                low, high = speedup - speedup * math.exp(-se), speedup * math.exp(se) - speedup
                ax.bar(xi, speedup, width=0.9 * step, color=PALETTE[vt], zorder=2)
                ax.errorbar(xi, speedup, yerr=[[low], [high]], fmt='none', ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
                ax.annotate(f'{speedup:.2f}\n±{max(low, high):.2f}', (xi, speedup + high), textcoords='offset points',
                            xytext=(0, 2), ha='center', va='bottom', fontsize=6.3, color=INK2, linespacing=1.0)
        ax.set_yscale('log')
        ax.axhspan(1 / (1 + floor), 1 + floor, color=GRID, alpha=0.7, lw=0, zorder=0)  # the no-change band
        ax.axhline(1, color=AXIS, lw=1, zorder=1)
        ax.set_ylim(lo, hi)
        log_ticks(ax, lo, hi)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], fontsize=7.5)
        ax.set_title(f'{op}({operands})', fontsize=9.5)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='y', labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('speedup (log)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[vt], label=common.value_type_label(vt)) for vt in value_types]
    height = fig.get_figheight()  # title, then the legend below it, top left; offsets in inches
    fig.suptitle(f'{title}, Benchmark: {suite}', x=0.01, y=1 - 0.1 / height, ha='left', va='top', fontsize=10, fontweight='bold')
    fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.005, 1 - 0.32 / height), ncol=len(handles), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='example:\n  python3 tools/benchmarks/charts.py benchmark-results')
    parser.add_argument('results', type=Path,
                        help='directory with the comparison.json of one comparison; the images and charts.md are written there')
    outdir = parser.parse_args(argv).results
    results = json.loads((outdir / 'comparison.json').read_text())
    suites = speedups(results)
    if not suites:
        failed = f' (the comparison failed: {results["error"]})' if results.get('error') else ''
        sys.exit(f'{outdir}: no measured cases, nothing to draw{failed}')

    def label(side):
        return common.side_label(results, side, results['builds'][side]['provenance']['commit'][:10])

    candidate, baseline = label('candidate'), label('baseline')
    title = f'candidate {candidate} vs baseline {baseline}'
    floor = results['settings']['min_improvement_percent'] / 100
    # charts.md links the images by file name; push-charts.sh points the links at the uploaded copies for the PR comment.
    page = [f'## Benchmark charts: candidate {candidate} vs baseline {baseline}', '',
            'Median speedup (baseline time / candidate time) per value type and size N, ± standard error; '
            f'grey band = no change (within {100 * floor:g}%).', '']
    for suite, cases in sorted(suites.items()):
        path = outdir / f'benchmark-{suite}.png'
        draw_suite(path, suite, cases, title, floor)
        page += [f'![{suite}]({path.name})', '']
        print(path)
    (outdir / 'charts.md').write_text('\n'.join(page))
    print(outdir / 'charts.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
