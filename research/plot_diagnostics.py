"""Render the measured development diagnostics, with no inferred leaderboard rank."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    report = json.loads((root / 'report.json').read_text())
    rows = list(csv.DictReader((root / 'shots.csv').open()))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    deficits = sorted(report['weighted_deficit'].items(), key=lambda pair: pair[1])
    axes[0, 0].barh([k for k, _ in deficits], [v * 1e4 for _, v in deficits], color='#356c9b')
    axes[0, 0].set(title='What costs score?', xlabel='Weighted score loss (× 10⁻⁴)')
    groups = {k: report['groups'][k] for k in ('ramp_up', 'flat_top', 'ramp_down')}
    positions = np.arange(len(groups))
    axes[0, 1].bar(positions - .18, [v['frame_share'] for v in groups.values()], .36,
                   label='Share of frames', color='#356c9b')
    axes[0, 1].bar(positions + .18, [v['flux_sse_share'] for v in groups.values()], .36,
                   label='Share of flux error', color='#d28e32')
    axes[0, 1].set_xticks(positions, ['Ramp up', 'Flat top', 'Ramp down'])
    axes[0, 1].set(title='Which discharge phase?', ylabel='Fraction')
    axes[0, 1].legend(frameon=False)
    image = axes[1, 0].imshow(np.load(root / 'spatial-rmse.npy'), origin='lower', cmap='magma')
    axes[1, 0].set(title='Flux root mean square error', xlabel='R grid index', ylabel='Z grid index')
    fig.colorbar(image, ax=axes[1, 0], label='Flux units')
    ordered = np.array([float(row['flux_sse']) for row in rows])
    axes[1, 1].plot(np.arange(1, len(rows) + 1) / len(rows), ordered.cumsum() / ordered.sum(),
                    color='#356c9b', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], '--', color='gray', label='Uniform error')
    axes[1, 1].set(title='How concentrated is flux error?', xlabel='Fraction of shots, worst first',
                   ylabel='Fraction of squared flux error', xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f'Eight-model ensemble · 256 development shots · local S = {report["score"]["S"]:.6f}',
                 fontsize=17)
    fig.savefig(root / 'diagnostics.png', dpi=170)
    fig.savefig(root / 'diagnostics.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
