#!/usr/bin/env python3
"""Plot a passed, provenance-checked paired inference snapshot, not extrapolations."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics


def measurements(report):
    if report.get('status') != 'passed' or report.get('provenance_errors'):
        raise ValueError('Benchmark must pass correctness and provenance gates')
    samples = report['samples']
    if not samples or any(sample.get('outputs_equal') is not True for sample in samples):
        raise ValueError('Every measured/warmup pair must have equal outputs')
    grouped = {}
    for sample in samples:
        if sample['warmup']:
            continue
        for engine in ('artifact', 'reference'):
            grouped.setdefault((sample['prompt'], engine), []).append(sample['runs'][engine])
    if not grouped:
        raise ValueError('No measured samples')
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    grouped = measurements(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'bml-paired-inference-v1'
    prompts = list(dict.fromkeys(key[0] for key in grouped))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [('prefill_ms', 'Prefill (ms)', 1), ('decode_ms', 'Decode (ms)', 1),
              ('wall_ms', 'Total process time (ms)', 1), ('max_rss_kib', 'Peak RSS (MiB)', 1 / 1024)]
    for ax, (metric, title, scale) in zip(axes.flat, panels):
        for engine, offset, color, label in [('artifact', -.2, '#2266aa', 'BML native snapshot'),
                                             ('reference', .2, '#d27824', 'llama.cpp reference')]:
            values = [[sample[metric] * scale for sample in grouped[(prompt, engine)]] for prompt in prompts]
            means = [statistics.mean(v) for v in values]
            errors = [statistics.pstdev(v) for v in values]
            positions = [i + offset for i in range(len(prompts))]
            ax.bar(positions, means, width=.36, yerr=errors, capsize=4, color=color, label=label)
            for position, mean, points in zip(positions, means, values):
                ax.scatter([position] * len(points), points, s=12, color='#252525', zorder=3)
                ax.annotate(f'{mean:.1f}', (position, max(points)), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(title + ' — lower is better')
        ax.set_xticks(range(len(prompts)), prompts)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.16)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    date = datetime.datetime.fromtimestamp(report['finished_unix'], datetime.timezone.utc).date()
    fig.suptitle(f"{Path(report['model']['path']).name} · paired inference snapshot · {date}", fontsize=15)
    protocol = report['protocol']
    fig.text(.5, .02, f"{protocol['cpu_threads_per_engine']} CPU thread · {protocol['predict']} requested tokens · {protocol['measured_repetitions_per_prompt_per_engine']} measured pairs/prompt · mean ± population SD\n"
             'Dots: raw samples. Warmups excluded. Wall time includes startup/integrity checks.\n'
             'Historical artifact hashes identify this snapshot; it does not measure subsequent source changes.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ('svg', 'png'):
        path = args.output.with_suffix('.' + extension)
        fig.savefig(path, dpi=160, metadata={'Date': None} if extension == 'svg' else None)
        outputs.append(path)
    manifest = {
        'source_report': str(args.report),
        'source_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
        'artifact_sha256': report['artifact']['sha256'],
        'reference_sha256': report['reference']['sha256'],
        'matplotlib': matplotlib.__version__,
        'scope': 'Historical paired snapshot, not current-tree performance or synthetic token equivalents',
        'outputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
