#!/usr/bin/env python3
"""Quality gates. Missing evidence and subprocess failures are errors."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'target/quality'


def command(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, **kwargs)


def diagnostics(path):
    seen, counts = set(), Counter()
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('reason') != 'compiler-message':
            continue
        msg = event['message']
        primary = next((s for s in msg['spans'] if s['is_primary']), None)
        if primary is None or msg['level'] not in ('warning', 'error'):
            continue
        code = (msg.get('code') or {}).get('code', 'unknown')
        key = ' | '.join([primary['file_name'], code, msg['message']])
        instance = (key, primary['line_start'])
        if instance not in seen:
            counts[key] += 1
            seen.add(instance)
    return counts


def lint():
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / 'clippy.jsonl'
    with path.open('w') as output:
        command(['cargo', 'clippy', '--workspace', '--all-targets', '--locked',
                 '--message-format=json', '--', '--cap-lints', 'warn'], stdout=output)
    current = diagnostics(path)
    baseline = Counter(json.loads((ROOT / 'quality/clippy-baseline.json').read_text())['diagnostics'])
    new = current - baseline
    print(f'Static diagnostics: {sum(current.values())}; baseline: {sum(baseline.values())}; new: {sum(new.values())}')
    (REPORTS / 'clippy-summary.json').write_text(json.dumps({'total': sum(current.values()), 'new': dict(new)}, indent=2)+'\n')
    if new:
        raise SystemExit('New diagnostics:\n' + '\n'.join(f'{n}: {key}' for key, n in new.items()))


def formatting(base, staged):
    args = ['git', 'diff', '--name-only', '--diff-filter=ACMR'] + (['--cached'] if staged else [base])
    files = command(args, capture_output=True).stdout.splitlines()
    if not staged:
        files += command(['git', 'ls-files', '--others', '--exclude-standard'], capture_output=True).stdout.splitlines()
    bad = []
    for filename in sorted(set(files)):
        if not filename.endswith('.rs'):
            continue
        original = command(['git', 'show', ':'+filename], capture_output=True).stdout if staged else (ROOT / filename).read_text()
        formatted = command(['rustfmt', '--edition', '2021', '--config', 'skip_children=true', '--emit', 'stdout'], input=original, capture_output=True).stdout
        if original != formatted:
            bad.append(filename)
    if bad:
        raise SystemExit('Run rustfmt on:\n' + '\n'.join(bad))
    print('Changed Rust files: format OK')


def trace():
    scenarios = re.findall(r'@(C\d+)', (ROOT / 'quality/contracts.feature').read_text())
    mapping = json.loads((ROOT / 'quality/traceability.json').read_text())
    if len(scenarios) != len(set(scenarios)) or set(scenarios) != set(mapping):
        raise SystemExit('Scenarios must be unique and mapped exactly once')
    discovered = {}
    missing = {}
    for scenario, spec in mapping.items():
        suite = (spec['package'], spec['target'])
        if suite not in discovered:
            output = command(['cargo', 'test', '-p', suite[0], '--test', suite[1], '--locked', '--', '--list'], capture_output=True)
            discovered[suite] = {line.removesuffix(': test') for line in output.stdout.splitlines() if line.endswith(': test')}
        if not spec['tests'] or not set(spec['tests']) <= discovered[suite]:
            missing[scenario] = spec
    if missing:
        raise SystemExit('Unregistered scenario tests: '+str(missing))
    REPORTS.mkdir(parents=True, exist_ok=True)
    suites = {'::'.join(suite): sorted(tests) for suite, tests in discovered.items()}
    (REPORTS / 'traceability.json').write_text(json.dumps({'scenarios': len(scenarios), 'mapped': len(mapping), 'tests': suites}, indent=2)+'\n')
    print(f'Scenarios registered in Cargo: {len(mapping)}/{len(scenarios)} (execution checked by cargo test)')


def coverage(path):
    totals = json.loads(Path(path).read_text())['data'][0]['totals']
    baseline = json.loads((ROOT / 'quality/coverage-baseline.json').read_text())
    measured = {m: totals[m]['percent'] for m in ['lines', 'functions']}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'coverage-summary.json').write_text(json.dumps(measured, indent=2)+'\n')
    print('Coverage: '+json.dumps(measured))
    for metric, percentage in measured.items():
        if percentage < baseline[metric]:
            raise SystemExit(f'{metric} coverage {percentage:.2f}% < floor {baseline[metric]}%')


def mutations(path):
    data = json.loads(Path(path).read_text())
    outcomes = [o for o in data['outcomes'] if isinstance(o['scenario'], dict) and 'Mutant' in o['scenario']]
    if not data.get('end_time') or len(outcomes) != data['total_mutants']:
        raise SystemExit('Incomplete mutation campaign')
    counts = Counter(o['summary'] for o in outcomes)
    if set(counts) - {'CaughtMutant', 'MissedMutant', 'Timeout', 'Unviable'}:
        raise SystemExit('Unknown mutation outcome')
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'mutation-summary.json').write_text(json.dumps(counts, indent=2)+'\n')
    print('Mutation outcomes: '+json.dumps(counts))
    if not outcomes:
        print('Mutation score: N/A (no generated mutations)')
        return
    baselines = [o for o in data['outcomes'] if o['scenario'] == 'Baseline']
    if len(baselines) != 1 or baselines[0]['summary'] != 'Success':
        raise SystemExit('Mutation baseline must pass')
    caught = counts['CaughtMutant']
    denominator = caught + counts['MissedMutant'] + counts['Timeout']
    print(f'Mutation score: {100*caught/denominator:.2f}%' if denominator else 'Mutation score: N/A (no viable mutations)')
    if counts['MissedMutant'] or counts['Timeout']:
        raise SystemExit('Mutation gate failed: inspect outcomes and logs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gate', choices=['format', 'lint', 'trace', 'coverage', 'mutations'])
    parser.add_argument('--base', default='origin/master')
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.gate == 'format': formatting(args.base, args.staged)
    elif args.gate == 'lint': lint()
    elif args.gate == 'trace': trace()
    elif args.gate == 'coverage': coverage(args.report)
    elif args.gate == 'mutations': mutations(args.report)


if __name__ == '__main__':
    main()
