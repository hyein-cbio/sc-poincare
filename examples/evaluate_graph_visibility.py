"""Post-process paired graph studies without fitting or selecting new conditions.

Supply benchmark_graph_modes.py output directories. Completed pairs are audited
against their saved input/coordinate hashes; failed active cases remain visible.
Seed 0 is the predetermined display representative. No source data are copied.

    python examples/evaluate_graph_visibility.py data/paired-study \
        --output data/new-visibility-report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np

from sc_poincare import visibility as visibility_module, evaluation as evaluation_module, _disk
from sc_poincare.evaluation import evaluate_neighborhoods, compare_neighborhoods
from sc_poincare.visibility import evaluate_visibility, compare_visibility, plot_visibility_pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studies', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output directory must not exist.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy
    from PIL import Image, ImageDraw, __version__ as pillow_version

    result = dict(policy=dict(canvas_pixels=600, marker_diameter=4., representative_seed=0),
                  source_sha256={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                 for m in (visibility_module, evaluation_module, _disk)},
                  cli_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                                matplotlib=matplotlib.__version__, pillow=pillow_version),
                  cases={}, failures=[], sources=[])
    figures = {}
    try:
        for study in args.studies:
            summary_path = study / 'summary.json'
            raw = summary_path.read_bytes()
            summary = json.loads(raw)
            result['sources'].append(dict(summary_sha256=hashlib.sha256(raw).hexdigest(),
                status=summary['status'], completed_fits=len(summary['runs']), planned_fits=summary['planned_fits']))
            if summary.get('failures'):
                result['failures'].extend(summary['failures'])
            elif summary['status'] != 'completed':
                result['failures'].append(dict(active=summary.get('active'), error=summary.get('error'),
                                                status=summary['status']))
            for pair in summary['comparisons']:
                name, seed = pair['dataset'], pair['seed']
                if (not isinstance(name, str) or not name or name in ('.', '..')
                        or '/' in name or '\\' in name
                        or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
                    raise ValueError('Dataset must be a safe path component and seed a nonnegative integer.')
                if any(row['seed'] == seed for row in result['cases'].get(name, [])):
                    raise ValueError(f'Duplicate paired result: {name} seed {seed}.')
                with np.load(study / name / 'input.npz', allow_pickle=False) as data:
                    features, labels, ids = data['features'], data['labels'], data['cell_ids']
                with np.load(study / name / f'seed{seed}.npz', allow_pickle=False) as data:
                    points = {mode: data[mode] for mode in ('mutual', 'union')}
                evaluated = {}
                for mode in ('mutual', 'union'):
                    reference = pair[mode]
                    global_policy = reference['global_feature_rank']
                    evaluated[mode] = evaluate_neighborhoods(features, points[mode], labels, cell_ids=ids,
                        ks=reference['ks'], global_n_anchors=global_policy['n_anchors'],
                        global_random_state=global_policy['seed'])
                    if evaluated[mode].summary['provenance'] != reference['provenance']:
                        raise ValueError(f'Input/coordinate hash mismatch: {name} seed {seed} {mode}.')
                structural = compare_neighborhoods(evaluated['mutual'], evaluated['union'])
                before = evaluate_visibility(evaluated['mutual'])
                after = evaluate_visibility(evaluated['union'])
                row = dict(seed=seed, mutual=before, union=after, delta=compare_visibility(before, after),
                           structural_delta=structural,
                           structural_mutual=evaluated['mutual'].summary,
                           structural_union=evaluated['union'].summary)
                result['cases'].setdefault(name, []).append(row)
                if seed == 0:
                    figures[name] = plot_visibility_pair(evaluated['mutual'], evaluated['union'],
                        title=f'{name}: fixed 600px disk / 4px markers, seed 0')
        if not result['cases'] and not result['failures']:
            raise ValueError('No completed pairs or recorded failures to evaluate.')
        if set(figures) != set(result['cases']):
            raise ValueError('Seed 0 is required for the predetermined representative plots.')
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'visibility_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        pictures = []
        for name, figure in figures.items():
            path = args.output / f'{name}_visibility.png'
            figure.savefig(path, dpi=figure.dpi)
            with Image.open(path) as picture:
                resampling = getattr(Image, 'Resampling', Image)
                pictures.append(picture.convert('RGB').resize((1200, 656), resampling.LANCZOS))
        for failure in result['failures']:
            image = Image.new('RGB', (1200, 180), 'white')
            draw = ImageDraw.Draw(image)
            active, error = failure.get('active') or {}, failure.get('error') or {}
            text = (f"NO PAIRED RESULT: {active.get('dataset', 'unknown')}\n"
                    + textwrap.fill(f"{error.get('type', failure['status'])}: {error.get('message', '')}", width=135))
            draw.multiline_text((25, 25), text, fill='black', spacing=10)
            pictures.append(image)
        overview = Image.new('RGB', (1200, sum(p.height for p in pictures)), 'white')
        offset = 0
        for picture in pictures:
            overview.paste(picture, (0, offset)); offset += picture.height
        overview.save(args.output / 'visibility_overview.png')
    finally:
        for figure in figures.values():
            plt.close(figure)
    return result


if __name__ == '__main__':
    main()
