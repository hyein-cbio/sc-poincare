"""Legacy boundaries and closed-research execution policy, without fitting archives."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sc_poincare import _fitting
from sc_poincare._legacy_numerics import _row_softmax_neg_distance, _sym_kl
from sc_poincare._objectives import _numpy_alignment_loss
from sc_poincare._research_policy import ensure_active_paths, CLOSED_RESEARCH_MARKER

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location('containment_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def closed(tmp_path):
    folder = tmp_path / 'closed'
    folder.mkdir()
    (folder / CLOSED_RESEARCH_MARKER).write_text(json.dumps(dict(status='closed')))
    return folder


def test_legacy_numpy_wrappers_are_preserved_but_absent_from_active_fitter():
    assert not hasattr(_fitting, '_row_softmax_neg_distance')
    assert not hasattr(_fitting, '_sym_kl')
    distance = np.array([[0., .2, .8], [.2, 0., .5], [.8, .5, 0.]])
    target = np.array([[.5, .3, .2], [.2, .5, .3], [.1, .2, .7]])
    q = _row_softmax_neg_distance(distance, 2.)
    np.testing.assert_allclose(q.sum(1), 1.)
    assert _sym_kl(target, q) == pytest.approx(_numpy_alignment_loss(target, distance, 2.))


def test_production_modules_do_not_import_retired_numpy_wrappers_or_historical_fitters():
    for path in (ROOT/'src/sc_poincare').glob('*.py'):
        if path.name == '_legacy_numerics.py': continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                assert node.name not in ('_row_softmax_neg_distance', '_sym_kl'), path
            if isinstance(node, ast.ImportFrom):
                assert '_legacy_numerics' not in (node.module or ''), path
                assert 'historical' not in (node.module or ''), path
            if isinstance(node, ast.Import):
                assert all('_legacy_numerics' not in a.name and 'historical' not in a.name for a in node.names), path


def test_legacy_export_block_is_explicit_and_display_api_remains_available():
    from sc_poincare import tl
    text = (ROOT/'src/sc_poincare/tl.py').read_text()
    assert text.count('BEGIN LEGACY DISPLAY EXPORTS') == text.count('END LEGACY DISPLAY EXPORTS') == 1
    for name in ('poincare_fan', 'poincare_jitter', 'poincare_repel'):
        assert getattr(tl, name).__module__ == 'sc_poincare._legacy_display'


def test_historical_examples_have_no_root_redirects():
    assert not (ROOT/'tests/test_experimental_geodesic.py').exists()
    assert (ROOT/'examples/historical/tests/test_experimental_geodesic.py').exists()
    for name in ['experimental_geodesic_fit.py', 'run_selected_conditions.py']:
        assert not (ROOT/'examples'/name).exists()
        assert (ROOT/'examples/historical'/name).exists()


def test_historical_cli_requires_acknowledgement_before_reading_input(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    module = load('examples/historical/run_selected_conditions.py')
    monkeypatch.setattr(module.pd, 'read_csv', lambda *a, **k: pytest.fail('Must stop before input or fitting'))
    out = tmp_path/'out.npz'
    monkeypatch.setattr(sys, 'argv', ['historical', 'ToggleSwitch', str(tmp_path/'missing.csv'), '--output', str(out)])
    with pytest.raises(SystemExit): module.main()
    assert not out.exists()


def test_historical_cli_can_run_only_explicit_active_tiny_reproduction(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pd = pytest.importorskip('pandas')
    module = load('examples/historical/run_selected_conditions.py')
    source = tmp_path/'tiny.csv'
    frame = pd.DataFrame(np.random.default_rng(4).normal(size=(8, 3)))
    frame['label'] = ['A']*4 + ['B']*4
    frame.to_csv(source, index=False)
    monkeypatch.setitem(module.RECIPES, 'ToggleSwitch', dict(k=3, sigma=1., pca=0, gamma=.5, lr=.02, stages=((2, 1.),)))
    out = tmp_path/'out.npz'
    monkeypatch.setattr(sys, 'argv', ['historical', '--acknowledge-historical', 'ToggleSwitch', str(source), '--output', str(out)])
    module.main()
    assert out.exists() and out.with_suffix('.json').exists()
    assert json.loads(out.with_suffix('.json').read_text())['parameters']['optimizer'] == 'spatial_coordinate_adam'


def test_closed_paths_and_symlinks_are_denied_but_sibling_inputs_work(tmp_path):
    archive = closed(tmp_path)
    source = archive/'nested'/'input.npy'
    for path in [archive, source, archive/'new-output', archive/'state.pt']:
        with pytest.raises(ValueError, match='Closed research tree'): ensure_active_paths(path)
    link = tmp_path/'link'
    link.symlink_to(archive, target_is_directory=True)
    with pytest.raises(ValueError, match='Closed research tree'): ensure_active_paths(link/'input.npy')
    ensure_active_paths(tmp_path/'active'/'input.npy', None)


@pytest.mark.parametrize('relative,options', [
    ('examples/run_graph_stress.py', ['--input', '{input}', '--output', '{output}']),
    ('examples/run_mu_study.py', ['--input', '{input}', '--output', '{output}', '--method', 'mu_fixed']),
    ('examples/benchmark_graph_modes.py', ['{input}', '--output', '{output}']),
    ('examples/benchmark_small.py', ['{input}', '--output', '{output}']),
    ('examples/benchmark_objectives.py', ['{input}', '--output', '{output}']),
    ('examples/compare_spectral_init.py', ['{input}', '--output', '{output}']),
    ('examples/inspect_rare_populations.py', ['{input}', '--output', '{output}']),
    ('examples/inspect_root_pseudotime.py', ['{input}', '--output', '{output}']),
])
def test_training_clis_reject_closed_input_before_output(relative, options, tmp_path, monkeypatch):
    pytest.importorskip('threadpoolctl')
    archive = closed(tmp_path);source = archive/'input.npy';output = tmp_path/'out'
    module = load(relative)
    args = [x.format(input=source, output=output) for x in options]
    with pytest.raises(ValueError, match='Closed research tree'):
        if relative.endswith(('benchmark_small.py', 'benchmark_objectives.py',
                              'compare_spectral_init.py', 'inspect_rare_populations.py',
                              'inspect_root_pseudotime.py')):
            monkeypatch.setattr(sys, 'argv', ['runner', *args]);module.main()
        else: module.main(args)
    assert not output.exists()


@pytest.mark.parametrize('script', [
    'compare_spectral_init', 'inspect_rare_populations', 'inspect_root_pseudotime',
    'inspect_pbmc3k_frozen', 'inspect_pbmc3k_graph_stress',
])
@pytest.mark.parametrize('symlink', [False, True])
def test_research_fitting_clis_reject_closed_output_before_loading(script, symlink, tmp_path, monkeypatch):
    from types import SimpleNamespace
    archive = closed(tmp_path)
    output = archive / 'result.json'
    if symlink:
        link = tmp_path / 'result.json'
        link.symlink_to(output)
        output = link
    monkeypatch.setitem(sys.modules, 'scanpy', SimpleNamespace())
    module = load(f'examples/{script}.py')
    options = [] if 'pbmc3k' in script else [str(tmp_path / 'missing.csv')]
    monkeypatch.setattr(sys, 'argv', [script, *options, '--output', str(output)])
    with pytest.raises(ValueError, match='Closed research tree'):
        module.main()
    assert not (archive / 'result.json').exists()


@pytest.mark.parametrize('script', ['inspect_pbmc3k_frozen', 'inspect_pbmc3k_graph_stress'])
@pytest.mark.parametrize('closed_cache', ['directory', 'pbmc3k_raw.h5ad', 'pbmc3k_processed.h5ad'])
def test_pbmc_fits_reject_closed_dataset_cache(script, closed_cache, tmp_path, monkeypatch):
    from types import SimpleNamespace
    archive = closed(tmp_path)
    datasetdir = archive
    if closed_cache != 'directory':
        datasetdir = tmp_path / 'active'
        datasetdir.mkdir()
        (datasetdir / closed_cache).symlink_to(archive / closed_cache)

    def deny_read():
        pytest.fail('Must check dataset paths before reading or downloading')

    monkeypatch.setitem(sys.modules, 'scanpy', SimpleNamespace(
        settings=SimpleNamespace(datasetdir=datasetdir),
        datasets=SimpleNamespace(pbmc3k=deny_read, pbmc3k_processed=deny_read),
    ))
    module = load(f'examples/{script}.py')
    output = tmp_path / 'out.json'
    monkeypatch.setattr(sys, 'argv', [script, '--output', str(output)])
    with pytest.raises(ValueError, match='Closed research tree'):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize('relative,filename', [
    ('examples/benchmark_small.py', 'report_seed0.npz'),
    ('examples/benchmark_objectives.py', 'report_symmetric_kl_seed0.npz'),
])
def test_benchmarks_check_derived_output_symlinks_before_loading(relative, filename, tmp_path, monkeypatch):
    module = load(relative);archive = closed(tmp_path)
    (tmp_path/filename).symlink_to(archive/'stored-result.npz')
    monkeypatch.setattr(module.pd, 'read_csv', lambda *a, **k: pytest.fail('Must reject before loading/fitting'))
    monkeypatch.setattr(sys, 'argv', ['benchmark', str(tmp_path/'input.csv'), '--output', str(tmp_path/'report.json'), '--seeds', '0'])
    with pytest.raises(ValueError, match='Closed research tree'): module.main()
    assert not (tmp_path/'report.json').exists()


def test_supervisor_checks_all_nested_job_inputs_before_launch(tmp_path):
    pytest.importorskip('psutil')
    module = load('examples/benchmark_mu_scaleup.py')
    archive = closed(tmp_path);jobs = tmp_path/'jobs.json';out = tmp_path/'out'
    jobs.write_text(json.dumps([dict(id='closed', input=str(archive/'input.npz'), method='mu_fixed', epochs=1, seed=0, threads=1)]))
    with pytest.raises(ValueError, match='Closed research tree'): module.main(['--jobs', str(jobs), '--output', str(out)])
    assert not out.exists()


def test_graph_cli_blocks_closed_resume_and_closed_output(tmp_path):
    module = load('examples/run_graph_stress.py');archive = closed(tmp_path)
    for args in [ ['--input', str(tmp_path/'active.npy'), '--resume', str(archive/'checkpoint.pt'), '--output', str(tmp_path/'out')],
                  ['--input', str(tmp_path/'active.npy'), '--output', str(archive/'new-out')] ]:
        with pytest.raises(ValueError, match='Closed research tree'): module.main(args)


def test_historical_ack_does_not_reopen_closed_inputs(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    module = load('examples/historical/run_selected_conditions.py')
    archive = closed(tmp_path);output = tmp_path/'out.npz'
    monkeypatch.setattr(sys, 'argv', ['historical', '--acknowledge-historical', 'ToggleSwitch', str(archive/'input.csv'), '--output', str(output)])
    with pytest.raises(ValueError, match='Closed research tree'): module.main()
    assert not output.exists()


def test_demo_does_not_fit_into_closed_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'scanpy', SimpleNamespace(settings=SimpleNamespace(figdir=tmp_path/'active-figures')))
    module = load('examples/run_scanpy_demo.py');archive = closed(tmp_path)
    monkeypatch.setattr(module, 'load_dataset', lambda *a: pytest.fail('Must reject before loading/fitting'))
    with pytest.raises(ValueError, match='Closed research tree'):
        module.main(['--outdir', str(archive/'new-output')])


def test_demo_checks_scanpy_secondary_figure_directory_before_fitting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    archive = closed(tmp_path)
    monkeypatch.setitem(sys.modules, 'scanpy', SimpleNamespace(settings=SimpleNamespace(figdir=archive/'figures')))
    module = load('examples/run_scanpy_demo.py')
    monkeypatch.setattr(module, 'load_dataset', lambda *a: pytest.fail('Must reject before fitting'))
    output = tmp_path/'active-output'
    with pytest.raises(ValueError, match='Closed research tree'): module.main(['--outdir', str(output)])
    assert not output.exists()


def test_closed_source_can_still_be_read_by_postfit_evaluation(tmp_path):
    from sc_poincare.evaluation import evaluate_neighborhoods
    archive = closed(tmp_path);source = archive/'input.npz'
    x = np.arange(5.)[:, None];p = np.column_stack([np.arange(5.)/20, np.zeros(5)])
    np.savez(source, features=x, coordinates=p, labels=['A','A','B','B','B'])
    with np.load(source) as data: result = evaluate_neighborhoods(data['features'], data['coordinates'], data['labels'], ks=[1])
    assert result.summary['n_cells'] == 5
