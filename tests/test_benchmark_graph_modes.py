"""Small executable paired-run and failure-record contracts; no private data."""
import importlib.util
import json
from pathlib import Path
import signal

import numpy as np
import pytest


@pytest.fixture
def runner():
    pytest.importorskip('torch')
    if not hasattr(signal, 'setitimer'):
        pytest.skip('Bounded research runner requires Unix interval timers')
    spec = importlib.util.spec_from_file_location('graph_modes_runner',
        Path(__file__).resolve().parents[1] / 'examples/benchmark_graph_modes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def input_path(tmp_path):
    path = tmp_path / 'synthetic.npz'
    np.savez(path, features=np.random.default_rng(35).normal(size=(12, 3)),
             labels=np.array(['A'] * 8 + ['B'] * 4), cell_ids=np.array([f'cell{i}' for i in range(12)]))
    return path


def test_paired_runner_shares_actual_sigma_and_initial_arrays(runner, input_path, tmp_path):
    output = tmp_path / 'output'
    summary = runner.main([str(input_path), '--output', str(output), '--epochs', '2',
                           '--seeds', '0', '--neighbors', '3', '--ks', '2', '3', '--wall-seconds', '30'])
    assert summary['status'] == 'completed'
    assert summary['planned_fits'] == len(summary['runs']) == 2
    a, b = summary['runs']
    assert a['sigma'] == b['sigma']
    assert a['initial_sha256'] == b['initial_sha256']
    assert a['fit_initializer'] == b['fit_initializer'] == 'provided'
    assert a['n_epochs'] == b['n_epochs'] == 2
    assert all(0 <= r['best_epoch'] <= 2 for r in summary['runs'])
    assert summary['comparisons'][0]['comparison']['cell_order_verified']
    assert (output / 'synthetic/comparison_k2.png').stat().st_size > 0
    assert (output / 'synthetic/comparison_k3.png').stat().st_size > 0
    with np.load(output / 'synthetic/seed0.npz', allow_pickle=False) as data:
        assert data['mutual_loss'].shape == data['union_loss'].shape == (2,)
        assert data['initial'].shape == data['mutual'].shape == data['union'].shape == (12, 2)
    stored = json.loads((output / 'summary.json').read_text())
    assert stored['status'] == 'completed'
    assert not list(output.glob('*.tmp'))
    with pytest.raises(SystemExit):
        runner.main([str(input_path), '--output', str(output)])


def test_timeout_keeps_failed_active_run_and_restores_timer(runner, input_path, tmp_path, monkeypatch):
    output = tmp_path / 'timeout'
    old_handler = signal.getsignal(signal.SIGALRM)
    def fail(*args, **kwargs):
        raise TimeoutError('forced timeout')
    monkeypatch.setattr(runner, 'fit_disk_embedding', fail)
    with pytest.raises(TimeoutError):
        runner.main([str(input_path), '--output', str(output), '--seeds', '0', '--neighbors', '3', '--ks', '2'])
    stored = json.loads((output / 'summary.json').read_text())
    assert stored['status'] == 'timeout'
    assert stored['active']['graph_mode'] == 'mutual'
    assert stored['active']['epoch'] == 0
    assert stored['runs'] == []
    assert signal.getsignal(signal.SIGALRM) == old_handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize('failure', [False, True])
def test_terminal_status_is_updated_only_after_timer_is_disarmed(runner, input_path, tmp_path, monkeypatch, failure):
    events = []
    class CheckedStatus(dict):
        def update(self, *args, **kwargs):
            if kwargs.get('status') in ('completed', 'failed', 'timeout'):
                assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
                events.append(kwargs['status'])
            return super().update(*args, **kwargs)
    monkeypatch.setattr(runner, 'dict', CheckedStatus, raising=False)
    if failure:
        def fail(*args, **kwargs):
            raise ValueError('forced failure while timer is armed')
        monkeypatch.setattr(runner, 'fit_disk_embedding', fail)
    arguments = [str(input_path), '--output', str(tmp_path / 'terminal'), '--epochs', '1',
                 '--neighbors', '3', '--ks', '2', '--seeds', '0', '--wall-seconds', '30']
    if failure:
        with pytest.raises(ValueError, match='forced failure'):
            runner.main(arguments)
        assert events == ['failed']
    else:
        assert runner.main(arguments)['status'] == 'completed'
        assert events == ['completed']


def test_invalid_evaluation_labels_fail_before_any_fit(runner, tmp_path, monkeypatch):
    path = tmp_path / 'missing.npz'
    np.savez(path, features=np.ones((5, 2)), labels=np.array(['A', 'A', '', 'B', 'B']),
             cell_ids=np.array([str(i) for i in range(5)]))
    output = tmp_path / 'failed'
    monkeypatch.setattr(runner, 'fit_disk_embedding', lambda *a, **kw: pytest.fail('Fitting must not run'))
    with pytest.raises(ValueError, match='empty'):
        runner.main([str(path), '--output', str(output), '--ks', '1'])
    stored = json.loads((output / 'summary.json').read_text())
    assert stored['status'] == 'failed'
    assert not stored['runs']
