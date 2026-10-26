"""Deprecated display APIs are isolated; training initialization jitter is not."""
from pathlib import Path
import ast
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sc_poincare._disk import confine_to_open_disk, polar_angles
from sc_poincare._legacy_display import (
    DisplayLayoutDeprecationWarning,
    _repel_points as repel_points,
    _jitter_angles as jitter_angles,
    _fan_dendrogram_angles as fan_dendrogram_angles,
)
from sc_poincare import tl


def test_zero_jitter_is_exactly_the_embedding():
    rng = np.random.default_rng(6)
    pts = confine_to_open_disk(rng.normal(scale=0.3, size=(10, 2)))
    out = jitter_angles(pts, jitter=0.0)
    assert out is not pts
    assert np.array_equal(out, pts)


def test_jitter_changes_only_angle_and_stays_inside_the_window():
    pts = np.array([[0.0, 0.0], [0.6, 0.0], [0.0, 0.4]])
    out = jitter_angles(pts, jitter=0.1, random_state=1)
    assert np.array_equal(out[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(pts, axis=1))
    original = polar_angles(pts[1:])
    moved = polar_angles(out[1:])
    delta = np.abs(np.arctan2(np.sin(moved - original), np.cos(moved - original)))
    assert np.all(delta <= 0.1 + 1e-12)
    assert np.any(delta > 0.0)


def test_zero_spread_is_exactly_the_poincare_embedding():
    rng = np.random.default_rng(3)
    pts = confine_to_open_disk(rng.normal(scale=0.4, size=(30, 2)))
    groups = np.array(["a", "b", "c"] * 10)
    fanned = fan_dendrogram_angles(pts, groups, spread=0.0)
    assert fanned is not pts
    assert np.array_equal(fanned, pts)


def test_full_spread_keeps_radii_and_separates_overlapping_branches():
    rng = np.random.default_rng(4)
    branch_a = rng.normal(scale=0.02, size=(40, 2)) + np.array([0.55, 0.02])
    branch_b = rng.normal(scale=0.02, size=(40, 2)) + np.array([0.62, 0.0])
    pts = confine_to_open_disk(np.vstack([branch_a, branch_b]))
    groups = np.array(["a"] * 40 + ["b"] * 40)
    before = polar_angles(pts)
    fanned = fan_dendrogram_angles(pts, groups, spread=1.0)
    assert np.allclose(np.linalg.norm(fanned, axis=1), np.linalg.norm(pts, axis=1))
    after = polar_angles(fanned)
    before_gap = abs(before[:40].mean() - before[40:].mean())
    after_gap = abs(after[:40].mean() - after[40:].mean())
    after_gap = min(after_gap, 2 * np.pi - after_gap)
    before_gap = min(before_gap, 2 * np.pi - before_gap)
    assert after_gap > before_gap


def test_zero_repel_is_exactly_the_embedding():
    rng = np.random.default_rng(5)
    pts = confine_to_open_disk(rng.normal(scale=0.3, size=(12, 2)))
    out = repel_points(pts, repel=0.0)
    assert out is not pts
    assert np.array_equal(out, pts)


def test_repel_separates_stacked_points_along_the_same_ray():
    pts = np.array([[0.0, 0.0], [0.4, 0.2], [0.4, 0.2], [0.4, 0.2]])
    out = repel_points(pts, repel=1.0)
    assert np.array_equal(out[0], [0.0, 0.0])
    original = pts[1]
    for row in out[1:]:
        assert abs(original[0] * row[1] - original[1] * row[0]) < 1e-9
        assert np.dot(original, row) > 0.0
    gaps = np.linalg.norm(out[1:, None, :] - out[None, 1:, :], axis=2)
    off = gaps[np.triu_indices(3, k=1)]
    assert np.min(off) > 0.02


def _adata():
    ad = pytest.importorskip('anndata')
    x = ad.AnnData(np.zeros((5, 2)))
    x.obsm['X_poincare'] = np.array([[0., 0.], [.2, .1], [.25, .15], [-.3, .2], [-.35, .25]])
    x.obs['group'] = ['root', 'A', 'A', 'B', 'B']
    return x


def test_public_jitter_warns_once_at_caller_and_preserves_training_coordinates():
    x = _adata(); original = x.obsm['X_poincare'].copy()
    with pytest.warns(DisplayLayoutDeprecationWarning, match='poincare_jitter.*deprecated') as recorded:
        out = tl.poincare_jitter(x, jitter=.1, random_state=4)
    assert out is None and len(recorded) == 1
    assert Path(recorded[0].filename).resolve() == Path(__file__).resolve()
    np.testing.assert_array_equal(x.obsm['X_poincare'], original)
    np.testing.assert_array_equal(x.obsm['X_poincare_jitter'], jitter_angles(original, jitter=.1, random_state=4))


def test_public_repel_warns_once_and_preserves_raw_coordinates():
    x = _adata()
    x.obsm['X_poincare'][1:4] = [.4, .2]
    raw = x.obsm['X_poincare'].copy()
    with pytest.warns(DisplayLayoutDeprecationWarning, match='poincare_repel.*deprecated') as recorded:
        result = tl.poincare_repel(x, repel=1.)
    assert result is None and len(recorded) == 1
    assert Path(recorded[0].filename).resolve() == Path(__file__).resolve()
    np.testing.assert_array_equal(x.obsm['X_poincare'], raw)
    np.testing.assert_array_equal(x.obsm['X_poincare_repel'], repel_points(raw, repel=1.))
    assert not np.array_equal(x.obsm['X_poincare_repel'], raw)


def test_public_repel_keeps_custom_basis_key_and_copy_contract():
    x = _adata(); raw = x.obsm['X_poincare'].copy()
    x.obsm['X_other'] = raw.copy(); x.obsm['X_other'][2] = x.obsm['X_other'][1]
    with pytest.warns(DisplayLayoutDeprecationWarning, match='poincare_repel.*deprecated'):
        out = tl.poincare_repel(x, basis='X_other', key_added='X_legacy', repel=1., copy=True)
    assert out is not x and 'X_legacy' not in x.obsm
    np.testing.assert_array_equal(out.obsm['X_legacy'], repel_points(x.obsm['X_other'], repel=1.))
    np.testing.assert_array_equal(out.obsm['X_poincare'], raw)
    np.testing.assert_array_equal(out.obsm['X_other'], x.obsm['X_other'])


def test_public_fan_warns_and_keeps_copy_basis_and_output_contract():
    x = _adata(); original = x.obsm['X_poincare'].copy()
    with pytest.warns(DisplayLayoutDeprecationWarning, match='poincare_fan.*deprecated') as recorded:
        out = tl.poincare_fan(x, groupby='group', spread=.6, copy=True, key_added='X_legacy')
    assert len(recorded) == 1 and out is not x
    assert 'X_legacy' not in x.obsm
    np.testing.assert_array_equal(out.obsm['X_poincare'], original)
    np.testing.assert_array_equal(out.obsm['X_legacy'], fan_dendrogram_angles(original, x.obs['group'].to_numpy(), spread=.6))


@pytest.mark.parametrize('api,kwargs,key', [
    (tl.poincare_repel, dict(repel=0.), 'X_poincare_repel'),
    (tl.poincare_jitter, dict(jitter=0.), 'X_poincare_jitter'),
    (tl.poincare_fan, dict(groupby='group', spread=0.), 'X_poincare_fan'),
])
def test_identity_calls_are_also_deprecated_but_still_exact_copies(api, kwargs, key):
    x = _adata()
    with pytest.warns(DisplayLayoutDeprecationWarning) as recorded:
        api(x, **kwargs)
    assert len(recorded) == 1
    np.testing.assert_array_equal(x.obsm[key], x.obsm['X_poincare'])
    assert not np.shares_memory(x.obsm[key], x.obsm['X_poincare'])


def test_core_has_no_dependency_on_retirable_module():
    root = Path(__file__).resolve().parents[1] / 'src/sc_poincare'
    for path in root.glob('*.py'):
        if path.name in ('tl.py', '_legacy_display.py'):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert node.name not in {'repel_points', 'repel_embedding', 'jitter_embedding', 'fan_branches'}, path
            if isinstance(node, ast.ImportFrom):
                assert '_legacy_display' not in (node.module or ''), path
            elif isinstance(node, ast.Import):
                assert all('_legacy_display' not in a.name for a in node.names), path
    assert tl.poincare_jitter.__module__ == 'sc_poincare._legacy_display'
    assert tl.poincare_fan.__module__ == 'sc_poincare._legacy_display'
    assert tl.poincare_repel.__module__ == 'sc_poincare._legacy_display'


def test_future_deletion_needs_only_module_and_tl_block_removal(tmp_path):
    package = Path(__file__).resolve().parents[1] / 'src/sc_poincare'
    copy = tmp_path / 'sc_poincare'
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns('__pycache__'))
    (copy / '_legacy_display.py').unlink()
    path = copy / 'tl.py'
    path.write_text(path.read_text().split('# Deprecated display-only API:')[0])
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import anndata as ad
import sc_poincare as dr
from sc_poincare.experimental import GraphStressConfig, prepare_graph_stress
assert not any(hasattr(dr.tl, name) for name in ('poincare_jitter', 'poincare_fan', 'poincare_repel'))
features = np.random.default_rng(2).normal(size=(19, 3))
prepare_graph_stress(features, config=GraphStressConfig(n_neighbors=4))
x = ad.AnnData(features)
dr.tl.poincare_map(x, method='rfa', n_neighbors=4, n_epochs=1)
assert x.uns['poincare']['method'] == 'rfa'
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path)],
        capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr or result.stdout)
