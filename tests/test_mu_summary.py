"""Plot/report collection must not turn missing or mismatched jobs into successes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sc_poincare.evaluation import evaluate_neighborhoods, _array_hash
from sc_poincare.visibility import evaluate_visibility


def load():
    spec=importlib.util.spec_from_file_location('mu_summary',Path(__file__).resolve().parents[1]/'examples/summarize_mu_scaleup.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def study(root,status='completed',stage='small',numeric_labels=False):
    (root/f'inputs/{stage}').mkdir(parents=True)
    (root/f'{stage}/toy_mu_fixed_s0').mkdir(parents=True)
    features=np.arange(33.)[:,None]
    points=np.column_stack([np.linspace(0,.3,33),np.zeros(33)])
    labels=np.array([0.]*18+[1.]*15) if numeric_labels else np.array(['A']*18+['B']*15);ids=np.array([f'c{i}' for i in range(33)])
    np.savez(root/f'inputs/{stage}/toy.npz',features=features,labels=labels,cell_ids=ids)
    np.savez(root/f'{stage}/toy_mu_fixed_s0/result.npz',coordinates=points)
    e=evaluate_neighborhoods(features,points,labels,cell_ids=ids)
    metadata=dict(evaluation=e.summary,geometry='poincare',n_cells=33,n_epochs=2,best_epoch=2,
        fitting_seconds=.01,preparation_seconds=.01,process_peak_rss_bytes=123,
        initial_disk_sha256=_array_hash(points),initial_tangent_sha256=_array_hash(points),
        visibility=evaluate_visibility(e),expression=None)
    entry=dict(status='completed',spec=dict(id='toy_mu_fixed_s0',dataset='toy',method='mu_fixed',seed=0),result=metadata)
    (root/f'{stage}/summary.json').write_text(json.dumps(dict(status=status,elapsed_seconds=.1,
        observed_peak_aggregate_rss_bytes=123,jobs=[entry])))
    (root/'study_input_manifest.json').write_text(json.dumps(
        dict(toy=dict(root_index=5),small={}) if stage=='full' else dict(small={'toy':{}})))
    return points


def test_summary_preserves_missing_controls_and_writes_native_separate_axes(tmp_path):
    root=tmp_path/'study';study(root)
    out=tmp_path/'report'
    r=load().main(['--study',str(root),'--output',str(out)])
    assert len(r['scores'])==1 and r['scores'][0]['status']=='completed'
    assert r['views']['small_toy_poincare_raw']['rfa']['status']=='unavailable'
    assert r['views']['small_toy_euclidean_native']['euclidean_fixed']['status']=='unavailable'
    assert (out/'small_toy_poincare_raw.png').stat().st_size>0
    assert (out/'scores.csv').exists()


@pytest.mark.parametrize('numeric_labels',[False,True])
def test_full_stage_flat_manifest_writes_declared_root_view(tmp_path,monkeypatch,numeric_labels):
    root=tmp_path/'study';raw=study(root,stage='full',numeric_labels=numeric_labels)
    module=load();original=module._root_view;seen=[]
    def inspect_root(snapshot,index):
        assert index==5
        shifted,validation=original(snapshot,index)
        assert np.linalg.norm(shifted.coordinates[5])<1e-10
        np.testing.assert_array_equal(snapshot.coordinates,raw)
        seen.append(index)
        return shifted,validation
    monkeypatch.setattr(module,'_root_view',inspect_root)
    output=tmp_path/'report'
    report=module.main(['--study',str(root),'--output',str(output)])
    view=report['views']['full_toy_poincare_root']
    assert seen==[5] and view['mu_fixed']['status']=='root_centered'
    assert view['rfa']['status']=='unavailable'
    assert view['mu_fixed']['validation']['max_absolute_distance_error']<1e-10
    assert (output/'full_toy_poincare_raw.png').exists()
    assert (output/'full_toy_poincare_root.png').exists()
    with np.load(root/'full/toy_mu_fixed_s0/result.npz') as saved:
        np.testing.assert_array_equal(saved['coordinates'],raw)


def test_summary_refuses_running_stage_before_writing(tmp_path):
    root=tmp_path/'study';study(root,status='running')
    out=tmp_path/'report'
    with pytest.raises(ValueError,match='supervisor'):
        load().main(['--study',str(root),'--output',str(out)])
    assert not out.exists()


def test_failed_null_result_is_reported_without_using_partial_coordinates(tmp_path):
    root=tmp_path/'study';study(root)
    path=root/'small/summary.json';summary=json.loads(path.read_text())
    summary['status']='partial'
    summary['jobs'][0].update(status='failed',result=None,stop_reason='worker_failure')
    path.write_text(json.dumps(summary))
    r=load().main(['--study',str(root),'--output',str(tmp_path/'report')])
    assert r['scores'][0]['status']=='failed'
    assert r['scores'][0]['error']=='worker_failure'
    assert 'k15_purity' not in r['scores'][0]
    assert r['views']['small_toy_poincare_raw']['mu_fixed']['status']=='unavailable'


def test_root_view_keeps_intrinsic_distances_and_does_not_mutate_raw():
    module=load()
    features=np.arange(33.)[:,None];points=np.column_stack([np.linspace(0,.3,33),np.zeros(33)])
    e=evaluate_neighborhoods(features,points,['A']*33,cell_ids=[f'c{i}' for i in range(33)])
    shifted,validation=module._root_view(e,5)
    assert np.linalg.norm(shifted.coordinates[5])<1e-10
    assert validation['max_absolute_distance_error']<1e-10
    np.testing.assert_array_equal(e.coordinates,points)
