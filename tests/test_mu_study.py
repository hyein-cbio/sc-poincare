"""Native expression controls and executable resource-bounded research jobs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sc_poincare.evaluation import evaluate_expression_neighborhoods


def load(name):
    path=Path(__file__).resolve().parents[1]/f'examples/{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_expression_mse_independent_reference_and_constant_gene():
    expression=np.array([[0.,2.,7.],[1.,4.,7.],[3.,8.,7.],[7.,16.,7.]])
    points=np.array([[0.,0.],[2.,0.],[8.,0.],[20.,0.]])
    original=expression.copy()
    r=evaluate_expression_neighborhoods(expression,points,ks=[1,2],geometry='euclidean',block_size=1)
    neighbors=np.array([[1,2],[0,2],[1,0],[2,1]])
    variance=expression[:,:2].var(axis=0)
    for k in [1,2]:
        prediction=expression[neighbors[:,:k],:2].mean(axis=1)
        expected=np.mean((prediction-expression[:,:2])**2/variance)
        assert r['normalized_mse'][str(k)]==pytest.approx(expected)
    assert r['n_genes_evaluated']==2 and r['n_zero_variance_genes']==1
    scaled=evaluate_expression_neighborhoods(expression*3+10,points,ks=[1,2],geometry='euclidean')
    assert scaled['normalized_mse']==pytest.approx(r['normalized_mse'])
    np.testing.assert_array_equal(expression,original)


@pytest.mark.parametrize('kwargs',[dict(ks=[0]),dict(ks=[4]),dict(ks=[]),dict(geometry='lorentz'),dict(block_size=0)])
def test_expression_options_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate_expression_neighborhoods(np.arange(4.)[:,None],np.zeros((4,2)),**kwargs)


def test_constant_expression_rejected():
    with pytest.raises(ValueError,match='positive-variance'):
        evaluate_expression_neighborhoods(np.ones((4,2)),np.zeros((4,2)),ks=[1])


@pytest.fixture
def job_input(tmp_path):
    f=np.random.default_rng(18).normal(size=(33,4))
    expression=np.column_stack([f,np.ones(33)])
    path=tmp_path/'input.npz'
    np.savez(path,features=f,expression=expression,labels=np.array(['A']*18+['B']*15),
             cell_ids=np.array([f'c{i}' for i in range(33)]))
    return path


@pytest.mark.parametrize('method',['rfa','mu_fixed','mu_local','euclidean_fixed','euclidean_local'])
def test_worker_records_complete_native_geometry_and_expression(method,job_input,tmp_path):
    pytest.importorskip('torch')
    output=tmp_path/method
    r=load('run_mu_study').main(['--input',str(job_input),'--output',str(output),
                               '--method',method,'--epochs','2','--threads','1'])
    assert r['status']=='completed' and r['n_epochs']==2
    assert r['evaluation']['embedding_metric']==('euclidean' if method.startswith('euclidean') else 'poincare')
    assert (r['visibility'] is None)==method.startswith('euclidean')
    assert r['expression']['n_genes_evaluated']==4
    assert r['initial_disk_sha256'] and r['initial_tangent_sha256']
    assert (output/'result.npz').exists()
    assert not list(output.glob('*.tmp'))


def test_worker_exact_resume_is_total_epochs_not_an_extra_run(job_input,tmp_path):
    pytest.importorskip('torch')
    worker=load('run_mu_study')
    base=['--input',str(job_input),'--method','euclidean_local','--threads','1']
    first=tmp_path/'first';continued=tmp_path/'continued';full=tmp_path/'full'
    worker.main(base+['--output',str(first),'--epochs','2'])
    resumed=worker.main(base+['--output',str(continued),'--epochs','4','--resume',str(first/'checkpoint.pt')])
    worker.main(base+['--output',str(full),'--epochs','4'])
    assert resumed['resume_from_epoch']==2 and resumed['n_epochs']==4
    with np.load(continued/'result.npz',allow_pickle=False) as a,np.load(full/'result.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(a['coordinates'],b['coordinates'])
        np.testing.assert_array_equal(a['loss_trace'],b['loss_trace'])


def specs(path):
    return [dict(id='job',input=str(path),method='mu_fixed',epochs=2,seed=0,threads=1)]


def test_supervisor_runs_owned_child_and_collects_result(job_input,tmp_path):
    pytest.importorskip('torch');pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    jobs=tmp_path/'jobs.json';jobs.write_text(json.dumps(specs(job_input)))
    out=tmp_path/'run'
    r=module.main(['--jobs',str(jobs),'--output',str(out),'--workers','1','--wall-seconds','30'])
    assert r['status']=='completed'
    assert r['jobs'][0]['result']['n_epochs']==2
    assert r['jobs'][0]['spec']['dataset']==job_input.stem  # old minimal job schema remains valid
    assert r['jobs'][0]['observed_peak_rss_bytes']>0
    assert not (out/'job/checkpoint.pt').exists()  # completed short runs do not accumulate states
    assert (out/'job/result.npz').exists()


def test_supervisor_refuses_low_system_memory_without_launch(job_input,tmp_path,monkeypatch):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=1))
    jobs=tmp_path/'jobs.json';jobs.write_text(json.dumps(specs(job_input)))
    out=tmp_path/'run'
    r=module.main(['--jobs',str(jobs),'--output',str(out)])
    assert r['status']=='partial' and r['stop_reason']=='system_memory_pressure'
    assert r['jobs'][0]['status']=='not_started'
    assert not (out/'job').exists()


def test_supervisor_time_limit_marks_unstarted_jobs(job_input,tmp_path):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    jobs=tmp_path/'jobs.json';jobs.write_text(json.dumps(specs(job_input)))
    r=module.main(['--jobs',str(jobs),'--output',str(tmp_path/'expired'),
                   '--wall-seconds','0.000000001','--min-available-gib','.001'])
    assert r['stop_reason']=='stage_time_limit'
    assert r['jobs'][0]['status']=='not_started'


def test_supervisor_stop_file_terminates_owned_child_and_keeps_pending_explicit(job_input,tmp_path,monkeypatch):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    class Child:
        pid=99999999
        returncode=None
        def poll(self):return self.returncode
        def terminate(self):self.returncode=-15
        def wait(self,timeout=None):return self.returncode
    child=Child()
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:child)
    original=module.atomic_json
    def stop_after_launch(path,value):
        original(path,value)
        if any(j['status']=='running' for j in value['jobs']):
            (path.parent/'STOP').write_text('operator requested stop')
    monkeypatch.setattr(module,'atomic_json',stop_after_launch)
    entries=specs(job_input)+[dict(specs(job_input)[0],id='second')]
    jobs=tmp_path/'jobs.json';jobs.write_text(json.dumps(entries))
    r=module.main(['--jobs',str(jobs),'--output',str(tmp_path/'stop'),
                   '--workers','1','--min-available-gib','.001'])
    assert r['stop_reason']=='operator_stop_file' and child.returncode==-15
    assert [j['status'] for j in r['jobs']]==['interrupted','not_started']


@pytest.mark.parametrize('dataset',[None,'','../escape','..\\escape'])
def test_supervisor_refuses_unsafe_dataset_before_launch(job_input,tmp_path,dataset):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    job=specs(job_input)[0];job['dataset']=dataset
    jobs=tmp_path/'jobs.json';jobs.write_text(json.dumps([job]))
    out=tmp_path/'run'
    with pytest.raises(SystemExit):module.main(['--jobs',str(jobs),'--output',str(out)])
    assert not out.exists()


@pytest.mark.parametrize('payload',[None,{},'not a list',[None],[1],[{}],
    [dict(input='input.npz',method='mu_fixed',epochs=2,threads=1,seed=0)]])
def test_supervisor_rejects_malformed_job_schema_before_output(tmp_path,payload):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    path=tmp_path/'jobs.json';path.write_text(json.dumps(payload));out=tmp_path/'output'
    with pytest.raises(SystemExit):module.main(['--jobs',str(path),'--output',str(out)])
    assert not out.exists()


def test_supervisor_rejects_unparseable_jobs_json_before_output(tmp_path):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    path=tmp_path/'jobs.json';path.write_text('{broken');out=tmp_path/'output'
    with pytest.raises(SystemExit):module.main(['--jobs',str(path),'--output',str(out)])
    assert not out.exists()


def test_supervisor_refuses_unsafe_job_id(job_input,tmp_path):
    pytest.importorskip('psutil')
    module=load('benchmark_mu_scaleup')
    job=specs(job_input)[0];job['id']='../escape'
    jobs=tmp_path/'jobs.json';jobs.write_text(json.dumps([job]))
    out=tmp_path/'run'
    with pytest.raises(SystemExit):module.main(['--jobs',str(jobs),'--output',str(out)])
    assert not out.exists()
