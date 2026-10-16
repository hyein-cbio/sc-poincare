"""Read-only granular checks, not a claim of independent biological validity."""
import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from sc_poincare.evaluation import evaluate_expression_neighborhoods, evaluate_ordering


def inputs():
    x = np.arange(5.)
    expression = np.column_stack([x, np.ones(5), x*x])
    points = np.column_stack([np.linspace(0,.4,5), np.zeros(5)])
    return expression, points


@pytest.mark.parametrize('block_size', [1,3,64])
def test_per_gene_matches_independent_reference_and_preserves_aggregate(block_size):
    values, points = inputs();original = values.copy()
    basic = evaluate_expression_neighborhoods(values,points,ks=[1,2],block_size=block_size)
    report = evaluate_expression_neighborhoods(values,points,ks=[1,2],block_size=block_size,
        per_gene=True,gene_names=['marker','constant','marker'],cell_ids=[f'c{i}' for i in range(5)])
    assert report['normalized_mse'] == basic['normalized_mse']
    assert not report['gene_names_unique']
    assert report['per_gene'][1]['normalized_mse'] == {'1':None,'2':None}
    assert not report['per_gene'][1]['evaluated']
    separation = ((points[:,None]-points[None,:])**2).sum(-1)
    radius = (points**2).sum(-1)
    distance = np.arccosh(1+2*separation/((1-radius[:,None])*(1-radius[None,:])))
    np.fill_diagonal(distance,np.inf)
    neighbors = np.argsort(distance,axis=1,kind='stable')
    for k in [1,2]:
        predicted = values[neighbors[:,:k]].mean(1)
        for index in [0,2]:
            expected = np.mean((predicted[:,index]-values[:,index])**2)/values[:,index].var()
            assert report['per_gene'][index]['normalized_mse'][str(k)] == pytest.approx(expected)
        assert np.mean([report['per_gene'][i]['normalized_mse'][str(k)] for i in [0,2]]) == pytest.approx(report['normalized_mse'][str(k)])
    np.testing.assert_array_equal(values,original)
    json.dumps(report,allow_nan=False)


def test_unnamed_genes_keep_column_indices_without_invented_names():
    report = evaluate_expression_neighborhoods(*inputs(),ks=[1],per_gene=True)
    assert [r['gene_index'] for r in report['per_gene']] == [0,1,2]
    assert all(r['gene_name'] is None for r in report['per_gene'])
    assert report['provenance']['cell_ids_sha256'] is None


@pytest.mark.parametrize('kwargs', [dict(per_gene='yes'),dict(gene_names=['a','b','c']),
    dict(per_gene=True,gene_names=['a','b']),dict(per_gene=True,gene_names=['a',None,'b']),
    dict(cell_ids=['same']*5)])
def test_invalid_gene_policy_or_names_are_not_silently_ignored(kwargs):
    with pytest.raises(ValueError):evaluate_expression_neighborhoods(*inputs(),ks=[1],**kwargs)


def test_order_uses_declared_sequence_not_lexicographic_sort_and_no_input_mutation():
    time = np.arange(6.);labels = ['PS','PS','HF','HF','4S','4S']
    report = evaluate_ordering(time,labels,stage_order=['PS','HF','4S'],cell_ids=[f'c{i}' for i in range(6)])
    assert report['spearman'] == pytest.approx(spearmanr(time,[0,0,1,1,2,2]).statistic)
    reversed_order = evaluate_ordering(time,labels,stage_order=['4S','HF','PS'])
    assert reversed_order['spearman'] == pytest.approx(-report['spearman'])
    assert report['stage_order'] == ['PS','HF','4S']
    np.testing.assert_array_equal(time,np.arange(6.))
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('time,stages', [(np.ones(4),['A','A','B','B']), (np.arange(4.),['A']*4)])
def test_constant_ordering_is_undefined_not_zero(time,stages):
    result = evaluate_ordering(time,stages,stage_order=['A','B','C'])
    assert result['spearman'] is None and result['status'].startswith('undefined')
    assert result['stages'][2] == dict(stage='C',n_cells=0,pseudotime_median=None)


@pytest.mark.parametrize('time,stages,order', [
    ([0.,1.],['A','missing'],['A','B']),([0.,np.nan],['A','B'],['A','B']),
    ([0.,1.],['A',None],['A','B']),([0.,1.],[0,1],[0,0.]),
    ([0.,1.],['A','B'],['A']),([0.,1.],['A'],['A','B']),
])
def test_invalid_order_or_missing_stages_rejected_without_dropping_cells(time,stages,order):
    with pytest.raises(ValueError):evaluate_ordering(time,stages,stage_order=order)
