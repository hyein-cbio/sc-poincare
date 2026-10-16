"""Unweighted topology reconstruction is labeled separately from learned targets."""
import numpy as np
import pytest

from sc_poincare.diagnostics import graph_topology


def test_known_fragmentation_and_connected_union_without_mutating_features():
    features=np.array([[0.],[.1],[.4],[1.2],[2.5]])
    original=features.copy()
    d=graph_topology(features,n_neighbors=1)
    assert d['scope']=='unweighted_pre_stitch_reconstruction'
    assert d['cutoff_tie_rows']==0
    assert d['mutual']['n_components_before_stitching']==4
    assert d['mutual']['n_isolated_before_stitching']==3
    assert d['mutual']['required_bridges_for_connection']==3
    assert d['union']['n_components_before_stitching']==1
    assert d['union']['required_bridges_for_connection']==0
    np.testing.assert_array_equal(features,original)


def test_duplicate_zero_length_edges_are_connections_and_ties_are_explicit():
    d=graph_topology(np.zeros((5,2)),n_neighbors=1)
    assert d['cutoff_tie_rows']==5
    complete=graph_topology(np.zeros((5,2)),n_neighbors=20)
    assert complete['n_neighbors_effective']==4 and complete['cutoff_tie_rows']==0
    for mode in ['mutual','union']:
        assert complete[mode]['n_components_before_stitching']==1
        assert complete[mode]['n_edges_before_stitching']==10
        assert complete[mode]['n_isolated_before_stitching']==0


@pytest.mark.parametrize('features,k',[(np.ones((2,3)),1),(np.ones((5,3)),0),
                                      (np.ones((5,3)),True),(np.ones((5,3),complex),2)])
def test_invalid_diagnostic_inputs(features,k):
    with pytest.raises(ValueError):graph_topology(features,n_neighbors=k)
