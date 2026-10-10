import numpy as np
import pytest
from src.phase101_audit import select_shift_stress, stress_environments, paired, ALL_METHODS

def test_select_stress_prefers_robust_route():
    rng=np.random.default_rng(42)
    s=np.column_stack((rng.normal(10,1,400),rng.normal(10.3,1,400)))
    stress={'heavy':np.column_stack((rng.normal(16,3,400),rng.normal(11,1,400)))}
    chosen,diag=select_shift_stress(s,stress,epsilon=.1,lam=1)
    assert chosen==1
    assert diag.feasible.all()

def test_zero_penalty_reduces_to_cvar():
    s=np.column_stack((np.full(200,10.),np.full(200,11.)))
    chosen,_=select_shift_stress(s,{'shift':s},epsilon=.2,lam=0)
    assert chosen==0

def test_infeasible_alternative_not_chosen():
    s=np.column_stack((np.full(200,10.),np.full(200,15.)))
    chosen,_=select_shift_stress(s,{'shift':s},epsilon=.1,lam=10)
    assert chosen==0

def test_stress_deterministic():
    means=np.array([10.,20.,30.])
    a=stress_environments(means,100,123)
    b=stress_environments(means,100,123)
    assert set(a)=={'iid_gamma','heavy_lognormal','correlated_gamma'}
    for key in a:
        assert a[key].shape==(100,3)
        np.testing.assert_array_equal(a[key],b[key])
        assert (a[key]>0).all()

def test_reject_invalid_input():
    with pytest.raises(ValueError):
        select_shift_stress(np.zeros((2,2)),{'x':np.zeros((2,2))})
    with pytest.raises(ValueError):
        stress_environments(np.array([1.,2.]),10,2)

def test_integration_quick_case():
    from src.phase101_audit import run_case
    rows,audit=run_case(topology='grid',n=25,seed=201,shift='iid_gamma',budget=8,
                        dev_scenarios=160,test_count=120,stress_count=100)
    assert set(r['method'] for r in rows)==set(ALL_METHODS)
    assert len(audit)>=2
    assert all(np.isfinite(r['cvar95']) for r in rows)
