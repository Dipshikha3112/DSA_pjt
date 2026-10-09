import numpy as np
import pandas as pd
import pytest
from src.distribution_shift_phase10 import (METHODS,SHIFTS,generate_test_scenarios,training_pools,
    run_case,paired_results,cluster_bootstrap,run_benchmark)


def test_shifts_positive():
    for shift in SHIFTS:
        x=generate_test_scenarios(np.array([8.,12.]),shift,500,42)
        assert x.shape==(500,2) and np.isfinite(x).all() and (x>=0).all()


def test_test_reproducible():
    a=generate_test_scenarios(np.array([8.,12.]),'correlated_gamma',200,42)
    b=generate_test_scenarios(np.array([8.,12.]),'correlated_gamma',200,42)
    assert np.array_equal(a,b)


def test_shock_correlation():
    x=generate_test_scenarios(np.ones(8)*10,'correlated_gamma',10000,77)
    assert np.corrcoef(x[:,0],x[:,1])[0,1] > .25


def test_bad_shift():
    with pytest.raises(ValueError): generate_test_scenarios(np.array([10.]),'bad',200,42)


def test_train_disjoint_reproducible():
    a=training_pools(np.array([8.,12.]),400,10)
    b=training_pools(np.array([8.,12.]),400,10)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert sum(len(x) for x in a)==400


def test_case_methods():
    rows=run_case('grid',25,201,'iid_gamma',8,160,200)
    assert {r['method'] for r in rows}==set(METHODS)
    assert len(rows)==8


def test_case_reproducible():
    a=run_case('grid',25,201,'heavy_lognormal',8,160,200)
    b=run_case('grid',25,201,'heavy_lognormal',8,160,200)
    assert [x['path'] for x in a]==[x['path'] for x in b]
    assert [x['cvar95'] for x in a]==[x['cvar95'] for x in b]


def test_shift_does_not_change_selection():
    a=run_case('grid',25,201,'iid_gamma',8,160,200)
    b=run_case('grid',25,201,'heavy_lognormal',8,160,200)
    assert [x['path'] for x in a]==[x['path'] for x in b]


def test_paired_zero_baseline():
    rows=run_case('grid',25,201,'iid_gamma',8,160,200)
    paired=paired_results(pd.DataFrame(rows))
    base=paired[paired.method=='cvar_fixed']
    assert len(base)==1 and base.iloc[0].delta_cvar95==0


def test_cluster_constant():
    rows=run_case('grid',25,201,'iid_gamma',8,160,200)
    paired=paired_results(pd.DataFrame(rows))
    result=cluster_bootstrap(paired,draws=100)
    assert result[result.method=='cvar_fixed'].iloc[0].ci95_high==0


def test_quick_integration(tmp_path):
    df,paired,summary,inf=run_benchmark(tmp_path/'new',True,160,200,100)
    assert len(df)==48 and len(paired)==48 and len(summary)==16
    assert (tmp_path/'new'/'experiment_config.json').exists()
    with pytest.raises(FileExistsError):run_benchmark(tmp_path/'new',True,160,200,100)
