import numpy as np
import pandas as pd
import pytest
from src.phase102_diagnostics import (controlled_tradeoff, sweep_objectives,
    pareto_candidates, analyze_candidates, read_audit)

KEY=dict(topology='grid',requested_n=25,seed=201,shift='iid_gamma',budget=8)

def sample():
    return pd.DataFrame([
        dict(KEY,candidate_id=0,selection_mean=10.,selection_cvar95=15.,
             worst_stress_cvar95=19.,shift_penalty=4.,shift_objective=16.,
             shift_feasible=True,test_mean=10.,test_cvar95=15.),
        dict(KEY,candidate_id=1,selection_mean=10.5,selection_cvar95=12.,
             worst_stress_cvar95=13.,shift_penalty=1.,shift_objective=12.25,
             shift_feasible=True,test_mean=10.5,test_cvar95=13.),
        dict(KEY,candidate_id=2,selection_mean=13.,selection_cvar95=9.,
             worst_stress_cvar95=10.,shift_penalty=1.,shift_objective=9.25,
             shift_feasible=False,test_mean=13.,test_cvar95=9.)])

def test_controlled_objectives_disagree():
    d=controlled_tradeoff(5000,10000,seed=42)
    chosen=dict(zip(d.method,d.chosen_route))
    assert chosen['expected_cost']=='B_risky'
    assert chosen['cvar_only']=='A_safe'
    assert d.loc[d.method=='expected_cost','test_cvar95'].iloc[0]>9.5

def test_epsilon_feasibility():
    x=sweep_objectives(sample(),epsilons=(0.,.1,.4),lambdas=(0.,))
    assert x.chosen_id.tolist()==[0,1,2]

def test_shift_penalty_can_change_choice():
    x=sample();x.loc[x.candidate_id==1,'shift_penalty']=100
    a=sweep_objectives(x,epsilons=(.1,),lambdas=(0.,1.))
    assert a.chosen_id.tolist()==[1,0]

def test_pareto():
    x=pareto_candidates(sample())
    assert x.pareto_efficient.all()

def test_candidate_audit_diagnostics():
    d=analyze_candidates(sample())
    assert d.iloc[0].feasible_count==2
    assert bool(d.iloc[0].has_test_better_alternative)
    assert d.iloc[0].chosen_id==1

def test_duplicate_detection(tmp_path):
    f=tmp_path/'audit.csv';pd.concat([sample(),sample().iloc[[0]]]).to_csv(f,index=False)
    with pytest.raises(ValueError,match='Duplicate'):read_audit(f)

def test_invalid_controlled_counts():
    with pytest.raises(ValueError):controlled_tradeoff(5,1000)
