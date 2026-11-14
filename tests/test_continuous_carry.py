"""Whole-chain timing, moving via-points, final orientation and release order."""
from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
import pytest
from scipy.interpolate import BPoly, PPoly

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.continuous_carry import compile_carry
from nero_core.two_second.prepared_cycle import concatenate
from nero_core.two_second.trajectory import TimedPath, extrema
from test_prepared_taught_cycle import fixture


def chain():
    cfg,p,q=fixture()
    p.model.ik=NS(config=NS(orientation_tolerance=.01))
    cfg.raw.update(geometry_check_hz=25.,taught_joint_jerk_limit_rad_s3=500.)
    states=[q(0.),q(.1),q(.4),q(.6)]
    parts=[]
    for i,(start,end) in enumerate(zip(states[:-1],states[1:])):
        curve=PPoly.from_bernstein_basis(BPoly.from_derivatives([0.,1.],
            [[start,np.zeros(7),np.zeros(7)],[end,np.zeros(7),np.zeros(7)]]))
        parts.append(TimedPath(curve,p.identity,1,str(i),1.,0.))
    # Nested pieces must retain BOTH internal join times in the final chain.
    transfer=concatenate(parts[1:],identity=p.identity,version=1,stage='direct_drop')
    path=concatenate([parts[0],transfer],identity=p.identity,version=1,stage='carry')
    p.model.set_state(states[-1],.04);goal=p.model.tcp().copy()
    return p,path,goal


def compile(p,path,goal,**kwargs):
    return compile_carry(p,path,.04,lift_end_s=1.,terminal_start_s=2.,
        destination=goal,final_radius_m=.1,cancelled=kwargs.get('cancelled',lambda:False))


def test_one_global_time_scale_and_no_rest_at_either_nested_join():
    p,path,goal=chain();p.config.acceleration_limits[:]=.1
    result=compile(p,path,goal);report=result.diagnostics['continuous_carry']
    assert report['single_stream'] and report['internal_planned_stops']==0
    assert report['internal_arrival_checks']==0 and not report['internal_planning']
    assert result.identity==path.identity and result.version==path.version
    scale=result.duration/path.duration
    assert scale>1 and len(report['joins'])==2
    for old_t in [0.,1.,3.]:
        np.testing.assert_allclose(result.curve(old_t*scale),path.curve(old_t),atol=1e-10)
    assert not np.allclose(result.curve(2*scale),path.curve(2.))
    assert result.diagnostics['protected_waypoint_times_s']==pytest.approx([scale])
    assert report['bin_approach_rounding']
    for join in report['joins']:
        assert join['max_joint_speed_rad_s']>0.01/scale
    assert report['lift_end_s']==pytest.approx(scale)
    assert report['terminal_start_s']==pytest.approx(2*scale)
    for d in [0,1,2]:
        knots=result.curve.x[1:-1]
        np.testing.assert_allclose(result.curve(np.nextafter(knots,-np.inf),d),
                                   result.curve(knots,d),atol=1e-8)
    for d,limit in [(1,p.config.velocity_limits),(2,p.config.acceleration_limits),(3,500.)]:
        assert np.all(abs(np.asarray(extrema(result.curve,d)))<=limit+1e-8)
    assert report['rest_qva_endpoints'] and ('check','carry',True) in p.calls


def test_failed_continuous_curve_is_not_reported_as_nonstop_carry():
    p,path,goal=chain()
    def reject(*a,**kw):raise PickRejected('rim geometry')
    p.check_path=reject
    with pytest.raises(PickRejected,match='连续连接') as error:
        compile(p,path,goal)
    assert all(x['reason']=='rim geometry' for x in error.value.details['failed_attempts'])
    with pytest.raises(PickRejected,match='取消'):
        compile(p,path,goal,cancelled=lambda:True)


def test_final_orientation_check_applies_to_whole_curve_after_blending():
    p,path,goal=chain()
    # Fixture TCP has identity orientation, so this goal cannot pass even
    # though the fake scene checker would otherwise accept the curve.
    goal[:3,:3]=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    with pytest.raises(PickRejected,match='连续连接') as error:
        compile(p,path,goal)
    assert all('姿态仍未完成' in x['reason'] for x in error.value.details['failed_attempts'])
