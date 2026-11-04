"""Alternate exact pre-cache and cached certificates on one test-model input.

The old implementation is reconstructed from the current function by undoing
only the sample cache. This is not a real-robot or real-scene benchmark.
"""
import inspect,textwrap,types,time,json,statistics
import numpy as np
from mujoco_conveyor_sorting.controllers.full_pick_planning import FullPickPlanningMixin
from test_planning_worker_scene_validator import model,make_engine
from test_full_pick_planning import full_request
src=textwrap.dedent(inspect.getsource(FullPickPlanningMixin._full_certify_curve))
start=src.index('    # A curve is immutable during this certification.')
end=src.index('    # Reject a dynamically impossible clock',start)
src=src[:start]+src[end:]
src=src.replace('q, velocity, acceleration = curve.sample_full(float(offset))\n        samples.append((q, velocity, acceleration))',
                'q, velocity, _ = curve.sample_full(float(offset))',1)
src=src.replace('states = tuple(sample[0] for sample in samples)','states = tuple(curve.sample_full(float(t))[0] for t in times)')
src=src.replace('previous = samples[0][0]','previous = curve.sample_full(0.)[0]')
src=src.replace('for offset, (q, velocity, _) in zip(times, samples):\n        if cancelled():','for offset in times:\n        if cancelled():',1)
needle="        closing_approach=(request.full_contact_handoff_enabled"
pos=src.index(needle); before=src[:pos]; after=src[pos:]; last=before.rfind("        if phase == 'closing' and not command:")
before=before[:last]+"        q, velocity, _ = curve.sample_full(float(offset))\n"+before[last:];src=before+after
ns={};exec(src,FullPickPlanningMixin._full_certify_curve.__globals__,ns);old=ns['_full_certify_curve']
req=full_request();baseline=[];new=[];reference=None
for i in range(10):
 for label in (('old','new') if i%2==0 else ('new','old')):
  engine=make_engine(model.__wrapped__())
  if label=='old':engine._full_certify_curve=types.MethodType(old,engine)
  start=time.perf_counter(); result=engine.plan(req,lambda:False); elapsed=time.perf_counter()-start
  if not result.success:raise RuntimeError(result.error)
  samples=np.array([q for _,_,q in result.full_pick_plan.checked_samples])
  times=np.array([t for _,t,_ in result.full_pick_plan.checked_samples])
  if reference is None:reference=(times,samples)
  if not(np.array_equal(reference[0],times) and np.array_equal(reference[1],samples)):raise RuntimeError('certificate samples differ')
  (baseline if label=='old' else new).append(elapsed)
print(json.dumps({'baseline':baseline,'optimized':new,'baseline_median_s':statistics.median(baseline),'optimized_median_s':statistics.median(new),'identical_certificate_samples':len(reference[0])}))
