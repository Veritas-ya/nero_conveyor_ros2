import numpy as np
from experiments.graspgenx_sim.run import warmup_model

def test_warmup_is_explicit_and_does_not_feed_target():
    class Backend:
        def infer(self, points, seed, count, topk):
            self.points=points.copy(); self.args=(seed,count,topk)
            return {'metadata':{'inference_seconds':.2}}
    b=Backend(); result=warmup_model(b, 42, 512, 512)
    assert b.points.shape==(512,3) and np.isfinite(b.points).all()
    assert b.args==(1000041,512,512)
    assert result['source']=='synthetic_kernel_warmup'
    assert result['used_for_control_or_success'] is False
    assert result['before_benchmark_wall_clock'] is True
