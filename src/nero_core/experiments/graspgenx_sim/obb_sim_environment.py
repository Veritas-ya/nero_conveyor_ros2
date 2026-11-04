"""A MuJoCo-only environment that checks commands at actuator application."""
from .nero7_perception import Nero7FixedSimulation
from .obb_sim_dispatch import SimulationDispatchRejected

class OBBCertifiedSimulation(Nero7FixedSimulation):
    def _queue_action(self, action):
        super()._queue_action(action)
        if action is not None and self._action_queue:
            self._action_queue[-1][1]['_obb_source_action']=dict(action)

    def _apply_due_actions(self):
        if not self._action_queue or self._action_queue[0][0] > self.data.time+1e-12:
            return
        if not hasattr(self, 'obb_apply_guard'):
            return super()._apply_due_actions()
        # Preserve the base environment's newest-due-wins behaviour.
        due=[]
        while self._action_queue and self._action_queue[0][0]<=self.data.time+1e-12:
            due.append(self._action_queue.popleft())
        source=due[-1][1].get('_obb_source_action')
        try:
            validated=self.obb_apply_guard(source,float(self.data.time))
        except SimulationDispatchRejected as exc:
            self.obb_dispatch_fault=str(exc)
            self._action_queue.clear()
            return
        self._action_queue.appendleft((float(self.data.time),validated))
        super()._apply_due_actions()
        if hasattr(self, 'obb_applied'):
            self.obb_applied(validated, float(self.data.time))
