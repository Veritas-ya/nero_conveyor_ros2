"""Offline orchestration contracts. No hardware operations are performed here."""
from dataclasses import dataclass
from enum import Enum

class Stage(str, Enum):
    WAIT = 'waiting'
    PLAN = 'planning'
    APPROACH = 'approach'
    FOLLOW_CLOSE = 'follow_close'
    LIFT = 'lift'
    PLACE = 'place'
    RETURN = 'return'
    FAILED = 'failed'

@dataclass(frozen=True)
class Identity:
    task: str
    target: str
    version: int
    epoch: int

class TaskGate:
    """Reject stale asynchronous results across cancellation, target and clock changes."""
    def __init__(self):
        self.epoch = 0
        self.active = None
        self.consumed = set()
        self.stage = Stage.WAIT

    def start(self, task, target):
        if self.stage != Stage.WAIT or target in self.consumed:
            raise ValueError('busy or already consumed target')
        self.active = Identity(task, target, 0, self.epoch)
        self.stage = Stage.PLAN
        return self.active

    def request(self):
        if self.active is None: raise ValueError('no active task')
        i = self.active
        self.active = Identity(i.task, i.target, i.version + 1, i.epoch)
        return self.active

    def accepts(self, result):
        return self.active is not None and result == self.active

    def cancel(self):
        if self.active is not None: self.consumed.add(self.active.target)
        self.active = None
        self.stage = Stage.WAIT

    def reset_clock(self):
        self.cancel()
        self.epoch += 1

    def advance(self, event):
        transitions = {(Stage.PLAN,'certified'):Stage.APPROACH, (Stage.APPROACH,'at_follow'):Stage.FOLLOW_CLOSE,
          (Stage.FOLLOW_CLOSE,'grasp_confirmed'):Stage.LIFT, (Stage.FOLLOW_CLOSE,'grasp_failed'):Stage.FAILED,
          (Stage.LIFT,'lifted'):Stage.PLACE, (Stage.PLACE,'released'):Stage.RETURN}
        if self.stage == Stage.RETURN and event == 'home':
            self.cancel(); return self.stage
        key = (self.stage, event)
        if key not in transitions: raise ValueError('invalid transition')
        self.stage = transitions[key]
        return self.stage
