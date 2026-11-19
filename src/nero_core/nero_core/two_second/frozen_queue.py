"""Bounded immutable-exposure mailbox; callers hold the vision lock.

Appending a target cannot replace the exposure already being planned. An idle
planner may explicitly promote a waiting target. No device owner lives here.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenEntry:
    target: object
    accepted_s: float

    def invalid_reason(self, config, epoch, now):
        target = self.target
        if target.epoch != epoch or not target.capture_s <= self.accepted_s <= now:
            return 'clock_invalidated'
        max_age = config.raw['maximum_observation_age_s']
        if self.accepted_s - target.capture_s > max_age:
            return 'stale_at_publication'
        predicted = target.entry_position + target.velocity * (now - target.capture_s)
        if not float(predicted @ config.direction) <= config.follow_bounds[1]:
            return 'past_follow_region'
        return None


class FrozenTargetQueue:
    def __init__(self, capacity):
        self.capacity=capacity
        self.entries=[]
        self.counts=Counter()

    @property
    def head(self):
        return self.entries[0].target if self.entries else None

    @property
    def full(self):
        return len(self.entries)>=self.capacity

    def clear(self):
        self.entries.clear()

    def offer(self,target,now,max_age):
        if not 0<=now-target.capture_s<=max_age:
            self.counts['stale_at_publication']+=1
            return False
        if any(e.target.identity==target.identity for e in self.entries):
            self.counts['duplicate_identity']+=1
            return False
        if self.full:
            self.counts['full']+=1
            return False
        self.entries.append(FrozenEntry(target,now))
        self.counts['accepted']+=1
        return True

    def remove(self,identity,reason):
        for i,entry in enumerate(self.entries):
            if entry.target.identity==identity:
                self.entries.pop(i)
                self.counts[reason]+=1
                return entry.target

    def promote(self, target):
        """Reserve an existing exposure only while the planning owner is idle."""
        for i, entry in enumerate(self.entries):
            if entry.target is target:
                self.entries.insert(0, self.entries.pop(i))
                if i:
                    self.counts['planning_promotions'] += 1
                return True
        return False

    def admitted(self,target,now,max_age):
        return any(e.target is target and target.capture_s<=e.accepted_s<=now
            and e.accepted_s-target.capture_s<=max_age for e in self.entries)

    def snapshot(self):
        return dict(capacity=self.capacity,size=len(self.entries),counts=dict(self.counts),
            targets=[dict(identity=e.target.identity,capture_s=e.target.capture_s,
                          accepted_monotonic_s=e.accepted_s) for e in self.entries])
