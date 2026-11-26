"""Conveyor-motion association of native components; no size-based deduplication."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    id: int
    position: np.ndarray
    stamp: float
    observations: int = 1
    frozen: bool = False


class MotionAssociation:
    def __init__(self, gate_m):
        self.gate = gate_m
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 0
        self.last_key = None

    def update(self, epoch, sequence, capture, positions, velocity, max_gap):
        if self.last_key and epoch != self.last_key[0]:
            self.reset()
        if self.last_key and (sequence <= self.last_key[1] or capture <= self.last_key[2]):
            raise ValueError('duplicate or non-monotonic exposure')
        self.last_key = epoch,sequence,capture
        self.tracks = {i:t for i,t in self.tracks.items() if capture-t.stamp <= max_gap}
        old = list(self.tracks.values())
        points = np.asarray(positions, dtype=float).reshape(-1,3)
        predicted = np.array([t.position+velocity*(capture-t.stamp) for t in old]).reshape(-1,3)
        edges = np.linalg.norm(points[:,None,:2]-predicted[None,:,:2],axis=2) <= self.gate
        assigned = []
        for i, pos in enumerate(points):
            matches = np.flatnonzero(edges[i])
            if not len(matches):
                track = Track(self.next_id,pos.copy(),capture)
                self.tracks[self.next_id]=track
                self.next_id+=1
            elif len(matches)==1 and edges[:,matches[0]].sum()==1:
                track=old[matches[0]]
                track.position,track.stamp=pos.copy(),capture
                track.observations+=1
            else:
                track=None  # Keep the observations; do not claim an ambiguous identity.
            assigned.append(track)
        return assigned
