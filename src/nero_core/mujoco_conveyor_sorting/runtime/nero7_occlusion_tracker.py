"""Conservative occlusion handling around the unchanged upstream tracker."""
import numpy as np

from vendor.nero7_speed_reference.nero7_perception.block_tracker import BlockTracker


class _BeltRelativeMotion:
    """Run the original filter in a frame moving at the operator's belt speed.

    Only association uses this prior. Observation stamps/counts still belong
    to accepted real detections; the prior cannot qualify a measured speed.
    Public positions stay in robot base coordinates, including after misses.
    """

    def __init__(self, motion, velocity_xy):
        self._motion = motion
        self._velocity = velocity_xy
        self._reference_s = motion.state_stamp

    def __getattr__(self, name):
        return getattr(self._motion, name)

    def _offset(self, stamp):
        return self._velocity * (float(stamp)-self._reference_s)

    def _base_state(self, state, stamp):
        state = state.copy()
        state[:2] += self._offset(stamp)
        state[2:] += self._velocity
        return state

    @property
    def state(self):
        return self._base_state(self._motion.state, self._motion.state_stamp)

    @property
    def observed_xy(self):
        return self._motion.observed_xy+self._offset(self._motion.observation_stamp)

    @property
    def speed_valid(self):
        return False  # The association state contains an operator-supplied prior.

    def forecast(self, stamp):
        state, covariance = self._motion.forecast(stamp)
        return self._base_state(state, stamp), covariance

    def innovation(self, xy, stamp):
        return self._motion.innovation(np.asarray(xy)-self._offset(stamp), stamp)

    def update(self, xy, stamp):
        return self._motion.update(np.asarray(xy)-self._offset(stamp), stamp)


class OcclusionAwareBlockTracker(BlockTracker):
    """Defer assignments when the existing motion gates permit multiple IDs.

    Never treat a prediction as an observation. The original Kalman filter,
    gating, expiry, and one-to-one tracker process all unambiguous detections.
    """

    def __init__(self, *args, association_velocity_xy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.association_velocity_xy = None
        if association_velocity_xy is not None:
            velocity = np.asarray(association_velocity_xy, dtype=float)
            if velocity.shape != (2,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity)<=0:
                raise ValueError('ID association belt velocity must be a finite nonzero XY vector')
            self.association_velocity_xy = velocity.copy()
            self.association_velocity_xy.setflags(write=False)
        self.ambiguous_frames = 0
        self.last_ambiguous = None

    def update(self, detections, t):
        t = float(t)
        if not np.isfinite(t) or (self._last_t is not None and t <= self._last_t):
            return super().update(detections, t)
        active = [(i, tr) for i, tr in enumerate(self.tracks)
                  if t - tr.stamp <= self.lost_timeout and tr.misses <= self.lost_frames]
        pairs = []
        for i, tr in active:
            physical_gate = self.gate_radius + self.max_vel * (t-tr.stamp)
            for j, detection in enumerate(detections):
                position = np.asarray(detection['centroid_base'], float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    continue
                innovation = tr.motion.innovation(position[:2], t)
                if (innovation is not None and innovation[1] <= physical_gate
                        and innovation[0] <= tr.motion.innovation_gate):
                    pairs.append((i, j))
        by_track, by_detection = {}, {}
        for i, j in pairs:
            by_track.setdefault(i, set()).add(j)
            by_detection.setdefault(j, set()).add(i)
        ambiguous_detections = set()
        for i, j in pairs:
            competing_tracks = by_detection[j]
            missed_competitor = any(self.tracks[k].misses for k in competing_tracks)
            merged_or_reacquired = len(competing_tracks) > 1 and (
                len(by_detection) < len(active) or missed_competitor)
            split_after_loss = self.tracks[i].misses > 0 and len(by_track[i]) > 1
            if merged_or_reacquired or split_after_loss:
                ambiguous_detections.update(by_track[i])
        # Propagate through the connected component so a neighboring track
        # cannot consume a competing detection after its peer is withheld.
        while True:
            previous = len(ambiguous_detections)
            for i, j in pairs:
                if j in ambiguous_detections:
                    ambiguous_detections.update(by_track[i])
            if len(ambiguous_detections) == previous:
                break
        if ambiguous_detections:
            self.ambiguous_frames += 1
            self.last_ambiguous = {'source_time': t,
                'track_ids': sorted({self.tracks[i].id for i,j in pairs if j in ambiguous_detections}),
                'detection_count': len(ambiguous_detections)}
        tracks = super().update([d for j,d in enumerate(detections) if j not in ambiguous_detections], t)
        if self.association_velocity_xy is not None:
            for track in tracks:
                if not isinstance(track.motion, _BeltRelativeMotion):
                    track.motion = _BeltRelativeMotion(track.motion, self.association_velocity_xy)
                    track._sync_motion()
        return tracks
