"""Bounded in-memory evidence for route failures; never changes acceptance."""
from collections import Counter
import numpy as np


class RouteDiagnostics:
    def __init__(self, *, stage, identity, payload, start, measured, width, goal_joints,
                 goal_pose, scene_revision, scene_report, budget_s):
        self.context = {'stage':stage, 'target_identity':identity, 'payload':bool(payload),
            'start_joints_rad':np.asarray(start).tolist(),
            'measured_joints_rad':np.asarray(measured).tolist(), 'gripper_width_m':float(width),
            'goal_joints_rad':None if goal_joints is None else np.asarray(goal_joints).tolist(),
            'goal_pose_base':None if goal_pose is None else np.asarray(goal_pose).tolist(),
            'scene_revision':scene_revision, 'scene_scope':None if scene_report is None else scene_report.get('scope'),
            'scene_frame_sequence':None if scene_report is None else scene_report.get('frame_sequence'),
            'scene_capture_monotonic_s':None if scene_report is None else scene_report.get('capture_monotonic_s'),
            'planning_budget_s':float(budget_s)}
        self.rejection_counts = Counter()
        self.check_counts = Counter()
        self.examples = []
        self.goals = []
        self.ik_attempts = 0
        self.ik_failures = 0
        self.ik_examples = []
        self.curve_failures = []
        self.times = {'goal_ik':0., 'path_search':0., 'time_parameterization':0., 'curve_geometry':0.}

    def checked(self, role, q, error=None):
        self.check_counts[role] += 1
        if error is None:
            return
        details = error.details or {}
        category = str(details.get('kind') or str(error).split('：')[0])
        key = role+' | '+category
        self.rejection_counts[key] += 1
        # Bounded evidence avoids saving every RRT candidate or giant point clouds.
        if len(self.examples) < 12 and not any(x['key']==key for x in self.examples):
            self.examples.append({'key':key, 'role':role, 'reason':str(error),
                'details':details, 'joints_rad':np.asarray(q).tolist()})

    def snapshot(self, *, outcome, elapsed_s, cancelled, timed_out):
        return {**self.context, 'outcome':outcome, 'elapsed_s':float(elapsed_s),
            'cancelled':bool(cancelled), 'timed_out':bool(timed_out),
            'operation_timings_s':dict(self.times), 'state_check_counts':dict(self.check_counts),
            'rejection_counts':dict(self.rejection_counts), 'rejection_examples':list(self.examples),
            'goal_attempts':list(self.goals), 'ik_attempts':self.ik_attempts,
            'ik_failures':self.ik_failures, 'ik_failure_examples':list(self.ik_examples),
            'curve_failures':list(self.curve_failures)}
