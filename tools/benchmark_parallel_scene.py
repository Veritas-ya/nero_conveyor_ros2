"""Offline only: compare a private precomputed belt scene with serial planning.

This experiment never imports hardware adapters and is not an execution mode.
All command-line arguments are passed to replay_recorded_pick_plan.py. Set
NERO_PARALLEL_SCENE_EXPERIMENT=1 to overlap scene preparation with IK.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import os
import time
import numpy as np
from scipy.spatial import cKDTree
import replay_recorded_pick_plan as replay
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.belt_collision_scene import install_belt_surface


class BenchmarkPlanner(MeasuredPlanner):
    @staticmethod
    def _snapshot_target(target):
        # The worker owns immutable copies of all scene arrays it will read.
        detection=dict(target.detection)
        detection['scene_points']=np.array(detection['scene_points'],copy=True)
        detection['scene_points'].setflags(write=False)
        pixels=detection.get('scene_pixel_indices')
        if pixels is not None:
            detection['scene_pixel_indices']=np.array(pixels,copy=True)
            detection['scene_pixel_indices'].setflags(write=False)
        region=dict(detection['belt_region'])
        for key in ('plane_to_base','bounds_xy_m','scene_bounds_xy_m'):
            if key in region:
                region[key]=np.array(region[key],copy=True)
                region[key].setflags(write=False)
        detection['belt_region']=region
        return SimpleNamespace(detection=detection,identity=target.identity,sequence=target.sequence,
            epoch=target.epoch,capture_s=target.capture_s,points=target.points)

    def _prepare_shadow(self, target):
        started = time.perf_counter()
        shadow = SimpleNamespace(belt_scene_filter_config=self.model.belt_scene_filter_config)
        report = install_belt_surface(shadow, target)
        target_tree = cKDTree(target.points)
        return shadow, report, target_tree, time.perf_counter()-started

    def intercept(self, target, state, *, cancelled=lambda: False):
        wrapper_started=time.perf_counter()
        self._benchmark_target = target
        self._parallel = os.environ.get('NERO_PARALLEL_SCENE_EXPERIMENT') == '1'
        if self.config.raw.get('environment_collision_scope') != 'measured_belt_surface_only':
            raise ValueError('experiment supports measured belt scene only')
        if not self._parallel:
            result=super().intercept(target, state, cancelled=cancelled)
        else:
            snapshot_started=time.perf_counter()
            snapshot=self._snapshot_target(target)
            self._snapshot_wall=time.perf_counter()-snapshot_started
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='offline-scene-prototype') as pool:
                self._shadow_future=pool.submit(self._prepare_shadow,snapshot)
                result=super().intercept(target, state, cancelled=cancelled)
        wrapper_wall=time.perf_counter()-wrapper_started
        result=replace(result,path=replace(result.path,planning_s=wrapper_wall))
        result.timings_s['full_planning_delivery']=wrapper_wall
        result.timings_s['scene_experiment']={
            'mode': 'parallel_thread_prototype' if self._parallel else 'serial',
            'scene_sha256':sha256(np.ascontiguousarray(self.model.scene).tobytes()).hexdigest(),
            'target_tree_sha256':sha256(np.ascontiguousarray(self.target_tree.data).tobytes()).hexdigest(),
            'scene_points':len(self.model.scene),
            'snapshot_copy_wall_s':getattr(self,'_snapshot_wall',None),
            'worker_wall_s':getattr(self,'_scene_worker_wall',None),
            'wait_wall_s':getattr(self,'_scene_wait_wall',None),
            'note':'worker_wall may overlap IK, never add it to total'}
        return result

    def install(self, target, state):
        if not self._parallel:
            return super().install(target,state)
        if target is not self._benchmark_target:
            raise ValueError('prototype scene identity mismatch')
        self.model.check_feedback(state)
        started=time.perf_counter()
        shadow,report,target_tree,worker_wall=self._shadow_future.result()
        self._scene_wait_wall=time.perf_counter()-started
        self._scene_worker_wall=worker_wall
        for name in ('scene','tree','target','scene_voxel_result','scene_source_indices',
                     'scene_native_pixel_indices','_observed_scene_index'):
            setattr(self.model,name,getattr(shadow,name))
        self.scene_report=report
        self.identity=target.identity
        self.target=target
        self.target_tree=target_tree
        self.payload_local=None
        self.payload_tree=self.released_points=self.released_tree=None
        self.scene_revision+=1
        self.version+=1
        plane=np.asarray(target.detection['belt_region']['plane_to_base'])
        self.plane=plane.copy()
        self.belt_bounds=np.asarray(target.detection['belt_region']['bounds_xy_m'])
        self.support,self.support_origin=plane[:3,2].copy(),plane[:3,3].copy()


if __name__ == '__main__':
    replay.MeasuredPlanner=BenchmarkPlanner
    replay.main()
