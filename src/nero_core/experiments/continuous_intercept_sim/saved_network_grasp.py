"""Explicit saved-network-grasp / simulator-history physical diagnostic.

The network exposure belongs to a previous rendered run. The same-exposure
object anchor here is interpolated from THIS diagnostic's simulated history;
it is not a fresh RGB-D/GraspGen-X/Nero7 run or an end-to-end timing result.
"""
from pathlib import Path
import hashlib
import json
from copy import deepcopy
import numpy as np
from .full_preplanned import matrix,pose
from experiments.graspgenx_sim.continuous import ContinuousGraspAdapter,GraspTrackOverride


class SavedNetworkObservationView:
    # Exact existing capture-to-grasp decoration methods; no new follower.
    shift_value=ContinuousGraspAdapter.shift_value
    shift_mapping=ContinuousGraspAdapter.shift_mapping
    decorate_target=ContinuousGraspAdapter.decorate_target
    decorate_observation=ContinuousGraspAdapter.decorate_observation
    compression_m=.006
    require_network_grasp=True
    _moving_approach_mode="continuous"


class SavedNetworkGraspMirrorMixin:
    def configure_saved_network_grasp(self,path,source_index):
        path=Path(path)
        summary=json.loads(path.read_text())
        choices=[r for r in summary['accepted'] if r['source_index']==source_index]
        if not choices:raise ValueError('saved accepted network candidate not found')
        self.saved_network_candidate=choices[0]
        self.saved_network_capture=float(summary['input']['capture_sim_s'])
        self.saved_network_ready=float(summary['capture_episode_sim_s'])
        self.saved_network_pose=np.array(choices[0]['tcp_pose_world_wxyz'],dtype=float)
        self.saved_network_source_capture=self.saved_network_capture
        self.saved_network_source_ready=self.saved_network_ready
        self.saved_network_source_pose=self.saved_network_pose.copy()
        self.saved_network_template_offset=None
        self.saved_network_target_id=None
        self.saved_network_history=[]
        self.saved_network_offset=None
        self.saved_network_provenance=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_index=source_index,capture_sim_s=self.saved_network_capture,
            replay_geometry_available_sim_s=self.saved_network_ready,
            neural_network_executed_in_this_run=False,camera_executed_in_this_run=False,
            velocity_source='current_diagnostic_simulator_history',
            binding_source='interpolated_current_diagnostic_simulator_history_at_saved_network_exposure',
            original_candidate=self.saved_network_candidate,object_bindings=[])

    def adapt_grasp_observation(self,obs,snap):
        now=float(snap['sim_s'])
        if obs['objects']:
            identifier=int(obs['objects'][0]['object_id'])
            if identifier!=self.saved_network_target_id:
                runtime=self._logical_to_slot[identifier]
                self.saved_network_target_id=identifier
                self.saved_network_capture=float(runtime.spawned_at)+self.saved_network_source_capture
                self.saved_network_ready=float(runtime.spawned_at)+self.saved_network_source_ready
                self.saved_network_history=[]
                self.saved_network_offset=None
        for identifier,prediction in obs['predictions'].items():
            if identifier!=self.saved_network_target_id:continue
            stamp=float(prediction['observed_timestamp'])
            if not self.saved_network_history or stamp>self.saved_network_history[-1][0]:
                self.saved_network_history.append((stamp,np.array(prediction['observed_pose'],copy=True)))
        # The simulator observer has its own exposure/publication delay. The
        # prior network run's ready time does not imply this run has received
        # observations on both sides of that exposure yet.
        bracket_pending=(not self.saved_network_history or
            self.saved_network_history[-1][0]<self.saved_network_capture)
        if now<self.saved_network_ready or now<self.saved_network_capture or bracket_pending:
            obs['objects']=[];obs['predictions']={}
            self._comparison_observation=obs
            return obs
        if self.saved_network_offset is None:
            times=np.array([t for t,_ in self.saved_network_history])
            positions=np.array([p[:3] for _,p in self.saved_network_history])
            if self.saved_network_capture<times[0] or self.saved_network_capture>times[-1]:
                raise ValueError('saved capture has no causal simulated history bracket')
            center=np.array([np.interp(self.saved_network_capture,times,positions[:,i]) for i in range(3)])
            if self.saved_network_template_offset is None:
                self.saved_network_template_offset=self.saved_network_source_pose[:3]-center
            self.saved_network_offset=self.saved_network_template_offset.copy()
            self.saved_network_pose=self.saved_network_source_pose.copy()
            self.saved_network_pose[:3]=center+self.saved_network_offset
            self.saved_network_provenance['object_bindings'].append(dict(
                object_id=self.saved_network_target_id,current_history_capture_s=self.saved_network_capture,
                current_history_capture_position_m=center.tolist(),binding_installed_sim_s=now,
                geometry_source='saved_relative_grasp_template_reused_on_same_shape_and_yaw',
                saved_network_source_capture_s=self.saved_network_source_capture,
                fresh_camera_or_network=False))
            self.saved_network_provenance.update(simulator_history_capture_position_m=center.tolist(),
                tcp_offset_from_simulated_capture_position_m=self.saved_network_offset.tolist(),
                binding_installed_sim_s=now)
            self._saved_network_view=SavedNetworkObservationView()
            self._saved_network_view.overrides={target['object_id']:GraspTrackOverride(
                int(target['object_id']),self.saved_network_pose,center,self.saved_network_capture,
                float(self.saved_network_candidate['closing_width_m']),
                float(self.saved_network_candidate.get('score',0.)),source='saved_GraspGenX_GraspMoE_replay')
                for target in obs['objects']}
        original=deepcopy(obs['predictions'])
        for target in obs['objects']:
            # This replay binds the NN TCP to the simulated body origin, not
            # to the catalog's separately expressed display/support position.
            target['saved_replay_original_reported_position']=np.asarray(target['position']).copy()
            target['position']=np.asarray(original[target['object_id']]['current_estimated_pose'])[:3].copy()
        obs=self._saved_network_view.decorate_observation(obs)
        for identifier,prediction in obs['predictions'].items():
            prediction['source_object_observed_pose']=np.asarray(original[identifier]['observed_pose']).copy()
            prediction['source_object_current_estimated_pose']=np.asarray(original[identifier]['current_estimated_pose']).copy()
        return obs

    def refine_selected_grasp_plan(self,object_id,*args,**kwargs):
        prediction=self._comparison_observation['predictions'].get(object_id)
        offset=None
        if prediction is not None:
            offset=np.linalg.inv(matrix(prediction['object_current_estimated_pose']))@matrix(prediction['current_ee_grasp_pose'])
            if kwargs.get('requested_pose') is not None:
                kwargs=dict(kwargs,requested_pose=pose(matrix(kwargs['requested_pose'])@np.linalg.inv(offset)))
        result=super().refine_selected_grasp_plan(object_id,*args,**kwargs)
        if offset is not None and result.get('accepted') and result.get('predicted_grasp_pose') is not None:
            result=dict(result,predicted_ee_grasp_pose=pose(matrix(result['predicted_grasp_pose'])@offset))
        return result

    retime_selected_grasp_plan=refine_selected_grasp_plan
