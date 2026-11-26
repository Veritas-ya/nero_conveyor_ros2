"""Certify measured carry inputs off the sender's Python process."""
from copy import deepcopy
from hashlib import sha256
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .carry_lookahead import revalidate_cycle, validate_inputs
from .runtime_sweep import scene_snapshot, restore_scene


def input_signature(planner,state,anchor):
    """Bind the result to the exact frozen attachment and measured opening."""
    digest=sha256()
    for value in (planner.payload_local,planner.capture_tcp,anchor,
                  state['joints'],state['pose'],[state['width_m']]):
        data=np.asarray(value,dtype=np.float64)
        digest.update(str(data.shape).encode());digest.update(data.tobytes())
    return digest.hexdigest()


def validation_request(planner,state,anchor,result,*,token,source_version):
    validate_inputs(planner,state,anchor,result,token=token,source_version=source_version,
                    cancelled=lambda:False)
    snapshot=scene_snapshot(planner,result['bundle'].carry)
    snapshot.update(payload_local=planner.payload_local.copy(),capture_tcp=planner.capture_tcp.copy(),
                    source_version=source_version)
    return {'snapshot':snapshot,'state':deepcopy(state),'anchor':np.asarray(anchor).copy(),
        'prediction':result,'prediction_token':token,'source_version':source_version,
        'input_signature':input_signature(planner,state,anchor),
        'submitted_monotonic_s':time.monotonic()}


def validate_in_private_model(planner,request,*,cancelled):
    began=time.monotonic();snapshot=request['snapshot']
    collisions=[]
    answer={'validation_token':snapshot['token'],'input_signature':request['input_signature'],
        'identity':snapshot['identity'],'scene_revision':snapshot['scene_revision'],
        'source_version':request['source_version'],'prediction_token':request['prediction_token'],
        'pid':os.getpid(),'queue_wall_s':began-request['submitted_monotonic_s']}
    try:
        if cancelled():raise PickRejected('搬运后台复核已取消')
        restore_scene(planner,snapshot)
        planner.collision_sink=collisions.append
        planner.version=request['source_version']
        planner.payload_local=snapshot['payload_local']
        planner.payload_tree=cKDTree(planner.payload_local)
        planner.capture_tcp=snapshot['capture_tcp']
        installed=time.monotonic()
        _,evidence=revalidate_cycle(planner,request['state'],request['anchor'],
            request['prediction'],token=request['prediction_token'],
            source_version=request['source_version'],cancelled=cancelled,
            defer_return_validation=True)
        answer.update(ok=True,evidence=evidence,scene_restore_wall_s=installed-began)
    except Exception as error:
        answer.update(ok=False,error=str(error),details=getattr(error,'details',{}),
                      collision_packets=collisions)
    finally:
        planner.collision_sink=None
    answer.update(worker_wall_s=time.monotonic()-began,completed_monotonic_s=time.monotonic())
    return answer


def accept_validation(planner,state,anchor,result,request,answer,*,token,source_version,cancelled):
    began=time.monotonic()
    bundle,_=validate_inputs(planner,state,anchor,result,token=token,
                            source_version=source_version,cancelled=cancelled)
    expected={'validation_token':request['snapshot']['token'],
        'input_signature':input_signature(planner,state,anchor),
        'identity':planner.identity,'scene_revision':planner.scene_revision,
        'source_version':source_version,'prediction_token':token}
    if any(answer.get(key)!=value for key,value in expected.items()):
        raise PickRejected('搬运后台复核不属于当前载物、开口或任务')
    if not answer.get('ok'):
        errors=[]
        sink=getattr(planner,'collision_sink',None)
        for packet in answer.get('collision_packets',()):
            if sink is not None:
                packet={**packet,'frame':getattr(planner.target,'frame',None)}
                try:sink(packet)
                except Exception as error:errors.append(f'{type(error).__name__}: {error}')
        raise PickRejected('搬运后台复核失败：'+answer.get('error','没有认证结果'),
            details={**answer.get('details',{}),'collision_record_errors':errors})
    if cancelled():raise PickRejected('搬运后台复核已取消')
    return bundle,{**answer['evidence'],'validation_execution':'private_process',
        'validation_worker_pid':answer['pid'],'validation_queue_wall_s':answer['queue_wall_s'],
        'validation_worker_wall_s':answer['worker_wall_s'],
        'validation_scene_restore_wall_s':answer['scene_restore_wall_s'],
        'validation_handoff_wall_s':time.monotonic()-began}
