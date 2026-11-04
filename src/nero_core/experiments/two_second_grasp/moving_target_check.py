"""Simulation adapter for production moving-workpiece point-cloud collision."""
import numpy as np
from scipy.spatial import cKDTree
from nero_core.two_second.moving_target_collision import first_moving_target_hit


class SimulationMovingTargetCheck:
    def __init__(self, mirror, cloud_world, world_to_base, width):
        self.mirror=mirror
        transform=np.asarray(world_to_base,dtype=float)
        cloud=np.asarray(cloud_world,dtype=float)
        self.points=(cloud@transform[:3,:3].T+transform[:3,3]).copy()
        self.points.setflags(write=False)
        self.tree=cKDTree(self.points)
        self.rotation=transform[:3,:3].copy()
        self.width=float(width)

    def check(self, q, reference, elapsed):
        """Check a pre-brake state; simulation time and exposure share one clock."""
        stamp=reference.start_s+float(elapsed)
        delta=(self.rotation@reference.velocity_world)*(stamp-reference.capture_s)
        self.mirror.set_state(q,self.width)
        hit=first_moving_target_hit(self.mirror,self.points,self.tree,delta,
                                    contact=stamp>=reference.contact_s-1e-8)
        if hit is None:return None
        geom,ids=hit
        return {'kind':'predicted_moving_target_cloud',
            'geometry':self.mirror.model.geom(geom).name,'hit_count':len(ids),
            'margin_m':.004,'elapsed_s':float(elapsed),'sim_time_s':stamp,
            'contact_allowed':bool(stamp>=reference.contact_s-1e-8),
            'source':'rendered_rgbd_visual_target_cloud_not_simulator_object_mesh'}
