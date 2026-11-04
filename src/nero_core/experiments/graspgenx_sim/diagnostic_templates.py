"""Explicit old grasp-template comparator; never a GraspGen-X fallback."""
from copy import deepcopy

from experiments.graspgenx_sim.sensor_control import SensorControlEnvironment


class DiagnosticTemplateEnvironment(SensorControlEnvironment):
    def _with_template(self, record):
        if record is None:
            return None
        result = deepcopy(record)
        # A visual classification indexes a static catalog. No runtime body,
        # true object pose, real category association or contact is consulted.
        spec = self._physics._object_spec_by_category.get(int(result["category_id"]))
        if spec is None:
            return result
        metadata = dict(spec.metadata)
        result.update(grasp_height=float(spec.grasp_height),
                      recommended_grip_width=float(spec.recommended_grip_width),
                      grip_width_margin=float(metadata.get("grip_width_margin", .040)),
                      symmetry=dict(spec.symmetry),
                      grasp_candidates=spec.as_dict()["grasp_candidates"],
                      grasp_yaw_mode=metadata.get("grasp_yaw_mode", "fixed"),
                      grasp_yaw_delta_range=list(metadata.get("grasp_yaw_delta_range", [])),
                      grasp_source="legacy_catalog_diagnostic_only")
        for key in ("contact_hold_squeeze_fraction", "contact_hold_max_relaxation_fraction",
                    "confirmed_grasp_settle_time", "closure_requires_arm_settle",
                    "close_lead_time", "close_settle_time", "max_close_time",
                    "closure_horizontal_tolerance", "closure_vertical_tolerance",
                    "closure_orientation_tolerance", "allow_predictive_closure_outside_window"):
            if key in metadata:
                result[key] = metadata[key]
        return result

    def get_observation(self, **kwargs):
        result = super().get_observation(**kwargs)
        result["objects"] = [self._with_template(x) for x in result["objects"]]
        return result

    def get_object_state(self, track_id, *args):
        return self._with_template(super().get_object_state(track_id, *args))

    get_target = get_object_state
    get_object = get_object_state

    def select_target(self, observation=None):
        return self._with_template(super().select_target(observation))

    select_sorting_target = select_target
