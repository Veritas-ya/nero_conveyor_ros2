"""Perception-side utilities for delayed dynamic-conveyor observations.

The package intentionally has no dependency on MuJoCo.  This keeps the truth
state producer (the environment) separate from the delayed observation and
prediction consumers used by controllers or learned policies.
"""

from .grasp_window import (
    DynamicGraspWindow,
    GraspWindow,
    GraspWindowCalculator,
    GraspWindowConfig,
)
from .motion_predictor import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    MotionSample,
    MotionModel,
    MotionPredictor,
    PredictionError,
    PredictionErrorStatistics,
    PredictionResult,
    SlidingWindowPredictor,
)
from .observation_buffer import (
    LatencyBreakdown,
    MultiObjectObservationBuffer,
    ObjectObservation,
    ObservationBuffer,
    TimestampedState,
)
from .rgbd_pose_provider import (
    PoseDetection,
    RGBDPoseProvider,
    RgbdPoseProvider,
    SegmentationInstance,
    bop_pose_to_world,
    bop_transform_to_world,
    opencv_camera_to_world,
)
from .rgbd_vision_pipeline import (
    ComponentMeasurement,
    DepthGeometryVisionPipeline,
    GeometryPrototype,
    VisionPoseDetection,
    connected_component_masks,
    deproject_depth_to_world,
)

__all__ = [
    "ConstantAccelerationPredictor",
    "ConstantVelocityPredictor",
    "ComponentMeasurement",
    "DepthGeometryVisionPipeline",
    "DynamicGraspWindow",
    "GraspWindow",
    "GraspWindowCalculator",
    "GraspWindowConfig",
    "GeometryPrototype",
    "MotionModel",
    "MotionSample",
    "MotionPredictor",
    "LatencyBreakdown",
    "MultiObjectObservationBuffer",
    "ObjectObservation",
    "ObservationBuffer",
    "PoseDetection",
    "PredictionError",
    "PredictionErrorStatistics",
    "PredictionResult",
    "RGBDPoseProvider",
    "RgbdPoseProvider",
    "SegmentationInstance",
    "SlidingWindowPredictor",
    "TimestampedState",
    "VisionPoseDetection",
    "bop_pose_to_world",
    "bop_transform_to_world",
    "connected_component_masks",
    "deproject_depth_to_world",
    "opencv_camera_to_world",
]
