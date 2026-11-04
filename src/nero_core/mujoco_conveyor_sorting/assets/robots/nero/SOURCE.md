# AgileX NERO model provenance

The original NERO meshes and upstream robot descriptions in this directory come from
the official AgileX repository:

- Repository: <https://github.com/agilexrobotics/agx_arm_urdf>
- Revision: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- Upstream paths: `nero/meshes/*.stl` and `nero/urdf/*`
- License: MIT; the upstream license is preserved as `LICENSE`

`models/robot.xml` adapts the official URDF kinematic tree, inertial data,
and meshes to the combined MuJoCo conveyor scene. The joint position limits
were subsequently changed on 2026-11-03 to match the manual values supplied
by the user: J1 ±157°, J2 ±102°, J3 ±160°, J4 −60° to +125°, J5 ±160°,
J6 −44° to +57°, J7 ±97°. The manual version and page were not supplied;
these joint limits should not be attributed to the referenced URDF revision.
It also adds
simulation-specific actuators, collision/contact tuning, a task TCP site,
finger contact sites, and a wrist camera.

The official AgileX MuJoCo model was consulted as an implementation reference:

- <https://github.com/agilexrobotics/agx_arm_sim/tree/master/mujoco/agilex_arm/agilex_nero>
- Reference revision: `f8cd8b147c75d59e14f90fb0646770eefa268ed0`

No mesh or XML file from `agx_arm_sim` is redistributed here; that repository's
README states MIT, but its checked revision does not contain a standalone
license file.

## Gripper drawing adjustment, 2026-10-12

The user supplied a screenshot headed `8.7 夹爪尺寸图` on 2026-10-12 and
authorized updating the installed gripper geometry to its dimensions. The
screenshot gives an axial chain of 50.5 mm, 19 mm, and 73 mm, totaling
142.5 mm from the drawing's upper mounting face to the distal finger plane.
Its product revision and manufacturing tolerances are not stated.

The adjustment assumes that this upper mounting face coincides with
`gripper_base` local z = 0. The existing 6 mm `gripper_flange` to
`gripper_base` transform and the SDK flange convention are unchanged.

The upstream `gripper_base.stl` is retained unchanged. The separate derived
`gripper_base_drawing_20261012.stl` supplies both the visual and collision
geometry in `robot.xml`. The local script
`tools/update_gripper_drawing_mesh.py` reproducibly generates it from the
original mesh.
For original local z values at or below 46 mm, its axial coordinates are
scaled by 50.5 / 46; higher coordinates are translated by +4.5 mm. The
mapping is continuous at 46 mm and preserves the existing 19 mm lower
body section, whose distal plane moves from 65 mm to 69.5 mm. X and Y
coordinates are retained.

Both original finger meshes remain unchanged. Their rigid mounting origins
move from z = 138 mm to z = 142.5 mm; their local finger contact sites and
inertias keep the upstream finger convention. The task TCP and attachment
site move from z = 128 mm to z = 132.5 mm, retaining the existing design
offset of 10 mm behind the fingertips. This is a drawing-based nominal TCP,
not a physical TCP calibration. The wrist camera and its housing move
together with the lower body so their relative camera geometry is retained.

The drawing labels 102.5 mm and 110.5 mm are surface gaps at different
heights in the depicted open state. The upstream finger shape already has
the same 8 mm difference between distal and upper gaps: at a modeled
100 mm joint opening these gaps are 100 mm and 108 mm. The drawing does
not identify the corresponding SDK command or an opening zero offset.
These labels therefore do not change the control width limit, finger
travel, or X/Y geometry in this adjustment.

The base mass, center of mass, and inertia retain the upstream approximate
values. The screenshot does not supply mass distribution or inertial
measurements, so it cannot determine corrected physical values for these
properties. Drawing dimensions determine nominal geometry and do not
establish calibrated physical dynamics or verify a recorded motion on
hardware.
