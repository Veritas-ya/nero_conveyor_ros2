from dataclasses import replace
from itertools import product

import mujoco
import numpy as np
import pytest
from scipy.optimize import lsq_linear, minimize

from mujoco_conveyor_sorting.controllers.segment_certificate import (
    AdaptiveSegmentValidator, CollisionPair, ConservativeSegmentCertificate,
    MujocoMotionBounds, SegmentConstraints, SegmentSample,
    convex_point_clearance_lower_bound, native_collision_pairs,
    native_separation_lower_bound,
)


def convex_vertices_world(model, data, geom):
    """Explicit compiled vertices for the test oracle, independent of support code."""
    kind = int(model.geom_type[geom])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        local = np.asarray(list(product((-1., 1.), repeat=3))) * model.geom_size[geom]
    elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[start:start+count], dtype=float)
    else:
        raise ValueError('oracle supports only boxes and convex mesh hulls')
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def convex_distance_oracle(model, data, first, second):
    """Solve convex closest points and independently bound its optimisation error."""
    a, b = convex_vertices_world(model, data, first), convex_vertices_world(model, data, second)
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    if int(model.geom_type[first]) == box and int(model.geom_type[second]) == box:
        ra, rb = (data.geom_xmat[g].reshape(3, 3) * model.geom_size[g]
                  for g in (first, second))
        matrix = np.column_stack((ra, -rb))
        offset = data.geom_xpos[second] - data.geom_xpos[first]
        solution = lsq_linear(matrix, offset, bounds=(-1., 1.), tol=1e-13, max_iter=300)
        assert solution.success and solution.optimality < 1e-9
        delta = matrix @ solution.x - offset
    else:
        # Convex combinations of compiled vertices represent the exact hulls.
        n, m = len(a), len(b)
        matrix = np.column_stack((a.T, -b.T))
        def objective(weights):
            delta = matrix @ weights
            return .5 * float(delta @ delta)
        def gradient(weights):
            return matrix.T @ (matrix @ weights)
        constraints = ({'type':'eq', 'fun':lambda w: np.sum(w[:n])-1.,
                        'jac':lambda w: np.r_[np.ones(n), np.zeros(m)]},
                       {'type':'eq', 'fun':lambda w: np.sum(w[n:])-1.,
                        'jac':lambda w: np.r_[np.zeros(n), np.ones(m)]})
        solution = minimize(objective, np.r_[np.full(n, 1./n), np.full(m, 1./m)],
                            jac=gradient, bounds=[(0., 1.)]*(n+m), constraints=constraints,
                            method='SLSQP', options={'ftol':1e-14, 'maxiter':500})
        assert solution.success
        assert abs(np.sum(solution.x[:n])-1.) < 1e-9
        assert abs(np.sum(solution.x[n:])-1.) < 1e-9
        delta = matrix @ solution.x
    upper = float(np.linalg.norm(delta))
    if upper < 1e-9:
        return 0.
    # A separating plane gives a lower bound; feasible points give an upper
    # bound. Refuse to use the result when their numerical gap is too large.
    direction = delta / upper
    lower = max(0., float(np.min(a @ direction) - np.max(b @ direction)))
    assert upper-lower < 2e-7, (upper, lower)
    return upper


def arm_model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body pos="0 0 .4"><joint name="a" axis="0 0 1" pos=".02 .01 0"/>
        <geom type="box" size=".05 .03 .04" pos=".10 0 0"/>
        <body pos=".25 .01 .04" euler="0 .2 0">
          <joint name="b" axis="0 1 0" pos=".01 0 .02"/>
          <geom type="capsule" size=".025 .10" pos=".08 .01 .05"/>
          <body pos=".20 0 .02"><joint name="jaw" type="slide"
             axis="0 1 0" range="-.2 .2" limited="true"/>
            <geom type="box" size=".01 .03 .06" pos="0 .01 0"/>
            <site name="tcp" pos=".04 0 -.10"/>
          </body>
        </body>
    </body></worldbody></mujoco>""")


def test_independent_convex_oracle_handles_known_separation_and_overlap():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      <body pos="1 0 0"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert convex_distance_oracle(model, data, 0, 1) == pytest.approx(.8, abs=1e-8)
    data.qpos[7] = .1
    mujoco.mj_forward(model, data)
    assert convex_distance_oracle(model, data, 0, 1) == pytest.approx(0., abs=2e-7)


def test_native_separation_remains_conservative_with_bad_mujoco_distance(monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body><freejoint/><geom type="box" size=".1 .06 .04"/></body>
      <body><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>""")
    data, rng = mujoco.MjData(model), np.random.default_rng(2467)
    def misleading_distance(model, data, first, second, cutoff, witness):
        witness[:] = [0., 0., 0., 1., 2., 3.]
        return 0.
    monkeypatch.setattr(mujoco, 'mj_geomDistance', misleading_distance)
    overlap = separated = 0
    for i in range(120):
        data.qpos[:3] = rng.uniform(-.04, .04, 3)
        q = rng.normal(size=4)
        data.qpos[3:7] = q/np.linalg.norm(q)
        data.qpos[7:10] = ([.01, 0., 0.] if i % 2 == 0 else [.55, .25, .2])
        mujoco.mj_forward(model, data)
        distance = convex_distance_oracle(model, data, 0, 1)
        bound = native_separation_lower_bound(model, data, 0, 1, margin_m=0.)
        assert bound <= distance+1e-7
        if distance < 1e-7:
            overlap += 1
            assert bound <= 0.
        else:
            separated += 1
    assert overlap == separated == 60


def sample_for(model, data, q, bounds, *, cloud=None, snapshot="fixed"):
    data.qpos[:2] = q
    mujoco.mj_forward(model, data)
    jpos, jrot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jpos, jrot, bounds.site_id)
    j = np.vstack((jpos[:, :2], .3*jrot[:, :2]))
    s = np.linalg.svd(j, compute_uv=False)
    clear = {}
    if cloud is not None:
        # Enclosing spheres give a simple, deliberately conservative distance.
        for g in range(model.ngeom):
            clear[str(g)] = float(np.min(np.linalg.norm(
                cloud-data.geom_xpos[g], axis=1))-model.geom_rbound[g])
    return SegmentSample(np.array(q), True, clear,
        float(data.site_xpos[bounds.site_id, 2]),
        -float(data.site_xmat[bounds.site_id].reshape(3, 3)[2, 2]),
        float(s[-1]), float(s[0]), data.site_xpos[bounds.site_id].copy(),
        data.xanchor[bounds.joint_ids].copy(), snapshot, data.geom_xpos.copy())


def test_kinematic_bound_encloses_all_geometry_vertices_across_random_segments():
    model, rng = arm_model(), np.random.default_rng(3211)
    data = mujoco.MjData(model)
    data.qpos[2] = .17  # Fixed jaw differs from qpos0, included in the bound.
    bounds = MujocoMotionBounds(model, [0, 1], 0, full_qpos=data.qpos)
    for _ in range(45):
        start, end = rng.uniform(-2, 2, (2, 2))
        data.qpos[:2] = start
        mujoco.mj_forward(model, data)
        # Corners of the rbound cube lie *outside* the sphere.  Use deterministic
        # sphere surface points, which cover every native collision shape.
        vectors = rng.normal(size=(model.ngeom, 30, 3))
        vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
        vectors *= model.geom_rbound[:, None, None]
        base = data.geom_xpos[:, None]+np.einsum("gij,gkj->gki",
            data.geom_xmat.reshape(-1, 3, 3), vectors)
        for fraction in np.linspace(0, 1, 17):
            data.qpos[:2] = start+(end-start)*fraction
            mujoco.mj_forward(model, data)
            points = data.geom_xpos[:, None]+np.einsum("gij,gkj->gki",
                data.geom_xmat.reshape(-1, 3, 3), vectors)
            actual = np.max(np.linalg.norm(points-base, axis=2), axis=1)
            assert np.all(actual <= bounds.geom_displacement(
                (end-start)*fraction)+1e-12)


def test_jacobian_perturbation_and_tcp_bounds_cover_interiors():
    model, rng = arm_model(), np.random.default_rng(888)
    data = mujoco.MjData(model)
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    for _ in range(35):
        q = rng.uniform(-2, 2, 2)
        dq = rng.uniform(-.15, .15, 2)
        a = sample_for(model, data, q, bounds)
        j0p, j0r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, j0p, j0r, 0)
        j0 = np.vstack((j0p[:, :2], .3*j0r[:, :2]))
        for t in np.linspace(0, 1, 13):
            b = sample_for(model, data, q+dq*t, bounds)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, 0)
            j = np.vstack((jp[:, :2], .3*jr[:, :2]))
            epsilon = bounds.jacobian_perturbation(a, dq*t, .3)
            assert np.linalg.norm(j-j0, ord=2) <= epsilon+1e-12
            assert np.linalg.norm(b.tcp_position-a.tcp_position) <= (
                bounds.tcp_displacement(dq*t)+1e-12)
            assert abs(b.sigma_min-a.sigma_min) <= epsilon+1e-12
            assert abs(b.sigma_max-a.sigma_max) <= epsilon+1e-12


def test_fixed_slide_reference_offset_is_included_in_hinge_radius():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint axis="0 0 1"/>
      <geom size=".01"/><body pos=".01 0 0">
        <joint type="slide" axis="1 0 0" ref="-3"/>
        <geom size=".01"/><site pos=".01 0 0"/>
      </body></body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[:] = [0., 0.]  # Actual fixed displacement is +3 m, not zero.
    bounds = MujocoMotionBounds(model, [0], 0, full_qpos=data.qpos)
    mujoco.mj_forward(model, data)
    start = data.geom_xpos[1].copy()
    data.qpos[0] = .1
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.geom_xpos[1]-start) <= (
        bounds.geom_displacement(np.array([.1]))[1]+1e-12)


class SlideBounds:
    def geom_displacement(self, dq):
        return np.array([abs(float(dq[0]))])

    def tcp_displacement(self, dq):
        return abs(float(dq[0]))

    def pair_displacement(self, first, second, dq):
        return 2*abs(float(dq[0]))

    def rotation_displacement(self, dq):
        return 0.0

    def jacobian_perturbation(self, sample, dq, scale):
        return 0.0


def slide_sample(q, obstacle=2.0):
    distance = abs(float(q[0])-obstacle)-.0002
    return SegmentSample(np.asarray(q), distance > 0, {"wall": distance},
        2.0, 1.0, 1.0, 1.0, np.array([q[0], 0, 2]),
        np.array([[0., 0., 0.]]), "same frozen scene")


def slide_certificate(**kwargs):
    return ConservativeSegmentCertificate(SlideBounds(), SegmentConstraints(
        minimum_tcp_height_m=0.0, minimum_downward_alignment=.5,
        maximum_jacobian_condition=100., collision_pairs=(
            CollisionPair("wall", 0, **kwargs),)))


def adaptive(evaluate=slide_sample, certify=None, **kwargs):
    return AdaptiveSegmentValidator(evaluate, certify or slide_certificate(),
        lower=np.array([0.]), upper=np.array([1.]), resolution=.001, **kwargs)


def test_clear_far_segment_uses_two_samples_instead_of_thousand():
    check = adaptive()
    assert check.validate(np.array([0.]), np.array([1.]))
    assert check.stats.samples == 2
    assert check.stats.certified_segments == 1
    assert check.stats.dense_fallback_segments == 0


def test_recursive_certificate_finds_interior_obstacle_between_safe_endpoints():
    check = adaptive(evaluate=lambda q: slide_sample(q, obstacle=.371))
    assert not check.validate(np.array([0.]), np.array([1.]))
    assert 2 < check.stats.samples < 60
    assert check.stats.invalid_samples == 1
    assert check.stats.certified_segments > 0


def test_no_certificate_preserves_exact_original_grid_and_checks_every_point():
    visited = []

    def evaluate(q):
        visited.append(float(q[0]))
        return slide_sample(q)

    check = adaptive(evaluate=evaluate, certify=lambda a, b: False)
    assert check.validate(np.array([0.]), np.array([1.]))
    assert check.stats.samples == 1001
    assert np.array_equal(np.sort(visited), np.arange(1001)/1000)
    assert check.stats.dense_fallback_segments == 1000


def test_adjacent_original_grid_states_do_not_need_an_extra_clearance_certificate():
    visited = []

    def evaluate(q):
        visited.append(q.copy())
        return slide_sample(q)

    def unnecessary_certificate(_a, _b):
        raise AssertionError("no interior grid sample can be skipped")

    check = adaptive(evaluate=evaluate, certify=unnecessary_certificate)
    start, end = np.array([.4]), np.array([.4005])
    assert check.validate(start, end)
    np.testing.assert_array_equal(visited, [start, end])
    assert check.stats.certified_segments == 0
    assert check.stats.dense_fallback_segments == 1
    assert check.stats.endpoint_only_segments == 1


def test_adjacent_original_grid_still_rejects_invalid_endpoint():
    def evaluate(q):
        return slide_sample(q, obstacle=.4005)

    check = adaptive(evaluate=evaluate)
    assert not check.validate(np.array([.4]), np.array([.4005]))
    assert check.stats.invalid_samples == 1
    assert check.stats.endpoint_only_segments == 0


def test_adjacent_original_grid_cancellation_after_last_state_still_rejects():
    visited = []

    def evaluate(q):
        visited.append(q.copy())
        return slide_sample(q)

    check = adaptive(evaluate=evaluate, cancelled=lambda: len(visited) >= 2)
    assert not check.validate(np.array([.4]), np.array([.4005]))


def test_missing_collision_constraint_cannot_be_certified():
    check = adaptive(evaluate=lambda q: replace(slide_sample(q), clearances_m={}))
    assert check.validate(np.array([0.]), np.array([1.]))
    assert check.stats.samples == 1001
    assert check.stats.certified_segments == 0


def test_snapshot_change_cannot_share_segment_proof():
    a, b = slide_sample(np.array([0.])), slide_sample(np.array([1.]))
    assert not slide_certificate()(a, replace(b, snapshot="new observation"))


def test_obstacle_motion_bound_is_not_halved_without_time_bound():
    a, b = slide_sample(np.array([0.]), 1.25), slide_sample(np.array([.1]), 1.25)
    assert slide_certificate()(a, b)
    assert not slide_certificate(obstacle_displacement_m=1.2)(a, b)


@pytest.mark.parametrize("changes", [
    {"tcp_height_m": .01}, {"downward_alignment": .49},
    {"sigma_min": 0.}, {"sigma_min": .001, "sigma_max": 1.},
])
def test_noncollision_constraints_must_each_have_a_proof(changes):
    a, b = slide_sample(np.array([0.])), slide_sample(np.array([.1]))
    assert not slide_certificate()(replace(a, **changes), replace(b, **changes))


def test_infinite_clearance_for_empty_world_is_supported():
    a, b = slide_sample(np.array([0.])), slide_sample(np.array([.1]))
    assert slide_certificate()(replace(a, clearances_m={"wall": np.inf}),
                               replace(b, clearances_m={"wall": np.inf}))


def test_cancelled_during_evaluation_cannot_accept_certificate():
    calls = []

    def evaluate(q):
        calls.append(q)
        return slide_sample(q)

    check = adaptive(evaluate=evaluate, cancelled=lambda: len(calls) > 0)
    assert not check.validate(np.array([0.]), np.array([1.]))
    assert check.stats.cancellations == 1


def test_original_endpoint_array_is_checked_without_reconstruction_rounding():
    visited = []

    def evaluate(q):
        visited.append(q.copy())
        return replace(slide_sample(q), valid=float(q[0]) != -.2)

    check = AdaptiveSegmentValidator(evaluate, lambda a, b: False,
        lower=np.array([-1.]), upper=np.array([1.]), resolution=.1)
    assert not check.validate(np.array([.7]), np.array([-.2]))
    assert visited[-1][0] == -.2


def test_certified_regions_skip_only_proven_boxes_and_keep_boundaries_inward():
    from mujoco_conveyor_sorting.controllers.planning_segment_validation import PlanningSegmentValidator

    owner = object.__new__(PlanningSegmentValidator)
    owner._proof_boxes, owner._proof_box_array = [], None
    sample = slide_sample(np.array([.3]))
    owner._remember_region(sample, np.array([.5]))
    assert owner._known_region(np.array([0.]), np.array([.6]))
    assert not owner._known_region(np.array([-.2]), np.array([.6]))
    assert not owner._known_region(np.array([0.]), np.array([.9]))
    visits = []

    def evaluate(q):
        visits.append(q)
        return slide_sample(q)

    check = adaptive(evaluate=evaluate, certified_region=owner._known_region)
    assert check.validate(np.array([0.]), np.array([.6]))
    assert not visits
    assert check.stats.reused_regions == 1


def test_point_halfspace_clearance_is_a_lower_bound_even_outside_corner():
    equations = np.array([[1, 0, 0, -1], [-1, 0, 0, -1],
                          [0, 1, 0, -1], [0, -1, 0, -1],
                          [0, 0, 1, -1], [0, 0, -1, -1]], dtype=float)
    cloud = np.array([[2., 2., 0.], [1.5, 0, 0]])
    lower = convex_point_clearance_lower_bound(cloud, equations, margin_m=.1)
    assert .399999 < lower <= .4
    assert lower <= min(np.linalg.norm(np.maximum(np.abs(p)-1.1, 0)) for p in cloud)
    assert convex_point_clearance_lower_bound(np.empty((0, 3)), equations) == np.inf


def test_nonunit_plane_margin_preserves_original_unnormalized_inequality():
    # .5*x <= .1 means x <= .2, rather than x <= .1.
    equations = np.array([[.5, 0., 0., 0.]])
    lower = convex_point_clearance_lower_bound(np.array([[.25, 0., 0.]]),
                                               equations, margin_m=.1)
    assert .049999999 < lower <= .05


def test_expanded_convex_radius_is_used_for_rotating_padding():
    model = arm_model()
    raw = MujocoMotionBounds(model, [0, 1], 0)
    padded = MujocoMotionBounds(model, [0, 1], 0,
        geom_radii_m={2: model.geom_rbound[2]+.2})
    assert padded.geom_displacement(np.array([.1, .1]))[2] > (
        raw.geom_displacement(np.array([.1, .1]))[2])
    clamped = MujocoMotionBounds(model, [0, 1], 0, geom_radii_m={2: 0.})
    assert np.array_equal(clamped.geom_coefficients, raw.geom_coefficients)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MujocoMotionBounds(model, [0, 1], 0, geom_radii_m={2: -1.})


def test_pair_bound_removes_shared_rigid_ancestor_motion():
    model = arm_model()
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    # Geoms 1 and 2 are both downstream of hinge b; moving a and b cannot
    # change their mutual separation, since the jaw joint is fixed.
    assert bounds.pair_displacement(1, 2, np.array([2., 2.])) == 0.
    assert bounds.pair_displacement(0, 2, np.array([2., 0.])) == 0.
    assert bounds.pair_displacement(0, 2, np.array([0., .2])) > 0.


def test_relative_pair_bound_covers_distance_changes_at_interiors():
    model, rng = arm_model(), np.random.default_rng(91112)
    data = mujoco.MjData(model)
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    for _ in range(30):
        start, end = rng.uniform(-1., 1., (2, 2))
        data.qpos[:2] = start
        mujoco.mj_forward(model, data)
        original = convex_distance_oracle(model, data, 0, 2)
        for t in np.linspace(0, 1, 17):
            data.qpos[:2] = start+(end-start)*t
            mujoco.mj_forward(model, data)
            distance = convex_distance_oracle(model, data, 0, 2)
            # Each independent optimisation is verified against its own
            # support-plane bound to 0.2 micrometres.
            assert abs(distance-original) <= bounds.pair_displacement(
                0, 2, (end-start)*t)+5e-7


def test_certified_random_segments_satisfy_dense_kinematic_oracle():
    model, rng = arm_model(), np.random.default_rng(18921)
    data = mujoco.MjData(model)
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    constraints = SegmentConstraints(-1., -1.1, 1000.,
        tuple(CollisionPair(str(g), g) for g in range(model.ngeom)))
    certify = ConservativeSegmentCertificate(bounds, constraints)
    cloud = np.array([[2., 2., 2.], [-2., 2., 2.]])
    accepted = 0
    for _ in range(50):
        q0 = rng.uniform(-1, 1, 2)
        q1 = q0+rng.uniform(-.02, .02, 2)
        a, b = (sample_for(model, data, q, bounds, cloud=cloud) for q in (q0, q1))
        if not certify(a, b):
            continue
        accepted += 1
        for t in np.linspace(0, 1, 51):
            s = sample_for(model, data, q0+(q1-q0)*t, bounds, cloud=cloud)
            assert min(s.clearances_m.values()) > 0
            assert s.tcp_height_m >= constraints.minimum_tcp_height_m
            assert s.downward_alignment >= constraints.minimum_downward_alignment
            assert s.sigma_max/s.sigma_min <= constraints.maximum_jacobian_condition
    assert accepted >= 40


def pair_test_model(filterparent="enable"):
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option><flag filterparent="{filterparent}"/></option><worldbody>
      <geom name="world" type="plane" size="1 1 .1"/>
      <body name="robot_a" pos="0 0 .1"><joint name="a" axis="0 0 1"/>
        <geom name="a1" size=".1" contype="2" conaffinity="2"/>
        <geom name="a2" size=".11" contype="0" conaffinity="0"/>
        <body name="robot_fixed" pos=".02 0 0"><geom name="fixed" size=".1"/>
          <body name="robot_b" pos=".01 0 0"><joint name="b" axis="0 1 0"/>
            <geom name="b" size=".1"/>
            <body name="robot_c" pos=".01 0 0"><joint name="c" axis="1 0 0"/>
              <geom name="c" size=".1"/>
            </body>
          </body>
        </body>
      </body>
      <body name="obstacle" pos=".08 0 .1"><geom name="obstacle" size=".1"/>
        <body name="obstacle_weld" pos=".01 0 0"><geom name="obstacle_weld" size=".1"/></body>
      </body>
      <body name="object_part" pos=".01 0 .1"><freejoint/>
        <geom name="object_part" size=".1"/>
      </body>
    </worldbody><contact>
      <exclude body1="robot_a" body2="robot_c"/>
      <pair geom1="a2" geom2="c"/>
      <pair geom1="a1" geom2="b"/>
    </contact></mujoco>""")


@pytest.mark.parametrize("filterparent", ["enable", "disable"])
def test_native_pair_list_covers_every_actual_worker_contact(filterparent):
    model = pair_test_model(filterparent)
    robot_ids = [b for b in range(model.nbody)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("robot_")]
    pairs = set(native_collision_pairs(model, robot_ids))
    data, rng = mujoco.MjData(model), np.random.default_rng(1939)
    for _ in range(150):
        data.qpos[:3] = rng.uniform(-np.pi, np.pi, 3)
        mujoco.mj_forward(model, data)
        for contact in data.contact[:data.ncon]:
            a, b = sorted((int(contact.geom1), int(contact.geom2)))
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            if ba not in robot_ids and bb not in robot_ids:
                continue
            other = bb if ba in robot_ids else ba
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if name.startswith("object_"):
                continue
            assert (a, b) in pairs


def test_explicit_pairs_override_masks_excludes_and_parent_filter():
    model = pair_test_model()
    robots = [model.body(name).id for name in
              ("robot_a", "robot_fixed", "robot_b", "robot_c")]
    pairs = set(native_collision_pairs(model, robots))
    for names in (("a2", "c"), ("a1", "b")):
        pair = tuple(sorted(model.geom(n).id for n in names))
        assert pair in pairs
    assert tuple(sorted(model.geom(n).id for n in ("a1", "c"))) not in pairs
    assert tuple(sorted(model.geom(n).id for n in ("a1", "fixed"))) not in pairs


@pytest.mark.parametrize("kind,size", [
    ("sphere", ".1"), ("capsule", ".1 .1"), ("box", ".1 .06 .04"),
    ("ellipsoid", ".1 .06 .04"), ("cylinder", ".1 .1"),
])
def test_native_support_distance_never_exceeds_exact_separation(kind, size):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody><geom type="plane" size="1 1 .1"/>
      <body pos=".03 0 .5"><freejoint/><geom type="{kind}" size="{size}"/>
      </body><body pos=".4 0 .5"><freejoint/>
        <geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>""")
    data, rng = mujoco.MjData(model), np.random.default_rng(9211)
    for _ in range(60):
        data.qpos[:3] = rng.uniform([-.2, -.2, .3], [.2, .2, .7])
        quaternion = rng.normal(size=4)
        data.qpos[3:7] = quaternion/np.linalg.norm(quaternion)
        mujoco.mj_forward(model, data)
        for a, b in ((0, 1), (1, 2)):
            if a == 1 and kind == 'box':
                exact = convex_distance_oracle(model, data, a, b)
            else:
                exact = mujoco.mj_geomDistance(model, data, a, b, 10., None)
            lower = native_separation_lower_bound(model, data, a, b, margin_m=0.)
            if exact >= 0:
                assert lower <= exact+1e-7
            else:
                assert lower <= 0


def test_native_mesh_support_uses_compiled_mesh_coordinates():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><asset><mesh name="skew" vertex="
      0 0 0  .3 0 0  0 .2 0  0 0 .1  .15 .1 .1"/></asset>
      <worldbody><body pos=".5 0 .3"><freejoint/>
        <geom type="mesh" mesh="skew" euler=".2 .3 .4"/>
      </body><body pos="0 0 .3"><freejoint/>
        <geom type="box" size=".05 .05 .05"/></body></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(11002)
    for _ in range(60):
        data.qpos[:3] = rng.uniform([.3, -.1, .2], [.8, .1, .6])
        q = rng.normal(size=4)
        data.qpos[3:7] = q/np.linalg.norm(q)
        mujoco.mj_forward(model, data)
        exact = convex_distance_oracle(model, data, 0, 1)
        lower = native_separation_lower_bound(model, data, 0, 1, margin_m=0.)
        assert lower <= max(0., exact)+1e-7
        assert lower > 0


def test_current_seven_axis_robot_complete_mesh_and_jacobian_motion_bounds():
    # Production geometry, private data only: no camera/CAN/device connection.
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel

    robot = RobotModel()
    model, data, rng = robot.model, robot.data, np.random.default_rng(75590)
    bounds = MujocoMotionBounds(model, robot.ids, robot.tcp_id,
        full_qpos=data.qpos, geom_radii_m=robot.collision_radii)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(18):
        q0 = rng.uniform(robot.command_lower, robot.command_upper)
        dq = rng.uniform(-.1, .1, 7)
        robot.set_state(q0, .1)
        base = {g: data.geom_xpos[g]+v@data.geom_xmat[g].reshape(3, 3).T
                for g, v in robot.mesh_vertices.items()}
        mujoco.mj_jacSite(model, data, jp, jr, robot.tcp_id)
        j0 = np.vstack((jp[:, robot.ik.dof_indices], .3*jr[:, robot.ik.dof_indices]))
        values = np.linalg.svd(j0, compute_uv=False)
        sample = SegmentSample(q0, True, {}, float(data.site_xpos[robot.tcp_id, 2]),
            0., float(values[-1]), float(values[0]),
            data.site_xpos[robot.tcp_id].copy(), data.xanchor[robot.ids].copy(),
            geom_centers=data.geom_xpos.copy())
        for t in np.linspace(0, 1, 9):
            robot.set_state(q0+dq*t, .1)
            movement = bounds.geom_displacement(dq*t)
            local_movement = bounds.local_geom_coefficients(sample, dq*t) @ np.abs(dq*t)
            assert np.all(local_movement <= movement+1e-12)
            for g, v in robot.mesh_vertices.items():
                current = data.geom_xpos[g]+v@data.geom_xmat[g].reshape(3, 3).T
                assert np.max(np.linalg.norm(current-base[g], axis=1)) <= movement[g]+1e-11
                assert np.max(np.linalg.norm(current-base[g], axis=1)) <= local_movement[g]+1e-11
            mujoco.mj_jacSite(model, data, jp, jr, robot.tcp_id)
            j1 = np.vstack((jp[:, robot.ik.dof_indices], .3*jr[:, robot.ik.dof_indices]))
            assert np.linalg.norm(j1-j0, ord=2) <= bounds.jacobian_perturbation(
                sample, dq*t, .3)+1e-11


@pytest.mark.parametrize('root_tilt', [0., .45])
def test_common_upstream_rotation_preserves_sigma_but_not_matrix_difference(root_tilt):
    model, rng = arm_model(), np.random.default_rng(430029)
    model.body_quat[1] = [np.cos(root_tilt/2), 0., np.sin(root_tilt/2), 0.]
    data = mujoco.MjData(model)
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    assert bounds.common_rigid_joint_indices == (0,)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(100):
        q = rng.uniform(-1., 1., 2)
        delta = np.array([rng.uniform(-2., 2.), 0.])
        a = sample_for(model, data, q, bounds)
        mujoco.mj_jacSite(model, data, jp, jr, 0)
        ja = np.vstack((jp[:, :2], .3*jr[:, :2]))
        b = sample_for(model, data, q+delta, bounds)
        mujoco.mj_jacSite(model, data, jp, jr, 0)
        jb = np.vstack((jp[:, :2], .3*jr[:, :2]))
        assert bounds.singular_value_perturbation(a, delta, .3) == 0.
        assert abs(a.sigma_min-b.sigma_min) < 1e-12
        assert abs(a.sigma_max-b.sigma_max) < 1e-12
        # Preserve the original *matrix* norm bound; it generally is nonzero.
        assert np.linalg.norm(jb-ja, ord=2) <= bounds.jacobian_perturbation(a, delta, .3)+1e-12
        assert abs(a.tcp_height_m-b.tcp_height_m) <= bounds.tcp_height_displacement(delta)+1e-12
        if root_tilt == 0.:
            assert bounds.tcp_height_displacement(delta) < 1e-12


@pytest.mark.parametrize('root_tilt', [0., .45])
def test_intrinsic_sigma_and_world_z_bounds_enclose_mixed_joint_interiors(root_tilt):
    model, rng = arm_model(), np.random.default_rng(93928)
    model.body_quat[1] = [np.cos(root_tilt/2), 0., np.sin(root_tilt/2), 0.]
    data = mujoco.MjData(model)
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    for _ in range(80):
        q = rng.uniform(-1., 1., 2)
        dq = np.array([rng.uniform(-2., 2.), rng.uniform(-.15, .15)])
        a = sample_for(model, data, q, bounds)
        tool_axis = rng.normal(size=3); tool_axis /= np.linalg.norm(tool_axis)
        alignment_a = -float((data.site_xmat[0].reshape(3, 3)@tool_axis)[2])
        for t in [.25, .5, .75, 1.]:
            b = sample_for(model, data, q+dq*t, bounds)
            sigma_bound = bounds.singular_value_perturbation(a, dq*t, .3)
            assert abs(a.sigma_min-b.sigma_min) <= sigma_bound+1e-12
            assert abs(a.sigma_max-b.sigma_max) <= sigma_bound+1e-12
            assert abs(a.tcp_height_m-b.tcp_height_m) <= bounds.tcp_height_displacement(dq*t)+1e-12
            alignment_b = -float((data.site_xmat[0].reshape(3, 3)@tool_axis)[2])
            assert abs(alignment_a-alignment_b) <= bounds.downward_alignment_displacement(dq*t)+1e-12


def test_common_slide_translation_preserves_sigma_and_horizontal_slide_height():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom size=".05"/><body pos=".2 0 0"><joint axis="0 1 0"/>
        <geom size=".05"/><site pos=".1 0 0"/>
      </body></body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    a = sample_for(model, data, np.array([0., .2]), bounds)
    b = sample_for(model, data, np.array([2., .2]), bounds)
    assert bounds.common_rigid_joint_indices == (0,)
    assert bounds.singular_value_perturbation(a, np.array([2., 0.]), .3) == 0.
    assert abs(a.sigma_min-b.sigma_min) < 1e-12
    assert abs(a.tcp_height_m-b.tcp_height_m) < 1e-12
    assert bounds.tcp_height_displacement(np.array([2., 0.])) < 1e-12


def test_separate_arm_branches_do_not_claim_a_common_rigid_joint():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint axis="0 0 1"/><geom size=".1"/>
      <site pos=".1 0 0"/></body><body><joint axis="0 1 0"/>
      <geom size=".1"/></body></worldbody></mujoco>""")
    bounds = MujocoMotionBounds(model, [0, 1], 0)
    assert bounds.common_rigid_joint_indices == ()
