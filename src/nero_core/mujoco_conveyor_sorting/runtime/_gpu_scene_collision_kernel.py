"""Optional Triton implementation; imported only for an actual CUDA request."""
import numpy as np
import torch
import triton
import triton.language as tl


class CudaObservedPointQueries:
    def __init__(self, points, equations):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.points = torch.tensor(np.asarray(points), dtype=torch.float64, device=self.device)
        counts = np.asarray([len(e) for e in equations], dtype=np.int32)
        begin = np.r_[0, np.cumsum(counts[:-1])].astype(np.int32)
        self.planes = torch.tensor(np.concatenate(equations), dtype=torch.float64, device=self.device)
        self.face_begin = torch.tensor(begin, device=self.device)
        self.face_count = torch.tensor(counts, device=self.device)
        self.geometry_count = len(equations)

    def candidates(self, positions, rotations, radii, margins, bounds, nearby=None):
        if not len(positions):
            return np.zeros(positions.shape[:2], dtype=bool)
        if nearby is None:
            raise ValueError("batched KD-tree point/body candidates are required")
        counts = np.fromiter((len(ids) for ids in nearby), dtype=np.int64)
        if not counts.sum():
            return np.zeros(positions.shape[:2], dtype=bool)
        bodies = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        point_ids = np.concatenate([ids for ids in nearby if len(ids)]).astype(np.int32)
        params = [torch.as_tensor(np.ascontiguousarray(a), device=self.device)
                  for a in (positions, rotations, margins, bounds, bodies, point_ids)]
        out = torch.zeros(positions.shape[:2], dtype=torch.int32, device=self.device)
        _nearby_convex_candidates[(triton.cdiv(len(point_ids), 128),)](
            self.points, *params, self.planes, self.face_begin, self.face_count, out,
            PAIRS=len(point_ids), G=self.geometry_count, BLOCK=128,
            num_warps=4, enable_fp_fusion=False)
        # Only B*G flags cross back, never B*G*N point masks.
        return out.cpu().numpy().astype(bool)


@triton.jit
def _nearby_convex_candidates(POINTS, POSITIONS, ROTATIONS, MARGINS, BOUNDS,
                              BODIES, POINT_IDS, PLANES, FACE_BEGIN, FACE_COUNT,
                              OUTPUT, PAIRS, G: tl.constexpr,
                              BLOCK: tl.constexpr):
    pair = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = pair < PAIRS
    body = tl.load(BODIES+pair, valid, other=0)
    point = tl.load(POINT_IDS+pair, valid, other=0)
    geometry = body % G
    x = tl.load(POINTS+3*point) - tl.load(POSITIONS+3*body)
    y = tl.load(POINTS+3*point+1) - tl.load(POSITIONS+3*body+1)
    z = tl.load(POINTS+3*point+2) - tl.load(POSITIONS+3*body+2)
    lx = x*tl.load(ROTATIONS+9*body) + y*tl.load(ROTATIONS+9*body+3) + z*tl.load(ROTATIONS+9*body+6)
    ly = x*tl.load(ROTATIONS+9*body+1) + y*tl.load(ROTATIONS+9*body+4) + z*tl.load(ROTATIONS+9*body+7)
    lz = x*tl.load(ROTATIONS+9*body+2) + y*tl.load(ROTATIONS+9*body+5) + z*tl.load(ROTATIONS+9*body+8)
    eps = 1.e-10*(1. + tl.abs(x) + tl.abs(y) + tl.abs(z))
    active = valid & (lx >= tl.load(BOUNDS+6*geometry)-eps) & (lx <= tl.load(BOUNDS+6*geometry+3)+eps)
    active &= (ly >= tl.load(BOUNDS+6*geometry+1)-eps) & (ly <= tl.load(BOUNDS+6*geometry+4)+eps)
    active &= (lz >= tl.load(BOUNDS+6*geometry+2)-eps) & (lz <= tl.load(BOUNDS+6*geometry+5)+eps)
    begin = tl.load(FACE_BEGIN+geometry)
    count = tl.load(FACE_COUNT+geometry)
    margin = tl.load(MARGINS+geometry)
    face = 0
    while tl.sum((active & (face < count)).to(tl.int32), 0) > 0:
        in_face = active & (face < count)
        offset = 4*(begin+face)
        a = tl.load(PLANES+offset, in_face, other=0.)
        b = tl.load(PLANES+offset+1, in_face, other=0.)
        c = tl.load(PLANES+offset+2, in_face, other=0.)
        d = tl.load(PLANES+offset+3, in_face, other=0.)
        value = lx*a + ly*b + lz*c + d
        face_eps = eps*(1. + tl.abs(a) + tl.abs(b) + tl.abs(c) + tl.abs(d) + tl.abs(margin))
        active &= (~in_face) | (value <= margin+face_eps)
        face += 1
    tl.atomic_or(OUTPUT+body, 1, mask=active, sem="relaxed")
