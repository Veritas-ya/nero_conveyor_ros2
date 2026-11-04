/* Batched narrow phase for the original observed-point convex hulls.
 *
 * The 0/1 outputs have a floating-point separation band. A query containing
 * a 2 must be repeated by the caller's original NumPy implementation. This
 * keeps exact-face and cancellation cases under the original calculation.
 * No points, faces, margins, or KD-tree neighbours are approximated here.
 */
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>

void nero_observed_convex_batch(
    const double *points, const int64_t *indices, const int64_t *offsets,
    const double *positions, const double *rotations,
    const double *planes, const int64_t *face_offsets,
    const double *margins, const double *bounds,
    const size_t queries, const size_t geometries, uint8_t *inside,
    uint8_t *uncertain)
{
    for (size_t query = 0; query < queries; ++query) {
        const size_t geometry = query % geometries;
        const double *position = positions + 3*query;
        const double *rotation = rotations + 9*query;
        const double *box = bounds + 6*geometry;
        const double margin = margins[geometry];
        uncertain[query] = 0;
        for (int64_t at = offsets[query]; at < offsets[query+1]; ++at) {
            const double *point = points + 3*indices[at];
            const double dx = point[0]-position[0];
            const double dy = point[1]-position[1];
            const double dz = point[2]-position[2];
            const double local[3] = {
                dx*rotation[0]+dy*rotation[3]+dz*rotation[6],
                dx*rotation[1]+dy*rotation[4]+dz*rotation[7],
                dx*rotation[2]+dy*rotation[5]+dz*rotation[8]};
            /* This deliberately includes absolute world coordinates, not
             * only the already-cancelled local point. Non-unit plane normals
             * are included below as well. The band decides fallback only;
             * it never enlarges the final collision hull. */
            const double coordinate_scale = 1.0 + fabs(point[0])+fabs(point[1])+
                fabs(point[2])+fabs(position[0])+fabs(position[1])+fabs(position[2]);
            const double coordinate_error = 128*DBL_EPSILON*coordinate_scale*
                (1.0+fabs(rotation[0])+fabs(rotation[1])+fabs(rotation[2])+
                 fabs(rotation[3])+fabs(rotation[4])+fabs(rotation[5])+
                 fabs(rotation[6])+fabs(rotation[7])+fabs(rotation[8]));
            int boundary = 0, outside = 0;
            for (size_t axis = 0; axis < 3; ++axis) {
                const double low = local[axis]-box[axis];
                const double high = box[3+axis]-local[axis];
                const double band = coordinate_error + 128*DBL_EPSILON*
                    (1.0+fabs(local[axis])+fabs(box[axis])+fabs(box[3+axis]));
                if (low < -band || high < -band) { outside = 1; break; }
                if (low <= band || high <= band) boundary = 1;
            }
            if (!outside) {
                for (int64_t face = face_offsets[geometry];
                     face < face_offsets[geometry+1]; ++face) {
                    const double *plane = planes + 4*face;
                    const double value = local[0]*plane[0]+local[1]*plane[1]+
                        local[2]*plane[2]+plane[3]-margin;
                    const double normal_scale = fabs(plane[0])+fabs(plane[1])+fabs(plane[2]);
                    const double arithmetic_scale = 1.0+fabs(local[0]*plane[0])+
                        fabs(local[1]*plane[1])+fabs(local[2]*plane[2])+
                        fabs(plane[3])+fabs(margin);
                    const double band = coordinate_error*normal_scale+
                        128*DBL_EPSILON*arithmetic_scale;
                    if (!isfinite(value) || !isfinite(band)) { boundary = 1; continue; }
                    if (value > band) { outside = 1; break; }
                    if (value >= -band) boundary = 1;
                }
            }
            inside[at] = outside ? 0 : (boundary ? 2 : 1);
            if (inside[at] == 2) uncertain[query] = 1;
        }
    }
}
