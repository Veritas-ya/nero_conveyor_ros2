// Thin C ABI over unmodified official VAMP/CAPT headers vendored in vendor/.
// Only conservative selected-target sphere pruning; final convex semantics
// stay in RobotModel. See vendor/vamp_capt/SOURCE.json for upstream hashes.
#include <vamp/collision/capt.hh>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

struct CaptHandle {
    std::unique_ptr<vamp::collision::CAPT> tree;
    double coordinate_scale;
    double point_quantization;
    float radius_max;
};

extern "C" {

void *nero_official_capt_create(const double *points, std::size_t n,
                                double maximum_radius, double *metadata) {
    try {
        if (!n || !std::isfinite(maximum_radius) || maximum_radius <= 0.) return nullptr;
        auto handle=std::make_unique<CaptHandle>();
        handle->coordinate_scale=1.;
        handle->point_quantization=0.;
        std::vector<vamp::collision::Point> cloud;
        cloud.reserve(n);
        for (std::size_t i=0; i<n; ++i) {
            vamp::collision::Point p{};
            double error=0.;
            for (std::size_t axis=0; axis<3; ++axis) {
                double original=points[i*3+axis];
                p[axis]=static_cast<float>(original);
                if (!std::isfinite(p[axis])) return nullptr;
                error += std::abs(original-static_cast<double>(p[axis]));
                handle->coordinate_scale=std::max(handle->coordinate_scale,std::abs(original));
            }
            handle->point_quantization=std::max(handle->point_quantization,error);
            cloud.push_back(p);
        }
        // The radius range is a CAPT construction contract, not an actual
        // collision margin. A query outside it is conservatively undecided.
        handle->radius_max=std::nextafter(static_cast<float>(maximum_radius+.001),
                                         std::numeric_limits<float>::infinity());
        float point_padding=std::nextafter(static_cast<float>(handle->point_quantization),
                                          std::numeric_limits<float>::infinity());
        handle->tree=std::make_unique<vamp::collision::CAPT>(cloud,0.f,
                                                           handle->radius_max,point_padding);
        if (!handle->tree->is_valid()) return nullptr;
        metadata[0]=handle->point_quantization;
        metadata[1]=point_padding;
        metadata[2]=handle->radius_max;
        metadata[3]=vamp::collision::CAPT::FVectorT::num_scalars;
        return handle.release();
    } catch (...) { return nullptr; }
}

void nero_official_capt_destroy(void *opaque) {
    delete static_cast<CaptHandle *>(opaque);
}

void nero_official_capt_query(void *opaque, const double *centers,
                              const double *radii, std::size_t n,
                              uint8_t *possible, double *maximum_expansion) {
    auto *handle=static_cast<CaptHandle *>(opaque);
    double max_expansion=0.;
    for (std::size_t i=0; i<n; ++i) {
        vamp::collision::Point center{};
        double center_error=0., scale=handle->coordinate_scale;
        for (std::size_t axis=0; axis<3; ++axis) {
            double original=centers[i*3+axis];
            center[axis]=static_cast<float>(original);
            center_error += std::abs(original-static_cast<double>(center[axis]));
            scale=std::max(scale,std::abs(original));
        }
        double radius=radii[i];
        // Triangle inequality covers double->float centers/points. The
        // outward allowance also encloses float subtraction/squared-distance
        // and internal CAPT cell tests; this is *only* a coarse envelope.
        double expansion=center_error+128.*std::numeric_limits<float>::epsilon()
                                         *(1.+scale+std::abs(radius));
        float query=std::nextafter(static_cast<float>(radius+expansion),
                                  std::numeric_limits<float>::infinity());
        max_expansion=std::max(max_expansion,static_cast<double>(query)-radius);
        if (!std::isfinite(query) || query<0.f || query>handle->radius_max
                || !std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2])) {
            possible[i]=1; // Unknown range retains the original narrow phase.
        } else {
            possible[i]=handle->tree->collides(center,query) ? 1 : 0;
        }
    }
    *maximum_expansion=max_expansion;
}

}
