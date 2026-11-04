/* Conservative batched robot/object broad phase.
 *
 * This is a small project kernel inspired by VAMP's structure-of-arrays,
 * multiple-configurations-per-SIMD-vector approach.  It is not the VAMP or
 * CAPT library.  Inputs/outputs are doubles; fast-math and FMA contraction
 * are deliberately disabled.  Surviving pairs require the original narrow
 * phase.  AABB distance is only a lower bound on convex-body clearance.
 */
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define AVX2_TARGET __attribute__((target("avx2")))
#define X86_SIMD 1
#else
#define AVX2_TARGET
#define X86_SIMD 0
#endif

int nero_simd_has_avx2(void) {
#if X86_SIMD
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return 0;
#endif
}

static double maximum(double a, double b) { return a > b ? a : b; }

static void boxes_scalar(const double *p, const double *r, const double *local,
                         size_t b, size_t g, double *out) {
    for (size_t k=0; k<g; ++k) for (size_t s=0; s<b; ++s) {
        const double *a=local+k*6;
        for (size_t axis=0; axis<3; ++axis) {
            double c=p[(k*3+axis)*b+s], e=0., scale=fabs(c);
            for (size_t j=0; j<3; ++j) {
                double m=r[(k*9+axis*3+j)*b+s];
                c += m*(a[j]+a[j+3])*.5;
                e += fabs(m)*(a[j+3]-a[j])*.5;
                scale += fabs(m)*fabs((a[j]+a[j+3])*.5);
            }
            double err=64.*DBL_EPSILON*(1.+scale+e);
            out[((k*2)*3+axis)*b+s]=c-e-err;
            out[((k*2+1)*3+axis)*b+s]=c+e+err;
        }
    }
}

#if X86_SIMD
AVX2_TARGET static __m256d abs4(__m256d x) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.), x);
}

AVX2_TARGET static void boxes_avx2(const double *p, const double *r, const double *local,
                                 size_t b, size_t g, double *out) {
    const __m256d one=_mm256_set1_pd(1.);
    const __m256d eps=_mm256_set1_pd(64.*DBL_EPSILON);
    for (size_t k=0; k<g; ++k) {
        const double *a=local+k*6;
        for (size_t axis=0; axis<3; ++axis) {
            size_t s=0;
            for (; s+4<=b; s+=4) {
                __m256d c=_mm256_loadu_pd(p+(k*3+axis)*b+s);
                __m256d scale=abs4(c);
                __m256d e=_mm256_setzero_pd();
                for (size_t j=0; j<3; ++j) {
                    __m256d m=_mm256_loadu_pd(r+(k*9+axis*3+j)*b+s);
                    c=_mm256_add_pd(c,_mm256_mul_pd(m,_mm256_set1_pd((a[j]+a[j+3])*.5)));
                    e=_mm256_add_pd(e,_mm256_mul_pd(abs4(m),_mm256_set1_pd((a[j+3]-a[j])*.5)));
                    scale=_mm256_add_pd(scale,_mm256_mul_pd(abs4(m),
                        _mm256_set1_pd(fabs((a[j]+a[j+3])*.5))));
                }
                __m256d err=_mm256_mul_pd(eps,_mm256_add_pd(one,_mm256_add_pd(scale,e)));
                _mm256_storeu_pd(out+((k*2)*3+axis)*b+s,_mm256_sub_pd(_mm256_sub_pd(c,e),err));
                _mm256_storeu_pd(out+((k*2+1)*3+axis)*b+s,_mm256_add_pd(_mm256_add_pd(c,e),err));
            }
            for (; s<b; ++s) {
                double c=p[(k*3+axis)*b+s], e=0., scale=fabs(c);
                for (size_t j=0; j<3; ++j) {
                    double m=r[(k*9+axis*3+j)*b+s];
                    c += m*(a[j]+a[j+3])*.5;
                    e += fabs(m)*(a[j+3]-a[j])*.5;
                    scale += fabs(m)*fabs((a[j]+a[j+3])*.5);
                }
                double err=64.*DBL_EPSILON*(1.+scale+e);
                out[((k*2)*3+axis)*b+s]=c-e-err;
                out[((k*2+1)*3+axis)*b+s]=c+e+err;
            }
        }
    }
}

/* Input bounds are [geometry, lower/upper, xyz, state]. */
AVX2_TARGET static void distance_avx2(const double *a, const double *other,
                                    size_t n, size_t stride, double *out) {
    const __m256d zero=_mm256_setzero_pd(), one=_mm256_set1_pd(1.);
    const __m256d eps=_mm256_set1_pd(64.*DBL_EPSILON);
    size_t s=0;
    for (; s+4<=n; s+=4) {
        __m256d sq=zero;
        for (size_t axis=0; axis<3; ++axis) {
            __m256d al=_mm256_loadu_pd(a+axis*stride+s);
            __m256d au=_mm256_loadu_pd(a+(axis+3)*stride+s);
            __m256d bl=_mm256_loadu_pd(other+axis*stride+s);
            __m256d bu=_mm256_loadu_pd(other+(axis+3)*stride+s);
            __m256d err=_mm256_mul_pd(eps,_mm256_add_pd(one,
                _mm256_add_pd(_mm256_add_pd(abs4(al),abs4(au)),
                              _mm256_add_pd(abs4(bl),abs4(bu)))));
            __m256d gap=_mm256_max_pd(zero,_mm256_sub_pd(
                _mm256_max_pd(_mm256_sub_pd(bl,au),_mm256_sub_pd(al,bu)),err));
            sq=_mm256_add_pd(sq,_mm256_mul_pd(gap,gap));
        }
        __m256d d=_mm256_sqrt_pd(sq);
        _mm256_storeu_pd(out+s,_mm256_max_pd(zero,_mm256_sub_pd(d,
            _mm256_mul_pd(eps,_mm256_add_pd(one,d)))));
    }
    for (; s<n; ++s) {
        double sq=0.;
        for (size_t axis=0; axis<3; ++axis) {
            double al=a[axis*stride+s], au=a[(axis+3)*stride+s];
            double bl=other[axis*stride+s], bu=other[(axis+3)*stride+s];
            double err=64.*DBL_EPSILON*(1.+fabs(al)+fabs(au)+fabs(bl)+fabs(bu));
            double gap=maximum(0., maximum(bl-au,al-bu)-err);
            sq += gap*gap;
        }
        double d=sqrt(sq);
        out[s]=maximum(0.,d-64.*DBL_EPSILON*(1.+d));
    }
}
#endif

static void distance_scalar(const double *a, const double *other,
                            size_t n, size_t stride, double *out) {
    for (size_t s=0; s<n; ++s) {
        double sq=0.;
        for (size_t axis=0; axis<3; ++axis) {
            double al=a[axis*stride+s], au=a[(axis+3)*stride+s];
            double bl=other[axis*stride+s], bu=other[(axis+3)*stride+s];
            double err=64.*DBL_EPSILON*(1.+fabs(al)+fabs(au)+fabs(bl)+fabs(bu));
            double gap=maximum(0., maximum(bl-au,al-bu)-err);
            sq += gap*gap;
        }
        double d=sqrt(sq);
        out[s]=maximum(0.,d-64.*DBL_EPSILON*(1.+d));
    }
}

void nero_world_aabbs(const double *p, const double *r, const double *local,
                     size_t b, size_t g, int avx2, double *out) {
#if X86_SIMD
    if (avx2 && nero_simd_has_avx2()) { boxes_avx2(p,r,local,b,g,out); return; }
#endif
    boxes_scalar(p,r,local,b,g,out);
}

void nero_pair_clearances(const double *bounds, const int64_t *pairs,
                          size_t b, size_t p, int avx2, double *out) {
    for (size_t i=0; i<p; ++i) {
        const double *a=bounds+pairs[i*2]*6*b, *other=bounds+pairs[i*2+1]*6*b;
#if X86_SIMD
        if (avx2 && nero_simd_has_avx2()) { distance_avx2(a,other,b,b,out+i*b); continue; }
#endif
        distance_scalar(a,other,b,b,out+i*b);
    }
}

void nero_object_clearances(const double *bounds, const double *object,
                            size_t b, size_t g, int avx2, double *out) {
    for (size_t i=0; i<g; ++i) {
#if X86_SIMD
        if (avx2 && nero_simd_has_avx2()) { distance_avx2(bounds+i*6*b,object,b,b,out+i*b); continue; }
#endif
        distance_scalar(bounds+i*6*b,object,b,b,out+i*b);
    }
}

static double points_scalar(const double *box, const double *points, size_t n) {
    double minimum=INFINITY;
    for (size_t i=0; i<n; ++i) {
        double sq=0.;
        for (size_t axis=0; axis<3; ++axis) {
            double value=points[axis*n+i];
            double lower=box[axis], upper=box[axis+3];
            double err=64.*DBL_EPSILON*(1.+fabs(value)+fabs(lower)+fabs(upper));
            double gap=maximum(0.,maximum(lower-value,value-upper)-err);
            sq += gap*gap;
        }
        if (sq<minimum) minimum=sq;
        if (minimum==0.) return 0.;
    }
    double d=sqrt(minimum);
    return maximum(0.,d-64.*DBL_EPSILON*(1.+d));
}

#if X86_SIMD
AVX2_TARGET static double points_avx2(const double *box, const double *points, size_t n) {
    const __m256d zero=_mm256_setzero_pd(), one=_mm256_set1_pd(1.);
    const __m256d eps=_mm256_set1_pd(64.*DBL_EPSILON);
    __m256d best=_mm256_set1_pd(INFINITY);
    size_t i=0;
    for (; i+4<=n; i+=4) {
        __m256d sq=zero;
        for (size_t axis=0; axis<3; ++axis) {
            __m256d x=_mm256_loadu_pd(points+axis*n+i);
            __m256d lower=_mm256_set1_pd(box[axis]), upper=_mm256_set1_pd(box[axis+3]);
            __m256d err=_mm256_mul_pd(eps,_mm256_add_pd(one,
                _mm256_add_pd(abs4(x),_mm256_add_pd(abs4(lower),abs4(upper)))));
            __m256d gap=_mm256_max_pd(zero,_mm256_sub_pd(
                _mm256_max_pd(_mm256_sub_pd(lower,x),_mm256_sub_pd(x,upper)),err));
            sq=_mm256_add_pd(sq,_mm256_mul_pd(gap,gap));
        }
        best=_mm256_min_pd(best,sq);
        if (_mm256_movemask_pd(_mm256_cmp_pd(best,zero,_CMP_EQ_OQ))) return 0.;
    }
    double lanes[4]; _mm256_storeu_pd(lanes,best);
    double minimum=fmin(fmin(lanes[0],lanes[1]),fmin(lanes[2],lanes[3]));
    for (; i<n; ++i) {
        double sq=0.;
        for (size_t axis=0; axis<3; ++axis) {
            double x=points[axis*n+i], lower=box[axis], upper=box[axis+3];
            double err=64.*DBL_EPSILON*(1.+fabs(x)+fabs(lower)+fabs(upper));
            double gap=maximum(0.,maximum(lower-x,x-upper)-err);
            sq += gap*gap;
        }
        minimum=fmin(minimum,sq);
    }
    double d=sqrt(minimum);
    return maximum(0.,d-64.*DBL_EPSILON*(1.+d));
}
#endif

/* Sparse object points only; this does not replace the static scene index. */
void nero_point_clearances(const double *bounds, const double *points,
                           size_t b, size_t g, size_t n, int avx2, double *out) {
    for (size_t i=0; i<b*g; ++i) {
#if X86_SIMD
        if (avx2 && nero_simd_has_avx2()) { out[i]=points_avx2(bounds+i*6,points,n); continue; }
#endif
        out[i]=points_scalar(bounds+i*6,points,n);
    }
}
