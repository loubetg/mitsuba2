#include <optix_world.h>
#include <math_constants.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

__device__ void coordinate_system(float3 n, float3 &x, float3 &y) {
    /* Based on "Building an Orthonormal Basis, Revisited" by
       Tom Duff, James Burgess, Per Christensen,
       Christophe Hery, Andrew Kensler, Max Liani,
       and Ryusuke Villemin (JCGT Vol 6, No 1, 2017) */

    float s = copysignf(1.f, n.z),
          a = -1.f / (s + n.z),
          b = n.x * n.y * a;

    x = make_float3(n.x * n.x * a * s + 1.f, b * s, -n.x * s);
    y = make_float3(b, s + n.y * n.y * a, -n.y);
}

__device__ inline float squared_norm(float3 v) {
    return dot(v, v);
}

__device__ float2 square_to_uniform_disk_concentric(float2 sample) {
    float x = 2.f * sample.x - 1.f,
          y = 2.f * sample.y - 1.f;

    float phi, r;
    if (x == 0 && y == 0) {
        r = phi = 0;
    } else if (x * x > y * y) {
        r = x;
        phi = (M_PI / 4.f) * (y / x);
    } else {
        r = y;
        phi = (M_PI / 2.f) - (x / y) * (M_PI / 4.f);
    }

    float s, c;
    sincosf(phi, &s, &c);
    return make_float2(r * c, r * s);
}

__device__ float3 square_to_von_mises_fisher(float2 sample, float kappa) {
    // Low-distortion warping technique based on concentric disk mapping
    float2 p = square_to_uniform_disk_concentric(sample);

    float r2 = p.x * p.x + p.y * p.y,
          sy = fmaxf(1.f - r2, 1e-6f),
          cos_theta = 1.f + logf(sy + (1.f - sy) * expf(-2.f * kappa)) / kappa;

    p *= sqrtf((1.f - cos_theta * cos_theta) / r2);

    return make_float3(p.x, p.y, cos_theta);
}