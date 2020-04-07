#include <optix_world.h>

#include "random.h"
#include "util.h"

using namespace optix;

rtDeclareVariable(int, fill_surface_interaction, , );

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(void *, accel, , );
rtDeclareVariable(unsigned long long, shape_ptr, , );
rtDeclareVariable(unsigned int, launch_index, rtLaunchIndex, );

rtDeclareVariable(float3, p, attribute p, );
rtDeclareVariable(float2, uv, attribute uv, );
rtDeclareVariable(float3, ns, attribute ns, );
rtDeclareVariable(float3, ng, attribute ng, );
rtDeclareVariable(float3, dp_du, attribute dp_du, );
rtDeclareVariable(float3, dp_dv, attribute dp_dv, );
rtDeclareVariable(Ray, ray, rtCurrentRay,);

rtBuffer<bool> in_mask;

rtBuffer<float> in_ox, in_oy, in_oz,
                in_dx, in_dy, in_dz,
                in_mint, in_maxt, in_kappa;

rtBuffer<float> out_t, out_u, out_v, out_ng_x, out_ng_y,
                out_ng_z, out_ns_x, out_ns_y, out_ns_z,
                out_p_x, out_p_y, out_p_z,
                out_dp_du_x, out_dp_du_y, out_dp_du_z,
                out_dp_dv_x, out_dp_dv_y, out_dp_dv_z;

rtBuffer<unsigned long long> out_shape_ptr;

rtBuffer<uint32_t> out_primitive_id;

rtBuffer<bool> out_hit;

struct PerRayData { };

RT_PROGRAM void ray_gen_closest() {
    float3 ro = make_float3(in_ox[launch_index],
                            in_oy[launch_index],
                            in_oz[launch_index]),
           rd = make_float3(in_dx[launch_index],
                            in_dy[launch_index],
                            in_dz[launch_index]);
    float  mint = in_mint[launch_index],
           maxt = in_maxt[launch_index];

    if (!in_mask[launch_index]) {
        out_shape_ptr[launch_index] = 0;
        out_t[launch_index] = CUDART_INF_F;
    } else {
        PerRayData prd;
        Ray ray = make_Ray(ro, rd, 0, mint, maxt);
        rtTrace(top_object, ray, prd);
    }
}

RT_PROGRAM void ray_gen_any() {
    float3 ro = make_float3(in_ox[launch_index],
                            in_oy[launch_index],
                            in_oz[launch_index]),
           rd = make_float3(in_dx[launch_index],
                            in_dy[launch_index],
                            in_dz[launch_index]);
    float  mint = in_mint[launch_index],
           maxt = in_maxt[launch_index];

    Ray ray = make_Ray(ro, rd, 0, mint, maxt);

    if (!in_mask[launch_index]) {
        out_hit[launch_index] = false;
    } else {
        PerRayData prd;
        rtTrace(top_object, ray, prd, RT_VISIBILITY_ALL,
                RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    }
}

RT_PROGRAM void ray_gen_occluder() {
    // if (launch_index == 10)
        // printf("Hello ray_gen_occluder --> launch_index %d, kappa: %f \n", launch_index, in_kappa[launch_index]);

    // TODO should be a variable
    int test_count = 4;

    float kappa = in_kappa[launch_index];

    unsigned int seed = tea<16>(launch_index, 0u);

    // if (launch_index == 0 || launch_index == 262144)
        // printf("--> launch_index %d, kappa: %f, dir %f, seed: %i \n", launch_index, in_kappa[launch_index], in_dx[launch_index], seed);

    float3 ro = make_float3(in_ox[launch_index],
                            in_oy[launch_index],
                            in_oz[launch_index]),
           rd = make_float3(in_dx[launch_index],
                            in_dy[launch_index],
                            in_dz[launch_index]);
    float  mint = in_mint[launch_index],
           maxt = in_maxt[launch_index];

    if (!in_mask[launch_index]) {
        out_shape_ptr[launch_index] = 0;
        out_t[launch_index] = CUDART_INF_F;
    } else {
        PerRayData prd;

        float res_t = CUDART_INF_F;

        float3 res_p;
        float3 res_ng;
        float2 res_uv;
        unsigned long long res_shape_ptr = 0;
        uint32_t res_prim_id;

        // Compute ray direction frame
        float3 rd_s, rd_t;
        coordinate_system(rd, rd_s, rd_t);

        for (int i = 0; i < test_count; i++) {
            // Sample random direction using vMF
            float2 sample = make_float2(rnd(seed), rnd(seed));
            float3 offset = square_to_von_mises_fisher(sample, kappa);


            // if (launch_index == 10) {
            //     printf("sample: %f, %f \n", sample.x, sample.y);
            //     printf("offset: %f, %f, %f \n", offset.x, offset.y, offset.z);
            //     printf("rd: %f, %f, %f \n", rd.x, rd.y, rd.z);
            //     printf("rd_s: %f, %f, %f \n", rd_s.x, rd_s.y, rd_s.z);
            //     printf("rd_t: %f, %f, %f \n", rd_t.x, rd_t.y, rd_t.z);
            // }

            float3 rd_offset = rd_s * offset.x + rd_t * offset.y + rd * offset.z;

            // Generate and trace ray
            Ray ray = make_Ray(ro, rd_offset, 0, mint, maxt);
            rtTrace(top_object, ray, prd);

            // If no hit, continue
            if (out_t[launch_index] == CUDART_INF_F)
                continue;

            // TODO: use ray payload for this
            float3 p1 = make_float3(out_p_x[launch_index], out_p_y[launch_index], out_p_z[launch_index]);
            float3 n1 = make_float3(out_ng_x[launch_index], out_ng_y[launch_index], out_ng_z[launch_index]);
            float2 uv1 = make_float2(out_u[launch_index], out_v[launch_index]);
            unsigned long long shape_ptr1 = out_shape_ptr[launch_index];

            // if (launch_index == 10) {
            //     // printf("launch_index: %d \n", launch_index);
            //     printf("p1: %f, %f, %f \n", p1.x, p1.y, p1.z);
            //     printf("n1: %f, %f, %f \n", n1.x, n1.y, n1.z);
            //     printf("uv1: %f, %f \n", uv1.x, uv1.y);
            //     printf("shape_ptr1: %d \n", shape_ptr1);
            // }

            // -------------------------------------
            // Process intersection


            if (i == 0) {
                res_t = out_t[launch_index];
                res_p = p1;
                res_ng = n1;
                res_uv = uv1;
                res_shape_ptr = shape_ptr1;
                res_prim_id = out_primitive_id[launch_index];
            } else if (res_shape_ptr != shape_ptr1) {
                // Check if ro and p are on the opposite side of the plane defined by {res_o, res_n}
                // If not, then update the res fields with new occluder
                if (dot(n1, res_p - p1) * dot(res_ng, ro - res_p) < 0.f) {
                    res_t = out_t[launch_index];
                    res_p = p1;
                    res_ng = n1;
                    res_uv = uv1;
                    res_shape_ptr = shape_ptr1;
                    res_prim_id = out_primitive_id[launch_index];
                }
            }
        }

        // Write result in ouput SurfaceInteraction3f
        out_t[launch_index] = res_t;

        out_p_x[launch_index] = res_p.x;
        out_p_y[launch_index] = res_p.y;
        out_p_z[launch_index] = res_p.z;

        out_u[launch_index] = res_uv.x;
        out_v[launch_index] = res_uv.y;

        out_shape_ptr[launch_index] = res_shape_ptr;
        out_primitive_id[launch_index] = res_prim_id;
    }
}

RT_PROGRAM void ray_hit() {
    if (out_hit.size() > 0) {
        out_hit[launch_index] = true;
    } else {
        out_shape_ptr[launch_index] = shape_ptr;

        out_primitive_id[launch_index] = rtGetPrimitiveIndex();

        out_u[launch_index] = uv.x;
        out_v[launch_index] = uv.y;

        out_p_x[launch_index] = p.x;
        out_p_y[launch_index] = p.y;
        out_p_z[launch_index] = p.z;

        out_ng_x[launch_index] = ng.x;
        out_ng_y[launch_index] = ng.y;
        out_ng_z[launch_index] = ng.z;

        if (fill_surface_interaction == 1) {
            out_ns_x[launch_index] = ns.x;
            out_ns_y[launch_index] = ns.y;
            out_ns_z[launch_index] = ns.z;

            out_dp_du_x[launch_index] = dp_du.x;
            out_dp_du_y[launch_index] = dp_du.y;
            out_dp_du_z[launch_index] = dp_du.z;

            out_dp_dv_x[launch_index] = dp_dv.x;
            out_dp_dv_y[launch_index] = dp_dv.y;
            out_dp_dv_z[launch_index] = dp_dv.z;
        }

        out_t[launch_index] = sqrt(squared_norm(p - ray.origin) / squared_norm(ray.direction));
    }
}

RT_PROGRAM void ray_miss() {
    if (out_hit.size() > 0) {
        out_hit[launch_index] = false;
    } else {
        out_shape_ptr[launch_index] = 0;
        out_t[launch_index] = CUDART_INF_F;
    }
}

RT_PROGRAM void ray_err() {
    rtPrintExceptionDetails();
}
