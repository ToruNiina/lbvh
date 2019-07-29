#include "lbvh/bvh.cuh"
#include <random>
#include <vector>

struct aabb_getter
{
    __device__
    lbvh::aabb<float> operator()(const float4 f) const noexcept
    {
        lbvh::aabb<float> retval;
        retval.upper = f;
        retval.lower = f;
        return retval;
    }
};
struct point_getter
{
    __device__
    float4 operator()(const float4 f) const noexcept
    {
        return f;
    }
};

int main()
{
    constexpr std::size_t N=10;
    std::vector<float4> ps(N);

    std::mt19937 mt(123456789);
    std::uniform_real_distribution<float> uni(0.0, 1.0);

    for(auto& p : ps)
    {
        p.x = uni(mt);
        p.y = uni(mt);
        p.z = uni(mt);
    }

    lbvh::bvh<float, float4, point_getter, aabb_getter> bvh(ps.begin(), ps.end());

    const auto bvh_dev = bvh.get_device_repr();
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<std::size_t>(0),
        thrust::make_counting_iterator<std::size_t>(N),
        [bvh_dev] __device__ (std::size_t idx) {
            unsigned int buffer[10];
            const auto self = bvh_dev.objects[idx];
            const float  dr = 0.1f;
            for(std::size_t i=1; i<10; ++i)
            {
                for(unsigned int j=0; j<10; ++j)
                {
                    buffer[j] = 0xFFFFFFFF;
                }
                const float r = dr * i;
                lbvh::aabb<float> query;
                query.lower = make_float4(self.x-r, self.y-r, self.z-r, 0);
                query.upper = make_float4(self.x+r, self.y+r, self.z+r, 0);
                const auto num_found = query_device(bvh_dev, query, 10, buffer);

                for(unsigned int j=0; j<10; ++j)
                {
                    const auto jdx    = buffer[j];
                    if(j >= num_found)
                    {
                        assert(jdx == 0xFFFFFFFF);
                        continue;
                    }

                    const auto other  = bvh_dev.objects[jdx];
                    assert(fabsf(self.x - other.x) < r); // check coordinates
                    assert(fabsf(self.y - other.y) < r); // are in the range
                    assert(fabsf(self.z - other.z) < r); // of query box
                }
            }
            return ;
        });

    return 0;
}
