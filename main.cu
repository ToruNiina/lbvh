#include "lbvh.cuh"
#include <random>
#include <vector>
#include <thrust/random.h>

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
struct distance_calculator
{
    __device__
    float operator()(const float4 point, const float4 object) const noexcept
    {
        return (point.x - object.x) * (point.x - object.x) +
               (point.y - object.y) * (point.y - object.y) +
               (point.z - object.z) * (point.z - object.z);
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

    lbvh::bvh<float, float4, aabb_getter> bvh(ps.begin(), ps.end(), true);

    const auto bvh_dev = bvh.get_device_repr();

    std::cout << "testing query_device:overlap ...\n";
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
                lbvh::aabb<float> query_box;
                query_box.lower = make_float4(self.x-r, self.y-r, self.z-r, 0);
                query_box.upper = make_float4(self.x+r, self.y+r, self.z+r, 0);
                const auto num_found = lbvh::query_device(
                        bvh_dev, lbvh::overlaps(query_box), buffer, 10);

                for(unsigned int j=0; j<10; ++j)
                {
                    const auto jdx    = buffer[j];
                    if(j >= num_found)
                    {
                        assert(jdx == 0xFFFFFFFF);
                        continue;
                    }
                    else
                    {
                        assert(jdx != 0xFFFFFFFF);
                        assert(jdx < bvh_dev.num_objects);
                    }
                    const auto other  = bvh_dev.objects[jdx];
                    assert(fabsf(self.x - other.x) < r); // check coordinates
                    assert(fabsf(self.y - other.y) < r); // are in the range
                    assert(fabsf(self.z - other.z) < r); // of query box
                }
            }
            return ;
        });

    std::cout << "testing query_device:nearest_neighbor ...\n";
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(N),
        [bvh_dev] __device__ (const unsigned int idx) {
            const auto self = bvh_dev.objects[idx];
            const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(self),
                                                 distance_calculator());
            assert(nest.first != 0xFFFFFFFF);
            const auto other   = bvh_dev.objects[nest.first];
            // of course, the nearest object is itself.
            assert(nest.second == 0.0f);
            assert(self.x == other.x);
            assert(self.y == other.y);
            assert(self.z == other.z);
            return ;
       });

    thrust::device_vector<float4> random_points(N);
    thrust::transform(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(N),
        random_points.begin(), [] __device__(const unsigned int idx) {
            thrust::default_random_engine rand;
            thrust::uniform_real_distribution<float> uni(0.0f, 1.0f);
            rand.discard(idx);
            const float x = uni(rand);
            const float y = uni(rand);
            const float z = uni(rand);
            return make_float4(x, y, z, 0);
        });

    thrust::for_each(random_points.begin(), random_points.end(),
        [bvh_dev] __device__ (const float4 pos) {
            const auto calc = distance_calculator();
            const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);
            assert(nest.first != 0xFFFFFFFF);

            for(unsigned int i=0; i<bvh_dev.num_objects; ++i)
            {
                const auto dist = calc(bvh_dev.objects[i], pos);
                if(i == nest.first)
                {
                    assert(dist == nest.second);
                }
                else
                {
                    assert(dist >= nest.second);
                }
            }
            return ;
        });

    std::cout << "testing query_host:overlap ...\n";
    {
        for(std::size_t i=0; i<10; ++i)
        {
            const auto self = bvh.objects_host()[i];
            const float dr = 0.1f;
            for(unsigned int cnt=1; cnt<10; ++cnt)
            {
                const float r = dr * cnt;
                lbvh::aabb<float> query_box;
                query_box.lower = make_float4(self.x-r, self.y-r, self.z-r, 0);
                query_box.upper = make_float4(self.x+r, self.y+r, self.z+r, 0);

                std::vector<std::size_t> buffer;
                const auto num_found = lbvh::query_host(bvh,
                        lbvh::overlaps(query_box), std::back_inserter(buffer));

                for(unsigned int jdx : buffer)
                {
                    assert(jdx < bvh.objects_host().size());

                    const auto other  = bvh.objects_host()[jdx];
                    assert(fabsf(self.x - other.x) < r); // check coordinates
                    assert(fabsf(self.y - other.y) < r); // are in the range
                    assert(fabsf(self.z - other.z) < r); // of query box
                }
            }
        }
    }
    return 0;
}
