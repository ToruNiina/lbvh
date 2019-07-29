# LBVH

an implementation of a paper

- Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", High Performance Graphics (2012)

and blog posts

- https://devblogs.nvidia.com/thinking-parallel-part-ii-tree-traversal-gpu/
- https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/

for point clouds.

## sample code

```cpp

struct object
{
    // any object you want.
};

struct aabb_getter
{
    // return aabb<float> if your object uses float.
    // if you chose double, return aabb<double>.
    __device__
    lbvh::aabb<float> operator()(const object& f) const noexcept
    {
        // ...
    }
};
struct point_getter
{
    // return float4 if your object uses float.
    // if you chose double, return double4.
    __device__
    float4 operator()(const object& f) const noexcept
    {
        // ...
    }
};

int main()
{
    std::vector<objects> objs;
    // initialize objs ...

    // construct a bvh
    lbvh::bvh<float, object, point_getter, aabb_getter> bvh(objs.begin(), objs.end());

    // get a set of device (raw) pointers. Do not use this on host!
    const auto bvh_dev = bvh.get_device_repr();

    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(N),
        [bvh_dev] __device__ (const unsigned int idx)
        {
            unsigned int buffer[10];

            point_getter get_point;
            const auto self = get_point(bvh_dev.objects[idx]);

            // make a query box.
            const lbvh::aabb<float> query(
                    make_float4(self.x-0.1, self.y-0.1, self.z-0.1, 0),
                    make_float4(self.x+0.1, self.y+0.1, self.z+0.1, 0)
                );

            // send a query!
            const auto num_found = query_device(bvh_dev, query, 10, buffer);

            for(unsigned int j=0; j<num_found; ++j)
            {
                const auto other_idx = buffer[j];
                const auto other     = bvh_dev.objects[other_idx];
                // do some stuff ...
            }
            return ;
        });

    return 0;
}
```
