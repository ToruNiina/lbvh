# LBVH

An implementation of the following paper

- Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", High Performance Graphics (2012)

and the following blog posts

- https://devblogs.nvidia.com/thinking-parallel-part-ii-tree-traversal-gpu/
- https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/

depending on [thrust](https://thrust.github.io/).

It is capable to contain any object and allows morton code overlap.

If the morton codes of objects are the same, it internally assign an AABB to the
group of objects that have the same morton code and consider the AABB as a leaf
node. It means that leaf node generally corresponds to a group of objects
(in the normal case in which all the objects have different morton codes, each
leaf node corresponds to one object).

Also, nearest neighbor query based on the following paper is supported.

- Nick Roussopoulos, Stephen Kelley, Frederic Vincent, "Nearest Neighbor Queries", ACM-SIGMOD (1995)

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
        // calculate aabb of object ...
    }
};
struct point_getter
{
    // return float4 if your object uses float.
    // if you chose double, return double4.
    __device__
    float4 operator()(const object& f) const noexcept
    {
        // calculate representative coordinate of object to compute morton code.
    }
};

int main()
{
    std::vector<objects> objs;

    // initialize objs ...

    // construct a bvh
    lbvh::bvh<float, object, point_getter, aabb_getter> bvh(objs.begin(), objs.end());

    // get a set of device (raw) pointers to use it in device functions.
    // Do not use this on host!
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
                const unsigned int other_idx = buffer[j];
                const object&      other     = bvh_dev.objects[other_idx];
                // do some stuff ...
            }
            return ;
        });

    return 0;
}
```

## Synopsis

### AABB

```cpp
template<typename T>
struct aabb
{
    /* T4 (float4 if T == float, double4 if T == double) */ upper;
    /* T4 (float4 if T == float, double4 if T == double) */ lower;
};

template<typename T>
__device__ __host__
inline bool intersects(const aabb<T>& lhs, const aabb<T>& rhs) noexcept;


template<typename T>
__device__ __host__
inline aabb<T> merge(const aabb<T>& lhs, const aabb<T>& rhs) noexcept
```

### BVH

```cpp
template<typename Real, typename Object, typename PointGetter, typename AABBGetter>
class bvh
{
  public:
    using real_type   = Real;
    using index_type = std::uint32_t;
    using object_type = Object;
    using aabb_type   = aabb<real_type>;
    using node_type   = detail::node;
    using point_getter_type = PointGetter;
    using aabb_getter_type  = AABBGetter;

    template<typename InputIterator>
    bvh(InputIterator first, InputIterator last)
        : objects_h_(first, last), objects_d_(objects_h_)
    {
        this->construct();
    }

    template<typename InputIterator>
    void assign(InputIterator first, InputIterator last);
    void clear();

    bvh_device<real_type, object_type>  get_device_repr()       noexcept;
    cbvh_device<real_type, object_type> get_device_repr() const noexcept;
};

template<typename Real, typename Objects, typename OutputIterator, bool IsConst>
__device__
unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const aabb<Real>& q, unsigned int max_buffer_size,
        OutputIterator outiter) noexcept

template<typename Real, typename Object>
struct bvh_device
{
    using real_type  = Real;
    using aabb_type  = aabb<real_type>;
    using node_type  = detail::node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_leaves;  // (# of leaves), N
    unsigned int num_objects; // (# of objects) ; can be larger than N

    node_type *  nodes;
    aabb_type *  aabbs;
    index_type*  ranges;
    index_type*  indices;
    object_type* objects;
};
template<typename Real, typename Object>
struct cbvh_device
{
    using real_type  = Real;
    using aabb_type  = aabb<real_type>;
    using node_type  = detail::node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_leaves;  // (# of leaves), N
    unsigned int num_objects; // (# of objects) ; can be larger than N

    node_type   const* nodes;
    aabb_type   const* aabbs;
    index_type  const* ranges;
    index_type  const* indices;
    object_type const* objects;
};
```
