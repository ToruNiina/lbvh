#ifndef LBVH_BVH_CUH
#define LBVH_BVH_CUH
#include "aabb.cuh"
#include "morton_code.cuh"
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

namespace lbvh
{
namespace detail
{
struct node
{
    std::uint32_t parent_idx; // parent node
    std::uint32_t left_idx;   // index of left  child node
    std::uint32_t right_idx;  // index of right child node
    std::uint32_t range_idx;  // > 0 if leaf node. index of range.
};

// a set of pointers to use it on device.
template<typename Real, typename Object, bool IsConst>
struct basic_device_bvh;
template<typename Real, typename Object>
struct basic_device_bvh<Real, Object, false>
{
    using real_type  = Real;
    using aabb_type  = aabb<real_type>;
    using node_type  = detail::node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_leaves;  // (# of leaves)                          N
    unsigned int num_objects; // (# of objects) ; can be larger than N
    node_type *  nodes;
    aabb_type *  aabbs;
    index_type*  ranges;
    index_type*  indices;
    object_type* objects;
};
template<typename Real, typename Object>
struct basic_device_bvh<Real, Object, true>
{
    using real_type  = Real;
    using aabb_type  = aabb<real_type>;
    using node_type  = detail::node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_leaves;  // (# of leaves)                          N
    unsigned int num_objects; // (# of objects) ; can be larger than N

    node_type   const* nodes;
    aabb_type   const* aabbs;
    index_type  const* ranges;
    index_type  const* indices;
    object_type const* objects;
};

__device__
inline uint2 determine_range(unsigned int* node_code, unsigned int num_leaves,
                             unsigned int idx)
{
    if(idx == 0)
    {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    const unsigned int self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx-1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx+1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = thrust::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0)
    {
        thrust::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

__device__
inline unsigned int find_split(unsigned int* node_code, unsigned int num_leaves,
                               unsigned int first, unsigned int last) noexcept
{
    const unsigned int first_code = node_code[first];
    const unsigned int last_code  = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}
} // detail

template<typename Real, typename Object>
using  bvh_device = detail::basic_device_bvh<Real, Object, false>;
template<typename Real, typename Object>
using cbvh_device = detail::basic_device_bvh<Real, Object, true>;

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

  public:

    template<typename InputIterator>
    bvh(InputIterator first, InputIterator last)
        : objects_h_(first, last), objects_d_(objects_h_)
    {
        this->construct();
    }

    bvh()                      = default;
    ~bvh()                     = default;
    bvh(const bvh&)            = default;
    bvh(bvh&&)                 = default;
    bvh& operator=(const bvh&) = default;
    bvh& operator=(bvh&&)      = default;

    void clear()
    {
        this->objects_h_.clear();
        this->objects_d_.clear();
        this->aabbs_.clear();
        this->nodes_.clear();
        this->indices_.clear();
        this->ranges_ .clear();
        return ;
    }

    template<typename InputIterator>
    void assign(InputIterator first, InputIterator last)
    {
        this->objects_h_.assign(first, last);
        this->objects_d_ = this->objects_h_;
        this->construct();
        return;
    }

    bvh_device<real_type, object_type> get_device_repr()       noexcept
    {
        return bvh_device<real_type, object_type>{
            static_cast<unsigned int>(nodes_.size()),
            static_cast<unsigned int>((nodes_.size() + 1) / 2),
            static_cast<unsigned int>(objects_d_.size()),
            nodes_.data().get(),  aabbs_.data().get(),
            ranges_.data().get(), indices_.data().get(),
            objects_d_.data().get()
        };
    }
    cbvh_device<real_type, object_type> get_device_repr() const noexcept
    {
        return cbvh_device<real_type, object_type>{
            static_cast<unsigned int>(nodes_.size()),
            static_cast<unsigned int>((nodes_.size() + 1) / 2),
            static_cast<unsigned int>(objects_d_.size()),
            nodes_.data().get(),  aabbs_.data().get(),
            ranges_.data().get(), indices_.data().get(),
            objects_d_.data().get()
        };
    }

    void construct()
    {
        assert(objects_h_.size() == objects_d_.size());
        if(objects_h_.size() == 0u) {return;}

        // BVH has N-1 internal nodes and N leaf nodes.
        // Note: if objects_h_.size() == 1, this function already returns
        const auto inf = std::numeric_limits<real_type>::infinity();

        // --------------------------------------------------------------------
        // calculate morton code of each points
        aabb_type default_aabb;
        default_aabb.upper.x = -inf; default_aabb.lower.x = inf;
        default_aabb.upper.y = -inf; default_aabb.lower.y = inf;
        default_aabb.upper.z = -inf; default_aabb.lower.z = inf;

        const auto aabb_whole = thrust::reduce(
            thrust::make_transform_iterator(objects_d_.begin(), aabb_getter_type()),
            thrust::make_transform_iterator(objects_d_.end(),   aabb_getter_type()),
            default_aabb,
            [] __device__ (const aabb_type& lhs, const aabb_type& rhs) {
                return merge(lhs, rhs);
            });

        thrust::device_vector<std::uint32_t> morton(this->objects_h_.size());
        thrust::transform(objects_d_.begin(), objects_d_.end(), morton.begin(),
            [aabb_whole] __device__ (const object_type& v) {
                point_getter_type point_getter;
                auto p = point_getter(v);
                p.x -= aabb_whole.lower.x;
                p.y -= aabb_whole.lower.y;
                p.z -= aabb_whole.lower.z;
                p.x /= (aabb_whole.upper.x - aabb_whole.lower.x);
                p.y /= (aabb_whole.upper.y - aabb_whole.lower.y);
                p.z /= (aabb_whole.upper.z - aabb_whole.lower.z);
                return morton_code(p);
            });

        // --------------------------------------------------------------------
        // sort object-indices by morton code
        this->indices_.resize(objects_h_.size());
        thrust::copy(thrust::make_counting_iterator<index_type>(0),
                     thrust::make_counting_iterator<index_type>(objects_h_.size()),
                     this->indices_.begin());
        // keep indices ascending order
        thrust::stable_sort_by_key(morton.begin(), morton.end(), this->indices_.begin());

        // --------------------------------------------------------------------
        // # construct ranges
        //
        // morton   => | 0| 0| 0| 1| 1| 3|... |max|
        // indices_ => | 0| 4| 5| 2| 3| 1|... |N-1|
        //  |
        //  | reduction!
        //  v
        // Umorton  => | 0| 1| 3| ... |
        // Nobjects => | 3| 2| ...    |
        //  |
        //  | inclusive_scan!
        //  v
        // ranges   => | 0| 3| 5| ... |

        this->ranges_.resize(objects_h_.size() + 1, /* fill by zero */ 0);
        thrust::device_vector<index_type> number_of_objects(morton.size());
        thrust::device_vector<index_type> node_morton(morton.size());

        const auto reduced = thrust::reduce_by_key(morton.begin(), morton.end(),
                              thrust::constant_iterator<index_type>(1),
                              node_morton.begin(), number_of_objects.begin());

        const unsigned int num_leaves = thrust::distance(node_morton.begin(), reduced.first);
        const unsigned int num_nodes  = num_leaves * 2 - 1;
        assert(num_leaves != 0);

        thrust::inclusive_scan(number_of_objects.begin(), number_of_objects.end(),
                               this->ranges_.begin() + 1);

        // --------------------------------------------------------------------
        // construct leaf nodes and aabbs

        node_type default_node;
        default_node.parent_idx = 0xFFFFFFFF;
        default_node.left_idx   = 0xFFFFFFFF;
        default_node.right_idx  = 0xFFFFFFFF;
        default_node.range_idx  = 0xFFFFFFFF;
        this->nodes_.resize(num_nodes, default_node);
        this->aabbs_.resize(num_nodes, default_aabb);

        // values to capture ...
        const auto invalid            = std::numeric_limits<index_type>::max();
        const auto num_internal_nodes = num_leaves - 1;
        const auto self               = this->get_device_repr();
        thrust::for_each(thrust::device,
            thrust::make_counting_iterator<index_type>(0),
            thrust::make_counting_iterator<index_type>(num_leaves),
            [num_internal_nodes, self, inf, invalid] __device__ (const index_type idx)
            {
                // skip region reserved for internal nodes
                self.nodes[num_internal_nodes + idx].range_idx = idx + 1;
                self.nodes[num_internal_nodes + idx].left_idx  = invalid;
                self.nodes[num_internal_nodes + idx].right_idx = invalid;

                // construct aabb that wraps all objects
                aabb_getter_type aabb_getter;
                const auto first = self.ranges[idx  ];
                const auto last  = self.ranges[idx+1];
                aabb_type box;
                box.lower.x =  inf; box.lower.y =  inf; box.lower.z =  inf;
                box.upper.x = -inf; box.upper.y = -inf; box.upper.z = -inf;
                for(index_type i = first; i < last; ++i)
                {
                    box = merge(box, aabb_getter(self.objects[self.indices[i]]));
                }
                self.aabbs[num_internal_nodes + idx] = box;
                return;
            });

        // --------------------------------------------------------------------
        // construct internal nodes

        const auto node_code = node_morton.data();
        thrust::for_each(thrust::device,
            thrust::make_counting_iterator<index_type>(0),
            thrust::make_counting_iterator<index_type>(num_internal_nodes),
            [self, node_code, num_leaves, inf] __device__ (const index_type idx)
            {
                self.nodes[idx].range_idx  = 0; // 0 is for internal nodes

                const uint2 ij  = detail::determine_range(node_code.get(), num_leaves, idx);
                const int gamma = detail::find_split(node_code.get(), num_leaves, ij.x, ij.y);

                self.nodes[idx].left_idx  = gamma;
                self.nodes[idx].right_idx = gamma + 1;
                if(thrust::min(ij.x, ij.y) == gamma)
                {
                    self.nodes[idx].left_idx += num_leaves - 1;
                }
                if(thrust::max(ij.x, ij.y) == gamma + 1)
                {
                    self.nodes[idx].right_idx += num_leaves - 1;
                }
                self.nodes[self.nodes[idx].left_idx].parent_idx  = idx;
                self.nodes[self.nodes[idx].right_idx].parent_idx = idx;
                return;
            });

        // --------------------------------------------------------------------
        // create AABB for each node by bottom-up strategy

        thrust::device_vector<int> flag_container(num_internal_nodes, 0);
        const auto flags = flag_container.data().get();

        thrust::for_each(thrust::device,
            thrust::make_counting_iterator<index_type>(num_internal_nodes),
            thrust::make_counting_iterator<index_type>(num_internal_nodes + num_leaves),
            [self, flags] __device__ (index_type idx)
            {
                unsigned int parent = self.nodes[idx].parent_idx;
                while(parent != 0xFFFFFFFF) // means idx == 0
                {
                    const int old = atomicCAS(flags + parent, 0, 1);
                    if(old == 0)
                    {
                        // this is the first thread entered here.
                        // wait the other thread from the other child node.
                        return;
                    }
                    assert(old == 1);
                    // here, the flag has already been 1. it means that this
                    // thread is the 2nd thread. merge AABB of both childlen.

                    const auto lidx = self.nodes[parent].left_idx;
                    const auto ridx = self.nodes[parent].right_idx;
                    const auto lbox = self.aabbs[lidx];
                    const auto rbox = self.aabbs[ridx];
                    self.aabbs[parent] = merge(lbox, rbox);

                    // look the next parent...
                    parent = self.nodes[parent].parent_idx;
                }
                return;
            });

        return;
    }


  private:

    thrust::host_vector  <object_type>   objects_h_;
    thrust::device_vector<object_type>   objects_d_;
    thrust::device_vector<aabb_type>     aabbs_; // aabb of the node.
    thrust::device_vector<node_type>     nodes_;

    // these two are the map from index of leaf nodes to the contents of objects.
    // When some particles may have the same morton index, they are contained
    // to the same node. These arrays manages the relationships.
    thrust::device_vector<std::uint32_t> indices_;
    thrust::device_vector<std::uint32_t> ranges_;
};


// query object indices that potentially overlaps with query aabb.
//
// requirements:
// - OutputIterator should be writable and its object_type should be uint32_t
//
template<typename Real, typename Objects, bool IsConst, typename OutputIterator>
__device__
unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const aabb<Real>& q, const unsigned int max_buffer_size,
        OutputIterator outiter) noexcept
{
    using bvh_type   = detail::basic_device_bvh<Real, Objects, IsConst>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    index_type  stack[64]; // is it okay?
    index_type* stack_ptr = stack;
    *stack_ptr++ = 0; // root node is always 0

    unsigned int num_found = 0;
    do
    {
        const index_type node  = *--stack_ptr;
        const index_type L_idx = bvh.nodes[node].left_idx;
        const index_type R_idx = bvh.nodes[node].right_idx;

        if(intersects(q, bvh.aabbs[L_idx]))
        {
            const auto range_idx = bvh.nodes[L_idx].range_idx;
            if(range_idx != 0)
            {
                const index_type first = bvh.ranges[range_idx-1];
                const index_type last  = bvh.ranges[range_idx];
                for(index_type i = first; i < last; ++i)
                {
                    if(num_found < max_buffer_size)
                    {
                        *outiter++ = bvh.indices[i];
                    }
                    ++num_found;
                }
            }
            else // range_idx == 0 means the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(intersects(q, bvh.aabbs[R_idx]))
        {
            const auto range_idx = bvh.nodes[R_idx].range_idx;
            if(range_idx != 0)
            {
                const index_type first = bvh.ranges[range_idx-1];
                const index_type last  = bvh.ranges[range_idx];
                for(index_type i = first; i < last; ++i)
                {
                    if(num_found < max_buffer_size)
                    {
                        *outiter++ = bvh.indices[i];
                    }
                    ++num_found;
                }
            }
            else // range_idx == 0 means the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    }
    while (stack < stack_ptr);
    return num_found;
}

// query object index that is the nearst to the query point.
//
// requirements:
// - DistanceCalculator must be able to calc distance between a point to an object.
//
template<typename Real, typename Objects, bool IsConst,
         typename DistanceCalculator>
__device__
thrust::pair<unsigned int, Real> query_device_nearest_neighbor(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const vector_of_t<Real>& q, DistanceCalculator calc_dist) noexcept
{
    using bvh_type   = detail::basic_device_bvh<Real, Objects, IsConst>;
    using real_type  = typename bvh_type::real_type;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    // pair of {node_idx, mindist}
    thrust::pair<index_type, real_type>  stack[64];
    thrust::pair<index_type, real_type>* stack_ptr = stack;
    *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], q));

    unsigned int nearest = 0xFFFFFFFF;
    real_type dist_to_nearest_object = infinity<real_type>();
    do
    {
        const auto node  = *--stack_ptr;
        if(node.second > dist_to_nearest_object)
        {
            // if aabb mindist > already_found_mindist, it cannot have a nearest
            continue;
        }

        const index_type L_idx = bvh.nodes[node.first].left_idx;
        const index_type R_idx = bvh.nodes[node.first].right_idx;

        const aabb_type& L_box = bvh.aabbs[L_idx];
        const aabb_type& R_box = bvh.aabbs[R_idx];

        const real_type L_mindist = mindist(L_box, q);
        const real_type R_mindist = mindist(R_box, q);

        const real_type L_minmaxdist = minmaxdist(L_box, q);
        const real_type R_minmaxdist = minmaxdist(R_box, q);

       // there should be an object that locates within minmaxdist.
       dist_to_nearest_object = thrust::min(dist_to_nearest_object, L_minmaxdist);
       dist_to_nearest_object = thrust::min(dist_to_nearest_object, R_minmaxdist);

        if(L_mindist <= R_minmaxdist) // L is worth considering
        {
            const auto range_idx = bvh.nodes[L_idx].range_idx;
            if(range_idx != 0) // leaf node
            {
                const index_type first = bvh.ranges[range_idx-1];
                const index_type last  = bvh.ranges[range_idx];
                for(index_type i = first; i < last; ++i)
                {
                    const index_type idx = bvh.indices[i];
                    const real_type dist = calc_dist(q, bvh.objects[idx]);
                    if(dist <= dist_to_nearest_object)
                    {
                        dist_to_nearest_object = dist;
                        nearest = idx;
                    }
                }
            }
            else
            {
                *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
            }
        }
        if(R_mindist <= L_minmaxdist) // R is worth considering
        {
            const auto range_idx = bvh.nodes[R_idx].range_idx;
            if(range_idx != 0) // leaf node
            {
                const index_type first = bvh.ranges[range_idx-1];
                const index_type last  = bvh.ranges[range_idx];
                for(index_type i = first; i < last; ++i)
                {
                    const index_type idx = bvh.indices[i];
                    const real_type dist = calc_dist(q, bvh.objects[idx]);
                    if(dist <= dist_to_nearest_object)
                    {
                        dist_to_nearest_object = dist;
                        nearest = idx;
                    }
                }
            }
            else
            {
                *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
            }
        }
    }
    while (stack < stack_ptr);
    return thrust::make_pair(nearest, dist_to_nearest_object);
}


} // lbvh
#endif// LBVH_BVH_CUH
