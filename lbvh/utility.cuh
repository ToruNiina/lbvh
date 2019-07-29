#ifndef LBVH_UTILITY_CUH
#define LBVH_UTILITY_CUH
#include <vector_types.h>

namespace lbvh
{

template<typename T> struct vector_of;
template<> struct vector_of<float>  {using type = float4;};
template<> struct vector_of<double> {using type = double4;};

template<typename T>
using vector_of_t = typename vector_of<T>::type;

} // lbvh
#endif// LBVH_UTILITY_CUH
