#include <THC.h>
#include "vtn_cuda_kernel_generic.h"

#define VTN_(NAME) TH_CONCAT_4(VTN_, CReal, _, NAME)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// From http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz52Z2bz0Bc
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                            __longlong_as_double(assumed)));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif


#include "generic/vtn_cuda_kernel_generic.cu"
#include "generic/THCGenerateFloatTypes_noHalf.h"
