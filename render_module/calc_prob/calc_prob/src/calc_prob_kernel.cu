#include <THC.h>
#include <stdbool.h>
#include <stdio.h>
#include <cuda.h>
#include "calc_prob_kernel.h"

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

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

#define EPS 1e-5

#define GET_DIRECT_3d(data, x0, x1, x2, sd0, sd1, sd2) \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2)])

#define SET_DIRECT_3d(data, x0, x1, x2, sd0, sd1, sd2, v)        \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) ]) = v

#define GET_DIRECT_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3)         \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3)])

#define SET_DIRECT_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3, v)        \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3)]) = v

#define GET_DIRECT_5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)])

#define SET_DIRECT_5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4, value) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)] = (value))

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


#define THCUNN_argCheck(STATE, COND, ARG, T, FORMAT) \
  if (!(COND)) { \
    THCDescBuff s1 = THCudaTensor_sizeDesc(state, T); \
    THArgCheck(COND, ARG, FORMAT, s1.str);           \
  }

#define THCUNN_check_dim_size(STATE, T, DIM, DIM_SIZE, SIZE) \
  if (THCudaTensor_nDimension(STATE, T) != DIM ||             \
      THCudaTensor_size(STATE, T, DIM_SIZE) != SIZE) {        \
      THCDescBuff s1 = THCudaTensor_sizeDesc(state, T);       \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" \
              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

/*static inline void gridgen_shapecheck(THCState* state,
                                    THCudaTensor* grid,
                                    THCudaTensor* camdist,
                                    THCudaTensor* fl){
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, grid) == 5, 2, grid,
      "5D input Grid tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, camdist) == 2, 2, camdist,
      "3D input camdist tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, fl) == 2, 2, fl,
      "3D input fl tensor expected but got: %s");

  int nbatch = THCudaTensor_size(state, grid, 0);

  //fprintf(stderr, "argcheck + size pass\n");
  THCUNN_check_dim_size(state, grid, 5, 4, 3);
  THCUNN_check_dim_size(state, camdist, 1, 0, nbatch);
  THCUNN_check_dim_size(state, fl, 1, 0, nbatch);
  }*/

static inline void calc_prob_shapecheck(THCState* state,
                                    THCudaTensor* prob_in,
                                    THCudaTensor* prob_out){
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, prob_in) == 5, 2, prob_in,
      "5D input Grid tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, prob_out) == 5, 2, prob_out,
      "3D input camdist tensor expected but got: %s");

  int nbatch = THCudaTensor_size(state, prob_in, 0);
  int nc = THCudaTensor_size(state, prob_in, 1);
  int sx = THCudaTensor_size(state, prob_in, 2);
  int sy = THCudaTensor_size(state, prob_in, 3);
  int sz = THCudaTensor_size(state, prob_in, 4);
  //fprintf(stderr, "argcheck + size pass\n");
  THCUNN_check_dim_size(state, prob_out, 5, 0, nbatch);
  THCUNN_check_dim_size(state, prob_out, 5, 1, nc);
  THCUNN_check_dim_size(state, prob_out, 5, 2, sx);
  THCUNN_check_dim_size(state, prob_out, 5, 3, sy);
  THCUNN_check_dim_size(state, prob_out, 5, 4, sz);
}

__launch_bounds__(CUDA_NUM_THREADS)
__global__ void calc_stop_forward_kernel(float* prob_in,
                                         int N, int NC,
                                         int pszx, int pszy, int pszz,
                                         int psdn, int psdc, int psdx, int psdy, int psdz,
                                         float* stop_prob,
                                         int ssdn, int ssdc, int ssdx, int ssdy, int ssdz,
                                         int nthreads){
  CUDA_KERNEL_LOOP(index, nthreads){
    const int n = index % N;
    const int ind_c = (index / N) % NC;
    const int ind_x = (index / (N * NC)) % pszx;
    const int ind_y = (index / (N * NC * pszx) ) % pszy;
    float cur_prob = 0.0;
    float prev_prob = 0.0;
    float prev_result = 0.0;
    float temp_result = 0.0;
    for(int z=0; z < pszz; z++){
      if(z==0){
        cur_prob = GET_DIRECT_5d(prob_in, n, ind_c, ind_x, ind_y, z, psdn, psdc, psdx, psdy, psdz);
        SET_DIRECT_5d(stop_prob, n, ind_c, ind_x, ind_y, z, ssdn, ssdc, ssdx, ssdy, ssdz, cur_prob);
      }
      else{
        prev_result = GET_DIRECT_5d(stop_prob, n, ind_c, ind_x, ind_y, z-1, ssdn, ssdc, ssdx, ssdy, ssdz);
        prev_prob =  GET_DIRECT_5d(prob_in, n, ind_c, ind_x, ind_y, z-1, psdn, psdc, psdx, psdy, psdz);
        cur_prob =  GET_DIRECT_5d(prob_in, n, ind_c, ind_x, ind_y, z, psdn, psdc, psdx, psdy, psdz);
        temp_result = prev_result * ((1.0 / prev_prob) - 1.0) * cur_prob;
        SET_DIRECT_5d(stop_prob, n, ind_c, ind_x, ind_y, z, ssdn, ssdc, ssdx, ssdy, ssdz, temp_result);
      }
    }
  }
}

__launch_bounds__(CUDA_NUM_THREADS)
__global__ void calc_stop_backward_kernel(float* prob_in,
                                          int N, int NC,
                                          int pszx, int pszy, int pszz,
                                          int psdn, int psdc, int psdx, int psdy, int psdz,
                                          float* stop_prob_weighted,
                                          int ssdn, int ssdc, int ssdx, int ssdy, int ssdz,
                                          float* grad_out,
                                          int gsdn, int gsdc, int gsdx, int gsdy, int gsdz,
                                          int nthreads){
  CUDA_KERNEL_LOOP(index, nthreads){
    const int n = index % N;
    const int ind_c = (index / N) % NC;
    const int ind_x = (index / (N * NC)) % pszx;
    const int ind_y = (index / (N * NC * pszx) ) % pszy;
    float head = 0.0;
    float delay_sum = 0.0;
    float cur_stop_prob = 0.0;
    float cur_prob = 0.0;
    float prev_prob = 0.0;
    float v1 = 0.0;
    float v2 = 0.0;
    float v3 = 0.0;

    for(int z=pszz-1; z>=0; z--){
      if(z==pszz-1){
        cur_prob = GET_DIRECT_5d(prob_in, n, ind_c, ind_x, ind_y, z, psdn, psdc, psdx, psdy, psdz);
        cur_stop_prob = GET_DIRECT_5d(stop_prob_weighted, n, ind_c, ind_x, ind_y, z, ssdn, ssdc, ssdx, ssdy, ssdz);
        head = cur_stop_prob/cur_prob;
        SET_DIRECT_5d(grad_out, n, ind_c, ind_x, ind_y, z, gsdn, gsdc, gsdx, gsdy, gsdz, head);
      }
      else{
        cur_prob = GET_DIRECT_5d(prob_in, n, ind_c, ind_x, ind_y, z, psdn, psdc, psdx, psdy, psdz);
        prev_prob = GET_DIRECT_5d(prob_in, n, ind_c, ind_x, ind_y, z+1, psdn, psdc, psdx, psdy, psdz);
        cur_stop_prob = GET_DIRECT_5d(stop_prob_weighted, n, ind_c, ind_x, ind_y, z, ssdn, ssdc, ssdx, ssdy, ssdz);
        v1 = cur_stop_prob / cur_prob;
        v2 = head * prev_prob / (1.0 - cur_prob);
        v3 = delay_sum * ( 1.0 - prev_prob) / ( 1 - cur_prob);
        delay_sum = v2+v3;
        head = v1;
        SET_DIRECT_5d(grad_out, n, ind_c, ind_x, ind_y, z, gsdn, gsdc, gsdx, gsdy, gsdz, v1-v2-v3);
      }
    }
  }
}

int calc_prob_forward_wrap(THCState* state, THCudaTensor* prob_in, THCudaTensor* prob_out){
  //fprintf(stderr,"calling cuda!!\n");
  THCUNN_assertSameGPU(state, 2, prob_in, prob_out);
  calc_prob_shapecheck(state, prob_in, prob_out);
  
  int N = THCudaTensor_size(state, prob_in, 0);
  int NC = THCudaTensor_size(state, prob_in, 1);
  int pszx = THCudaTensor_size(state, prob_in, 2);
  int pszy = THCudaTensor_size(state, prob_in, 3);
  int pszz = THCudaTensor_size(state, prob_in, 4);
  int count_ray = (N*NC*pszx*pszy);
  THCudaTensor_resize5d(state, prob_out, N, NC, pszx, pszy, pszz);
  THCudaTensor_zero(state, prob_out);
  calc_stop_forward_kernel
    <<<GET_BLOCKS(count_ray), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, prob_in),
    N,NC, pszx, pszy, pszz,
    THCudaTensor_stride(state, prob_in, 0),
    THCudaTensor_stride(state, prob_in, 1),
    THCudaTensor_stride(state, prob_in, 2),
    THCudaTensor_stride(state, prob_in, 3),
    THCudaTensor_stride(state, prob_in, 4),
    THCudaTensor_data(state, prob_out),
    THCudaTensor_stride(state, prob_out, 0),
    THCudaTensor_stride(state, prob_out, 1),
    THCudaTensor_stride(state, prob_out, 2),
    THCudaTensor_stride(state, prob_out, 3),
    THCudaTensor_stride(state, prob_out, 4),
    count_ray);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in calc_prob foward: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}
int calc_prob_backward_wrap(THCState* state, THCudaTensor* prob_in, THCudaTensor* stop_prob_weighted, THCudaTensor*  grad_out){
  //fprintf(stderr,"calling cuda!!\n");
  THCUNN_assertSameGPU(state, 3, prob_in, stop_prob_weighted, grad_out);
  int N = THCudaTensor_size(state, prob_in, 0);
  int NC = THCudaTensor_size(state, prob_in, 1);
  int pszx = THCudaTensor_size(state, prob_in, 2);
  int pszy = THCudaTensor_size(state, prob_in, 3);
  int pszz = THCudaTensor_size(state, prob_in, 4);
  int count_ray = (N*NC*pszx*pszy);
  THCudaTensor_resize5d(state, grad_out, N, NC, pszx, pszy, pszz);
  THCudaTensor_zero(state, grad_out);
  calc_stop_backward_kernel
    <<<GET_BLOCKS(count_ray), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, prob_in),
    N,NC, pszx, pszy, pszz,
    THCudaTensor_stride(state, prob_in, 0),
    THCudaTensor_stride(state, prob_in, 1),
    THCudaTensor_stride(state, prob_in, 2),
    THCudaTensor_stride(state, prob_in, 3),
    THCudaTensor_stride(state, prob_in, 4),
    THCudaTensor_data(state, stop_prob_weighted),
    THCudaTensor_stride(state, stop_prob_weighted, 0),
    THCudaTensor_stride(state, stop_prob_weighted, 1),
    THCudaTensor_stride(state, stop_prob_weighted, 2),
    THCudaTensor_stride(state, stop_prob_weighted, 3),
    THCudaTensor_stride(state, stop_prob_weighted, 4),
    THCudaTensor_data(state, grad_out),
    THCudaTensor_stride(state, grad_out, 0),
    THCudaTensor_stride(state, grad_out, 1),
    THCudaTensor_stride(state, grad_out, 2),
    THCudaTensor_stride(state, grad_out, 3),
    THCudaTensor_stride(state, grad_out, 4),
    count_ray);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in calc_prob foward: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

