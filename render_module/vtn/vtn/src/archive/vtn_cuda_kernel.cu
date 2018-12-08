// Currently we only implement for FloatTensor
#include <THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "vtn_cuda_kernel.h"

#define real float

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

#undef WITHIN_BOUNDS
#define WITHIN_BOUNDS(x1, x2, x3, sz1, sz2, sz3) (x1 >= 0 && x1 < sz1 && x2 >= 0 && x2 < sz2 && x3 >= 0 && x3 < sz3)

#undef THCTensor_fastGet5d
#define THCTensor_fastGet5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)])

#undef THCTensor_fastSet5d
#define THCTensor_fastSet5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4, value) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)] = (value))

#undef SAFE_GET
#define SAFE_GET(data, n, c, x1, x2, x3, sz1, sz2, sz3, stride0, stride1, stride2, stride3, stride4) \
  (WITHIN_BOUNDS(x1, x2, x3, sz1, sz2, sz3)) ? THCTensor_fastGet5d(data, n, c, x1, x2, x3, stride0, stride1, stride2, stride3, stride4) : 0

// note that atomicAdd is given in cuda 8.0
// reference: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd
// manual definition: https://github.com/pytorch/pytorch/blob/502aaf39cf4a878f9e4f849e5f409573aa598aa9/aten/src/THC/THCAtomics.cuh
#undef SAFE_ADD
#define SAFE_ADD(data, n, c, x1, x2, x3, sz1, sz2, sz3, stride0, stride1, stride2, stride3, stride4, value)    \
  do {    \
    if (WITHIN_BOUNDS(x1, x2, x3, sz1, sz2, sz3)) {    \
      atomicAdd((data)+((n)*(stride0)+(c)*(stride1)+(x1)*(stride2)+(x2)*(stride3)+(x3)*(stride4)), value);   \
    }   \
  } while(0)

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
// #define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

// CUDA_KERNEL_LOOP is useful when each thread computes multiple values. Not useful here, though.
#undef CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// THCUNN check from torch/aten/src/THCUNN/common.h
#undef THCUNN_argCheck
#define THCUNN_argCheck(STATE, COND, ARG, T, FORMAT) \
  if (!(COND)) { \
    THCDescBuff s1 = THCudaTensor_sizeDesc(state, T); \
    THArgCheck(COND, ARG, FORMAT, s1.str);           \
  }

#undef THCUNN_check_dim_size
#define THCUNN_check_dim_size(STATE, T, DIM, DIM_SIZE, SIZE) \
  if (THCudaTensor_nDimension(STATE, T) != DIM ||             \
      THCudaTensor_size(STATE, T, DIM_SIZE) != SIZE) {        \
      THCDescBuff s1 = THCudaTensor_sizeDesc(state, T);       \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" \
              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#undef THCUNN_assertSameGPU
#define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

static inline void VTN_SpatialGridSamplerBilinear_shapeCheck_cuda(
    THCState *state,
    THCudaTensor *inputTensor,
    THCudaTensor *grid,
    THCudaTensor *gradOutput) {
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, inputTensor) == 5, 2, inputTensor,
      "5D input tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, grid) == 5, 2, grid,
      "5D grid tensor expected but got: %s");

  int nbatch   = THCudaTensor_size(state, inputTensor, 0);
  int channels = THCudaTensor_size(state, inputTensor, 1);
  // int isz1   = THCudaTensor_size(state, inputTensor, 2);
  // int isz2    = THCudaTensor_size(state, inputTensor, 3);
  // int isz3    = THCudaTensor_size(state, inputTensor, 4);
  int osz1   = THCudaTensor_size(state, grid, 1);
  int osz2   = THCudaTensor_size(state, grid, 2);
  int osz3   = THCudaTensor_size(state, grid, 3);

  THCUNN_check_dim_size(state, grid, 5, 0, nbatch);
  THCUNN_check_dim_size(state, grid, 5, 4, 3);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 5, 0, nbatch);
    THCUNN_check_dim_size(state, gradOutput, 5, 1, channels);
    THCUNN_check_dim_size(state, gradOutput, 5, 2, osz1);
    THCUNN_check_dim_size(state, gradOutput, 5, 3, osz2);
    THCUNN_check_dim_size(state, gradOutput, 5, 4, osz3);
  }
}




__launch_bounds__(CUDA_NUM_THREADS)
__global__ void VTN_BilinearSampler3DChannelFirst_updateOutput_cuda_kernel(
    float* inputTensor, // N * C * isz1 * isz2 * isz3
    int N,
    int C,
    int isz1, int isz2, int isz3,
    int isdb, int isdc, int isd1, int isd2, int isd3,
    float* grid,        // N * osz1 * osz2 * osz3 * 3
    int gsdb, int gsd1, int gsd2, int gsd3, int gsd4,
    float* output,      // N * C * osz1 * osz2 * osz3
    int osz1, int osz2, int osz3,
    int osdb, int osdc, int osd1, int osd2, int osd3,
    int nthreads) {

  int index;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index % N;
    const int ind1 = (index / N) % osz1;
    const int ind2 = (index / (N * osz1)) % osz2;
    const int ind3 = (index / (N * osz1 * osz2)) % osz3;
    int c;

    real ix = THCTensor_fastGet5d(grid, n, ind1, ind2, ind3, 0, gsdb, gsd1, gsd2, gsd3, gsd4);
    real iy = THCTensor_fastGet5d(grid, n, ind1, ind2, ind3, 1, gsdb, gsd1, gsd2, gsd3, gsd4);
    real iz = THCTensor_fastGet5d(grid, n, ind1, ind2, ind3, 2, gsdb, gsd1, gsd2, gsd3, gsd4);

    // normalize from [-1,1] to [0, sz-1]
    ix = ((ix + 1) / 2) * (isz1 - 1);
    iy = ((iy + 1) / 2) * (isz2 - 1);
    iz = ((iz + 1) / 2) * (isz3 - 1);

    // get 8 neighboring coordinates
    int ix_0 = floor(ix);
    int iy_0 = floor(iy);
    int iz_0 = floor(iz);
    int ix_1 = ix_0 + 1;
    int iy_1 = iy_0 + 1;
    int iz_1 = iz_0 + 1;

    // get coefficient to each neighbor
    real coeff_000 = (ix_1 - ix) * (iy_1 - iy) * (iz_1 - iz);
    real coeff_001 = (ix_1 - ix) * (iy_1 - iy) * (iz - iz_0);
    real coeff_010 = (ix_1 - ix) * (iy - iy_0) * (iz_1 - iz);
    real coeff_011 = (ix_1 - ix) * (iy - iy_0) * (iz - iz_0);
    real coeff_100 = (ix - ix_0) * (iy_1 - iy) * (iz_1 - iz);
    real coeff_101 = (ix - ix_0) * (iy_1 - iy) * (iz - iz_0);
    real coeff_110 = (ix - ix_0) * (iy - iy_0) * (iz_1 - iz);
    real coeff_111 = (ix - ix_0) * (iy - iy_0) * (iz - iz_0);

    for (c = 0; c < C; ++c) {
      real val_000 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_001 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_010 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_011 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_100 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_101 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_110 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_111 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real out_val = val_000 * coeff_000 + val_001 * coeff_001 + val_010 * coeff_010 + val_011 * coeff_011 + val_100 * coeff_100 + val_101 * coeff_101 + val_110 * coeff_110 + val_111 * coeff_111;
      THCTensor_fastSet5d(output, n, c, ind1, ind2, ind3, osdb, osdc, osd1, osd2, osd3, out_val);
    }

  }
}

int VTN_BilinearSampler3DChannelFirst_updateOutput_cuda_wrap(THCState* state, THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* output) {

  THCUNN_assertSameGPU(state, 3, inputTensor, grid, output);
  VTN_SpatialGridSamplerBilinear_shapeCheck_cuda(state, inputTensor, grid, output);

  // call kernel function
  int N = THCudaTensor_size(state, inputTensor, 0);
  int C = THCudaTensor_size(state, inputTensor, 1);
  int isz1 = THCudaTensor_size(state, inputTensor, 2);
  int isz2 = THCudaTensor_size(state, inputTensor, 3);
  int isz3 = THCudaTensor_size(state, inputTensor, 4);
  int osz1 = THCudaTensor_size(state, grid, 1);
  int osz2 = THCudaTensor_size(state, grid, 2);
  int osz3 = THCudaTensor_size(state, grid, 3);
  THCudaTensor_resize5d(state, output, N, C, osz1, osz2, osz3);

  int count = (N * osz1 * osz2 * osz3);
  VTN_BilinearSampler3DChannelFirst_updateOutput_cuda_kernel
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, inputTensor),
      N, C, isz1, isz2, isz3,
      THCudaTensor_stride(state, inputTensor, 0),
      THCudaTensor_stride(state, inputTensor, 1),
      THCudaTensor_stride(state, inputTensor, 2),
      THCudaTensor_stride(state, inputTensor, 3),
      THCudaTensor_stride(state, inputTensor, 4),
      THCudaTensor_data(state, grid),
      THCudaTensor_stride(state, grid, 0),
      THCudaTensor_stride(state, grid, 1),
      THCudaTensor_stride(state, grid, 2),
      THCudaTensor_stride(state, grid, 3),
      THCudaTensor_stride(state, grid, 4),
      THCudaTensor_data(state, output),
      osz1, osz2, osz3,
      THCudaTensor_stride(state, output, 0),
      THCudaTensor_stride(state, output, 1),
      THCudaTensor_stride(state, output, 2),
      THCudaTensor_stride(state, output, 3),
      THCudaTensor_stride(state, output, 4),
      count);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler3D update output: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

__launch_bounds__(CUDA_NUM_THREADS)
__global__ void VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda_kernel(
    float* inputTensor, // N * C * isz1 * isz2 * isz3
    int N,
    int C,
    int isz1, int isz2, int isz3,
    int isdb, int isdc, int isd1, int isd2, int isd3,
    float* grid,        // N * osz1 * osz2 * osz3 * 3
    int gsdb, int gsd1, int gsd2, int gsd3, int gsd4,
    float* gradInput,   // N * C * isz1 * isz2 * isz3
    int gisdb, int gisdc, int gisd1, int gisd2, int gisd3,
    float* gradGrid,    // N * osz1 * osz2 * osz3 * 3
    int ggsdb, int ggsd1, int ggsd2, int ggsd3, int ggsd4,
    float* gradOutput,  // N * C * osz1 * osz2 * osz3
    int osz1, int osz2, int osz3,
    int gosdb, int gosdc, int gosd1, int gosd2, int gosd3,
    int nthreads) {

  int index;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index % N;
    const int ind1 = (index / N) % osz1;
    const int ind2 = (index / (N * osz1)) % osz2;
    const int ind3 = (index / (N * osz1 * osz2)) % osz3;
    int c;

    real ix = THCTensor_fastGet5d(grid, n, ind1, ind2, ind3, 0, gsdb, gsd1, gsd2, gsd3, gsd4);
    real iy = THCTensor_fastGet5d(grid, n, ind1, ind2, ind3, 1, gsdb, gsd1, gsd2, gsd3, gsd4);
    real iz = THCTensor_fastGet5d(grid, n, ind1, ind2, ind3, 2, gsdb, gsd1, gsd2, gsd3, gsd4);

    real gix = 0;
    real giy = 0;
    real giz = 0;

    // normalize from [-1,1] to [0, sz-1]
    ix = ((ix + 1) / 2) * (isz1 - 1);
    iy = ((iy + 1) / 2) * (isz2 - 1);
    iz = ((iz + 1) / 2) * (isz3 - 1);

    // get 8 neighboring coordinates
    int ix_0 = floor(ix);
    int iy_0 = floor(iy);
    int iz_0 = floor(iz);
    int ix_1 = ix_0 + 1;
    int iy_1 = iy_0 + 1;
    int iz_1 = iz_0 + 1;

    // get coefficient to each neighbor
    real coeff_000 = (ix_1 - ix) * (iy_1 - iy) * (iz_1 - iz);
    real coeff_001 = (ix_1 - ix) * (iy_1 - iy) * (iz - iz_0);
    real coeff_010 = (ix_1 - ix) * (iy - iy_0) * (iz_1 - iz);
    real coeff_011 = (ix_1 - ix) * (iy - iy_0) * (iz - iz_0);
    real coeff_100 = (ix - ix_0) * (iy_1 - iy) * (iz_1 - iz);
    real coeff_101 = (ix - ix_0) * (iy_1 - iy) * (iz - iz_0);
    real coeff_110 = (ix - ix_0) * (iy - iy_0) * (iz_1 - iz);
    real coeff_111 = (ix - ix_0) * (iy - iy_0) * (iz - iz_0);
    real gradout;

    for (c = 0; c < C; ++c) {
      gradout = THCTensor_fastGet5d(gradOutput, n, c, ind1, ind2, ind3, gosdb, gosdc, gosd1, gosd2, gosd3);

      // calculate and set gradInput
      SAFE_ADD(gradInput, n, c, ix_0, iy_0, iz_0, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_000 * gradout);
      SAFE_ADD(gradInput, n, c, ix_0, iy_0, iz_1, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_001 * gradout);
      SAFE_ADD(gradInput, n, c, ix_0, iy_1, iz_0, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_010 * gradout);
      SAFE_ADD(gradInput, n, c, ix_0, iy_1, iz_1, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_011 * gradout);
      SAFE_ADD(gradInput, n, c, ix_1, iy_0, iz_0, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_100 * gradout);
      SAFE_ADD(gradInput, n, c, ix_1, iy_0, iz_1, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_101 * gradout);
      SAFE_ADD(gradInput, n, c, ix_1, iy_1, iz_0, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_110 * gradout);
      SAFE_ADD(gradInput, n, c, ix_1, iy_1, iz_1, isz1, isz2, isz3, gisdb, gisdc, gisd1, gisd2, gisd3, coeff_111 * gradout);

      // calculate gradGrid
      real val_000 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_001 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_010 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_011 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_100 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_101 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_110 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_0, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);
      real val_111 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_1, isz1, isz2, isz3, isdb, isdc, isd1, isd2, isd3);

      gix -= val_000 * gradout * (iy_1 - iy) * (iz_1 - iz);
      gix -= val_001 * gradout * (iy_1 - iy) * (iz - iz_0);
      gix -= val_010 * gradout * (iy - iy_0) * (iz_1 - iz);
      gix -= val_011 * gradout * (iy - iy_0) * (iz - iz_0);
      gix += val_100 * gradout * (iy_1 - iy) * (iz_1 - iz);
      gix += val_101 * gradout * (iy_1 - iy) * (iz - iz_0);
      gix += val_110 * gradout * (iy - iy_0) * (iz_1 - iz);
      gix += val_111 * gradout * (iy - iy_0) * (iz - iz_0);

      giy -= val_000 * gradout * (ix_1 - ix) * (iz_1 - iz);
      giy -= val_001 * gradout * (ix_1 - ix) * (iz - iz_0);
      giy += val_010 * gradout * (ix_1 - ix) * (iz_1 - iz);
      giy += val_011 * gradout * (ix_1 - ix) * (iz - iz_0);
      giy -= val_100 * gradout * (ix - ix_0) * (iz_1 - iz);
      giy -= val_101 * gradout * (ix - ix_0) * (iz - iz_0);
      giy += val_110 * gradout * (ix - ix_0) * (iz_1 - iz);
      giy += val_111 * gradout * (ix - ix_0) * (iz - iz_0);

      giz -= val_000 * gradout * (ix_1 - ix) * (iy_1 - iy);
      giz += val_001 * gradout * (ix_1 - ix) * (iy - iy_0);
      giz -= val_010 * gradout * (ix_1 - ix) * (iy_1 - iy);
      giz += val_011 * gradout * (ix_1 - ix) * (iy - iy_0);
      giz -= val_100 * gradout * (ix - ix_0) * (iy_1 - iy);
      giz += val_101 * gradout * (ix - ix_0) * (iy - iy_0);
      giz -= val_110 * gradout * (ix - ix_0) * (iy_1 - iy);
      giz += val_111 * gradout * (ix - ix_0) * (iy - iy_0);
    }

    // un-normalize gradGrid back to [-1,1]
    gix = gix * (isz1 - 1) / 2;
    giy = giy * (isz2 - 1) / 2;
    giz = giz * (isz3 - 1) / 2;

    real gix_old = THCTensor_fastGet5d(gradGrid, n, ind1, ind2, ind3, 0, ggsdb, ggsd1, ggsd2, ggsd3, ggsd4);
    real giy_old = THCTensor_fastGet5d(gradGrid, n, ind1, ind2, ind3, 1, ggsdb, ggsd1, ggsd2, ggsd3, ggsd4);
    real giz_old = THCTensor_fastGet5d(gradGrid, n, ind1, ind2, ind3, 2, ggsdb, ggsd1, ggsd2, ggsd3, ggsd4);

    THCTensor_fastSet5d(gradGrid, n, ind1, ind2, ind3, 0, ggsdb, ggsd1, ggsd2, ggsd3, ggsd4, gix_old + gix);
    THCTensor_fastSet5d(gradGrid, n, ind1, ind2, ind3, 1, ggsdb, ggsd1, ggsd2, ggsd3, ggsd4, giy_old + giy);
    THCTensor_fastSet5d(gradGrid, n, ind1, ind2, ind3, 2, ggsdb, ggsd1, ggsd2, ggsd3, ggsd4, giz_old + giz);
  }

}



int VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda_wrap(THCState* state, THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* gradInput, THCudaTensor* gradGrid, THCudaTensor* gradOutput) {

  THCUNN_assertSameGPU(state, 5, inputTensor, gradInput, grid, gradGrid, gradOutput);
  VTN_SpatialGridSamplerBilinear_shapeCheck_cuda(state, inputTensor, grid, gradOutput);


  // call kernel function
  int N = THCudaTensor_size(state, inputTensor, 0);
  int C = THCudaTensor_size(state, inputTensor, 1);
  int isz1 = THCudaTensor_size(state, inputTensor, 2);
  int isz2 = THCudaTensor_size(state, inputTensor, 3);
  int isz3 = THCudaTensor_size(state, inputTensor, 4);
  int osz1 = THCudaTensor_size(state, grid, 1);
  int osz2 = THCudaTensor_size(state, grid, 2);
  int osz3 = THCudaTensor_size(state, grid, 3);
  THCudaTensor_resize5d(state, gradInput, N, C, isz1, isz2, isz3);
  THCudaTensor_resize5d(state, gradGrid, N, osz1, osz2, osz3, 3);
  THCudaTensor_zero(state, gradInput);
  THCudaTensor_zero(state, gradGrid);

  int count = (N * osz1 * osz2 * osz3);
  VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda_kernel
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, inputTensor),
      N, C, isz1, isz2, isz3,
      THCudaTensor_stride(state, inputTensor, 0),
      THCudaTensor_stride(state, inputTensor, 1),
      THCudaTensor_stride(state, inputTensor, 2),
      THCudaTensor_stride(state, inputTensor, 3),
      THCudaTensor_stride(state, inputTensor, 4),
      THCudaTensor_data(state, grid),
      THCudaTensor_stride(state, grid, 0),
      THCudaTensor_stride(state, grid, 1),
      THCudaTensor_stride(state, grid, 2),
      THCudaTensor_stride(state, grid, 3),
      THCudaTensor_stride(state, grid, 4),
      THCudaTensor_data(state, gradInput),
      THCudaTensor_stride(state, gradInput, 0),
      THCudaTensor_stride(state, gradInput, 1),
      THCudaTensor_stride(state, gradInput, 2),
      THCudaTensor_stride(state, gradInput, 3),
      THCudaTensor_stride(state, gradInput, 4),
      THCudaTensor_data(state, gradGrid),
      THCudaTensor_stride(state, gradGrid, 0),
      THCudaTensor_stride(state, gradGrid, 1),
      THCudaTensor_stride(state, gradGrid, 2),
      THCudaTensor_stride(state, gradGrid, 3),
      THCudaTensor_stride(state, gradGrid, 4),
      THCudaTensor_data(state, gradOutput),
      osz1, osz2, osz3,
      THCudaTensor_stride(state, gradOutput, 0),
      THCudaTensor_stride(state, gradOutput, 1),
      THCudaTensor_stride(state, gradOutput, 2),
      THCudaTensor_stride(state, gradOutput, 3),
      THCudaTensor_stride(state, gradOutput, 4),
      count);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler3D update gradInput: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}
#undef WITHIN_BOUNDS
#undef THCTensor_fastGet5d
#undef THCTensor_fastSet5d
#undef SAFE_GET
#undef SAFE_ADD
#undef MIN
#undef MAX
#undef CUDA_KERNEL_LOOP
#undef THCUNN_argCheck
#undef THCUNN_check_dim_size
#undef THCUNN_assertSameGPU
