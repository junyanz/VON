#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "vtn_cuda_kernel_generic.h"

// this symbol will be resolved automatically by Pytorch libs
extern THCState *state;

int VTN_Cuda_BilinearSampler3DChannelFirst_updateOutput(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* output) {
  int success = 0;
  success = VTN_Cuda_BilinearSampler3DChannelFirst_updateOutput_wrap(state, inputTensor, grid, output);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}



int VTN_Cuda_BilinearSampler3DChannelFirst_updateGradInput(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* gradInput, THCudaTensor* gradGrid, THCudaTensor* gradOutput) {
  int success = 0;
  success = VTN_Cuda_BilinearSampler3DChannelFirst_updateGradInput_wrap(state, inputTensor, grid, gradInput, gradGrid, gradOutput);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}


int VTN_CudaDouble_BilinearSampler3DChannelFirst_updateOutput(THCudaDoubleTensor* inputTensor, THCudaDoubleTensor* grid, THCudaDoubleTensor* output) {
  int success = 0;
  success = VTN_CudaDouble_BilinearSampler3DChannelFirst_updateOutput_wrap(state, inputTensor, grid, output);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}



int VTN_CudaDouble_BilinearSampler3DChannelFirst_updateGradInput(THCudaDoubleTensor* inputTensor, THCudaDoubleTensor* grid, THCudaDoubleTensor* gradInput, THCudaDoubleTensor* gradGrid, THCudaDoubleTensor* gradOutput) {
  int success = 0;
  success = VTN_CudaDouble_BilinearSampler3DChannelFirst_updateGradInput_wrap(state, inputTensor, grid, gradInput, gradGrid, gradOutput);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

