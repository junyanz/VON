#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "vtn_cuda_kernel.h"

#define real float

// this symbol will be resolved automatically by Pytorch libs
extern THCState *state;

int VTN_BilinearSampler3DChannelFirst_updateOutput_cuda(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* output) {
  int success = 0;
  success = VTN_BilinearSampler3DChannelFirst_updateOutput_cuda_wrap(state, inputTensor, grid, output);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}



int VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* gradInput, THCudaTensor* gradGrid, THCudaTensor* gradOutput) {
  int success = 0;
  success = VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda_wrap(state, inputTensor, grid, gradInput, gradGrid, gradOutput);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

