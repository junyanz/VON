#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/vtn_cuda_generic.c"
#else

#include <stdbool.h>
#include <stdio.h>
#include "vtn_cuda_kernel_generic.h"


int VTN_(BilinearSampler3DChannelFirst_updateOutput)(THCTensor* inputTensor, THCTensor* grid, THCTensor* output) {
  int success = 0;
  success = VTN_(BilinearSampler3DChannelFirst_updateOutput_wrap)(state, inputTensor, grid, output);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}



int VTN_(BilinearSampler3DChannelFirst_updateGradInput)(THCTensor* inputTensor, THCTensor* grid, THCTensor* gradInput, THCTensor* gradGrid, THCTensor* gradOutput) {
  int success = 0;
  success = VTN_(BilinearSampler3DChannelFirst_updateGradInput_wrap)(state, inputTensor, grid, gradInput, gradGrid, gradOutput);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

#endif
