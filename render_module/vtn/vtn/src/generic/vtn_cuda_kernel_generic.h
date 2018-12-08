#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/vtn_cuda_kernel_generic.h"
#else
int VTN_(BilinearSampler3DChannelFirst_updateOutput_wrap)(THCState* state, THCTensor* inputTensor, THCTensor* grid, THCTensor* output);
int VTN_(BilinearSampler3DChannelFirst_updateGradInput_wrap)(THCState* state, THCTensor* inputTensor, THCTensor* grid, THCTensor* gradInput, THCTensor* gradGrid, THCTensor* gradOutput);
#endif