#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/vtn_cuda_generic.h"
#else
int VTN_(BilinearSampler3DChannelFirst_updateOutput)(THCState* state, THCTensor* inputTensor, THCTensor* grid, THCTensor* output);
int VTN_(BilinearSampler3DChannelFirst_updateGradInput)(THCState* state, THCTensor* inputTensor, THCTensor* grid, THCTensor* gradInput, THCTensor* gradGrid, THCTensor* gradOutput);
#endif