#ifdef __cplusplus
extern "C" {
#endif

int VTN_BilinearSampler3DChannelFirst_updateOutput_cuda_wrap(THCState* state, THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* output);
int VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda_wrap(THCState* state, THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* gradInput, THCudaTensor* gradGrid, THCudaTensor* gradOutput);




#ifdef __cplusplus
}
#endif