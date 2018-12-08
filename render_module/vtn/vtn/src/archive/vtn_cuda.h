
int VTN_BilinearSampler3DChannelFirst_updateOutput_cuda(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* output);
int VTN_BilinearSampler3DChannelFirst_updateGradInput_cuda(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* gradInput, THCudaTensor* gradGrid, THCudaTensor* gradOutput);

