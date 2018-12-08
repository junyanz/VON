
int VTN_Cuda_BilinearSampler3DChannelFirst_updateOutput(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* output);
int VTN_Cuda_BilinearSampler3DChannelFirst_updateGradInput(THCudaTensor* inputTensor, THCudaTensor* grid, THCudaTensor* gradInput, THCudaTensor* gradGrid, THCudaTensor* gradOutput);
int VTN_CudaDouble_BilinearSampler3DChannelFirst_updateOutput(THCudaDoubleTensor* inputTensor, THCudaDoubleTensor* grid, THCudaDoubleTensor* output);
int VTN_CudaDouble_BilinearSampler3DChannelFirst_updateGradInput(THCudaDoubleTensor* inputTensor, THCudaDoubleTensor* grid, THCudaDoubleTensor* gradInput, THCudaDoubleTensor* gradGrid, THCudaDoubleTensor* gradOutput);
