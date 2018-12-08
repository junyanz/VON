
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/vtn.h"
#else

void VTN_(BilinearSampler3DChannelFirst_updateOutput)(THTensor *inputTensor, THTensor *grid, THTensor *output);
void VTN_(BilinearSampler3DChannelFirst_updateGradInput)(THTensor *inputTensor, THTensor *grid, THTensor *gradInputTensor,
                    THTensor *gradGrid, THTensor *gradOutput);
#endif