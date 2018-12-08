#include <THC/THC.h>

#define VTN_(NAME) TH_CONCAT_4(VTN_, CReal, _, NAME)

// this symbol will be resolved automatically by Pytorch libs
extern THCState *state;

#include "generic/vtn_cuda_generic.c"
#include "generic/THCGenerateFloatTypes_noHalf.h"
