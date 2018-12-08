#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/vtn.c"
#else

#include <stdbool.h>
#include <stdio.h>

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

#undef THTensor_fastGet5d
#define THTensor_fastGet5d(self, x0, x1, x2, x3, x4)                        \
  (( THTensor_(data)(self)+THTensor_(storageOffset)(self))[(x0)*THTensor_(stride)(self, 0)+(x1)*THTensor_(stride)(self, 1)+(x2)*THTensor_(stride)(self, 2)+(x3)*THTensor_(stride)(self, 3)+(x4)*THTensor_(stride)(self, 4)])

#undef THTensor_fastSet5d
#define THTensor_fastSet5d(self, x0, x1, x2, x3, x4, value)                 \
  ( (THTensor_(data)(self)+THTensor_(storageOffset)(self))[(x0)*THTensor_(stride)(self, 0)+(x1)*THTensor_(stride)(self, 1)+(x2)*THTensor_(stride)(self, 2)+(x3)*THTensor_(stride)(self, 3)+(x4)*THTensor_(stride)(self, 4)]= value)

#undef SAFE_GET
#define SAFE_GET(input, n, c, x1, x2, x3, sz1, sz2, sz3) x1 >= 0 && x1 < sz1 && x2 >= 0 \
  && x2 < sz2 && x3 >= 0 && x3 < sz3 ? THTensor_fastGet5d(input, n, c, x1, x2, x3) : 0

#undef CLIP_COORDINATES
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

#undef SAFE_ADD
#define SAFE_ADD(input, n, c, x1, x2, x3, sz1, sz2, sz3, value)    \
  do {                \
    if (x1 >= 0 && x1 < sz1 && x2 >= 0 && x2 < sz2 && x3 >=0 && x3 < sz3) {      \
      real old_value = THTensor_fastGet5d(input, n, c, x1, x2, x3); \
      THTensor_fastSet5d(input, n, c, x1, x2, x3, value + old_value); \
    }               \
  } while(0)

// THNN check from torch/aten/src/THNN/init.c
#undef THNN_ARGCHECK
#define THNN_ARGCHECK(COND, ARG, T, FORMAT) \
  if (!(COND)) {        \
    THDescBuff s1 = THTensor_(sizeDesc)(T); \
    THArgCheck(COND, ARG, FORMAT, s1.str);  \
  }

#undef THNN_CHECK_DIM_SIZE
#define THNN_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE)     \
  if (THTensor_(nDimension)(T) != DIM ||        \
      THTensor_(size)(T, DIM_SIZE) != SIZE) {       \
      THDescBuff s1 = THTensor_(sizeDesc)(T);       \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" \
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }


static inline void VTN_(BilinearSampler3DChannelFirst_shapeCheck)
     (THTensor *inputTensor, THTensor *grid, THTensor *gradOutput) {
  THNN_ARGCHECK(THTensor_(nDimension)(inputTensor) == 5, 2, inputTensor,
    "5D input tensor expected but got: %s");
  THNN_ARGCHECK(THTensor_(nDimension)(grid) == 5, 2, grid,
    "5D grid tensor expected but got: %s");

  int nbatch   = THTensor_(size)(inputTensor, 0);
  int channels = THTensor_(size)(inputTensor, 1);
  // int isz1   = THTensor_(size)(inputTensor, 2);
  // int isz2    = THTensor_(size)(inputTensor, 3);
  // int isz3    = THTensor_(size)(inputTensor, 4);
  int osz1   = THTensor_(size)(grid, 1);
  int osz2   = THTensor_(size)(grid, 2);
  int osz3   = THTensor_(size)(grid, 3);

  THNN_CHECK_DIM_SIZE(grid, 5, 0, nbatch);
  THNN_CHECK_DIM_SIZE(grid, 5, 4, 3);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 0, nbatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 1, channels);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 2, osz1);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 3, osz2);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 4, osz3);
  }
}

void VTN_(BilinearSampler3DChannelFirst_updateOutput)(THTensor *inputTensor, THTensor *grid, THTensor *output) {
  VTN_(BilinearSampler3DChannelFirst_shapeCheck)(inputTensor, grid, NULL);
  int batchsize = THTensor_(size)(inputTensor, 0);
  int inputTensor_channels = THTensor_(size)(inputTensor, 1);
  int inputTensor_sz1 = THTensor_(size)(inputTensor, 2);
  int inputTensor_sz2 = THTensor_(size)(inputTensor, 3);
  int inputTensor_sz3 = THTensor_(size)(inputTensor, 4);

  int output_sz1 = THTensor_(size)(output, 2);
  int output_sz2 = THTensor_(size)(output, 3);
  int output_sz3 = THTensor_(size)(output, 4);

  int n, c, ind1, ind2, ind3;
#pragma omp parallel for private(n, c, ind1, ind2, ind3)
  for (n=0; n < batchsize; ++n){
    for (ind1=0; ind1 < output_sz1; ++ind1) {
      for (ind2=0; ind2 < output_sz2; ++ind2) {
        for (ind3=0; ind3 < output_sz3; ++ind3) {
          // get the corresponding input coordinate from grid
          real ix = THTensor_fastGet5d(grid, n, ind1, ind2, ind3, 0);
          real iy = THTensor_fastGet5d(grid, n, ind1, ind2, ind3, 1);
          real iz = THTensor_fastGet5d(grid, n, ind1, ind2, ind3, 2);

          // normalize from [-1,1] to [0, sz-1]
          ix = ((ix + 1) / 2) * (inputTensor_sz1 - 1);
          iy = ((iy + 1) / 2) * (inputTensor_sz2 - 1);
          iz = ((iz + 1) / 2) * (inputTensor_sz3 - 1);

          // get 8 neighboring coordinates
          int ix_0 = floor(ix);
          int iy_0 = floor(iy);
          int iz_0 = floor(iz);
          int ix_1 = ix_0 + 1;
          int iy_1 = iy_0 + 1;
          int iz_1 = iz_0 + 1;

          // get coefficient to each neighbor
          real coeff_000 = (ix_1 - ix) * (iy_1 - iy) * (iz_1 - iz);
          real coeff_001 = (ix_1 - ix) * (iy_1 - iy) * (iz - iz_0);
          real coeff_010 = (ix_1 - ix) * (iy - iy_0) * (iz_1 - iz);
          real coeff_011 = (ix_1 - ix) * (iy - iy_0) * (iz - iz_0);
          real coeff_100 = (ix - ix_0) * (iy_1 - iy) * (iz_1 - iz);
          real coeff_101 = (ix - ix_0) * (iy_1 - iy) * (iz - iz_0);
          real coeff_110 = (ix - ix_0) * (iy - iy_0) * (iz_1 - iz);
          real coeff_111 = (ix - ix_0) * (iy - iy_0) * (iz - iz_0);


          // calculate weighted voxel value
          for (c = 0; c < inputTensor_channels; ++c) {
            real val_000 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_001 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_010 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_011 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_100 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_101 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_110 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_111 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real out_val = val_000 * coeff_000 + val_001 * coeff_001 + val_010 * coeff_010 + val_011 * coeff_011 + val_100 * coeff_100 + val_101 * coeff_101 + val_110 * coeff_110 + val_111 * coeff_111;
            THTensor_fastSet5d(output, n, c, ind1, ind2, ind3, out_val);
          }
        }
      }
    }
  }
}

void VTN_(BilinearSampler3DChannelFirst_updateGradInput)(THTensor *inputTensor, THTensor *grid, THTensor *gradInput,
                    THTensor *gradGrid, THTensor *gradOutput) {
  VTN_(BilinearSampler3DChannelFirst_shapeCheck)(inputTensor, grid, gradOutput);
  int batchsize = THTensor_(size)(inputTensor, 0);
  int inputTensor_channels = THTensor_(size)(inputTensor, 1);
  int inputTensor_sz1 = THTensor_(size)(inputTensor, 2);
  int inputTensor_sz2 = THTensor_(size)(inputTensor, 3);
  int inputTensor_sz3 = THTensor_(size)(inputTensor, 4);

  int output_sz1 = THTensor_(size)(grid, 1);
  int output_sz2 = THTensor_(size)(grid, 2);
  int output_sz3 = THTensor_(size)(grid, 3);

  THTensor_(resize5d)(gradInput, batchsize, inputTensor_channels, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
  THTensor_(resize5d)(gradGrid, batchsize, output_sz1, output_sz2, output_sz3, 3);
  THTensor_(zero)(gradInput);
  THTensor_(zero)(gradGrid);

  // loop over each output voxel
  int n, c, ind1, ind2, ind3;
#pragma omp parallel for private(n, c, ind1, ind2, ind3)
  for (n = 0; n < batchsize; ++n) {
    for (ind1 = 0; ind1 < output_sz1; ++ind1) {
      for (ind2 = 0; ind2 < output_sz2; ++ind2) {
        for (ind3 = 0; ind3 < output_sz3; ++ind3) {
          // get the corresponding input coordinate from grid
          real ix = THTensor_fastGet5d(grid, n, ind1, ind2, ind3, 0);
          real iy = THTensor_fastGet5d(grid, n, ind1, ind2, ind3, 1);
          real iz = THTensor_fastGet5d(grid, n, ind1, ind2, ind3, 2);

          real gix = 0;
          real giy = 0;
          real giz = 0;

          // normalize from [-1,1] to [0, sz-1]
          ix = ((ix + 1) / 2) * (inputTensor_sz1 - 1);
          iy = ((iy + 1) / 2) * (inputTensor_sz2 - 1);
          iz = ((iz + 1) / 2) * (inputTensor_sz3 - 1);

          // get 8 neighboring coordinates
          int ix_0 = floor(ix);
          int iy_0 = floor(iy);
          int iz_0 = floor(iz);
          int ix_1 = ix_0 + 1;
          int iy_1 = iy_0 + 1;
          int iz_1 = iz_0 + 1;

          // get coefficient to each neighbor
          real coeff_000 = (ix_1 - ix) * (iy_1 - iy) * (iz_1 - iz);
          real coeff_001 = (ix_1 - ix) * (iy_1 - iy) * (iz - iz_0);
          real coeff_010 = (ix_1 - ix) * (iy - iy_0) * (iz_1 - iz);
          real coeff_011 = (ix_1 - ix) * (iy - iy_0) * (iz - iz_0);
          real coeff_100 = (ix - ix_0) * (iy_1 - iy) * (iz_1 - iz);
          real coeff_101 = (ix - ix_0) * (iy_1 - iy) * (iz - iz_0);
          real coeff_110 = (ix - ix_0) * (iy - iy_0) * (iz_1 - iz);
          real coeff_111 = (ix - ix_0) * (iy - iy_0) * (iz - iz_0);

          for (c = 0; c < inputTensor_channels; ++c) {
            real gradout = THTensor_fastGet5d(gradOutput, n, c, ind1, ind2, ind3);

            // calculate and set gradInput
            SAFE_ADD(gradInput, n, c, ix_0, iy_0, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_000 * gradout);
            SAFE_ADD(gradInput, n, c, ix_0, iy_0, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_001 * gradout);
            SAFE_ADD(gradInput, n, c, ix_0, iy_1, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_010 * gradout);
            SAFE_ADD(gradInput, n, c, ix_0, iy_1, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_011 * gradout);
            SAFE_ADD(gradInput, n, c, ix_1, iy_0, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_100 * gradout);
            SAFE_ADD(gradInput, n, c, ix_1, iy_0, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_101 * gradout);
            SAFE_ADD(gradInput, n, c, ix_1, iy_1, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_110 * gradout);
            SAFE_ADD(gradInput, n, c, ix_1, iy_1, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3, coeff_111 * gradout);

            // calculate gradGrid
            real val_000 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_001 = SAFE_GET(inputTensor, n, c, ix_0, iy_0, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_010 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_011 = SAFE_GET(inputTensor, n, c, ix_0, iy_1, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_100 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_101 = SAFE_GET(inputTensor, n, c, ix_1, iy_0, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_110 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_0, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);
            real val_111 = SAFE_GET(inputTensor, n, c, ix_1, iy_1, iz_1, inputTensor_sz1, inputTensor_sz2, inputTensor_sz3);

            gix -= val_000 * gradout * (iy_1 - iy) * (iz_1 - iz);
            gix -= val_001 * gradout * (iy_1 - iy) * (iz - iz_0);
            gix -= val_010 * gradout * (iy - iy_0) * (iz_1 - iz);
            gix -= val_011 * gradout * (iy - iy_0) * (iz - iz_0);
            gix += val_100 * gradout * (iy_1 - iy) * (iz_1 - iz);
            gix += val_101 * gradout * (iy_1 - iy) * (iz - iz_0);
            gix += val_110 * gradout * (iy - iy_0) * (iz_1 - iz);
            gix += val_111 * gradout * (iy - iy_0) * (iz - iz_0);

            giy -= val_000 * gradout * (ix_1 - ix) * (iz_1 - iz);
            giy -= val_001 * gradout * (ix_1 - ix) * (iz - iz_0);
            giy += val_010 * gradout * (ix_1 - ix) * (iz_1 - iz);
            giy += val_011 * gradout * (ix_1 - ix) * (iz - iz_0);
            giy -= val_100 * gradout * (ix - ix_0) * (iz_1 - iz);
            giy -= val_101 * gradout * (ix - ix_0) * (iz - iz_0);
            giy += val_110 * gradout * (ix - ix_0) * (iz_1 - iz);
            giy += val_111 * gradout * (ix - ix_0) * (iz - iz_0);

            giz -= val_000 * gradout * (ix_1 - ix) * (iy_1 - iy);
            giz += val_001 * gradout * (ix_1 - ix) * (iy_1 - iy);
            giz -= val_010 * gradout * (ix_1 - ix) * (iy - iy_0);
            giz += val_011 * gradout * (ix_1 - ix) * (iy - iy_0);
            giz -= val_100 * gradout * (ix - ix_0) * (iy_1 - iy);
            giz += val_101 * gradout * (ix - ix_0) * (iy_1 - iy);
            giz -= val_110 * gradout * (ix - ix_0) * (iy - iy_0);
            giz += val_111 * gradout * (ix - ix_0) * (iy - iy_0);

          }


          // un-normalize gradGrid back to [-1,1]
          gix = gix * (inputTensor_sz1 - 1) / 2;
          giy = giy * (inputTensor_sz2 - 1) / 2;
          giz = giz * (inputTensor_sz3 - 1) / 2;

          real gix_old = THTensor_fastGet5d(gradGrid, n, ind1, ind2, ind3, 0);
          real giy_old = THTensor_fastGet5d(gradGrid, n, ind1, ind2, ind3, 1);
          real giz_old = THTensor_fastGet5d(gradGrid, n, ind1, ind2, ind3, 2);

          THTensor_fastSet5d(gradGrid, n, ind1, ind2, ind3, 0, gix_old + gix);
          THTensor_fastSet5d(gradGrid, n, ind1, ind2, ind3, 1, giy_old + giy);
          THTensor_fastSet5d(gradGrid, n, ind1, ind2, ind3, 2, giz_old + giz);
        }
      }
    }
  }


}

#undef MIN
#undef MAX
#undef THTensor_fastGet5d
#undef THTensor_fastSet5d
#undef SAFE_GET
#undef CLIP_COORDINATES
#undef SAFE_ADD
#undef THNN_ARGCHECK
#undef THNN_CHECK_DIM_SIZE

#endif
