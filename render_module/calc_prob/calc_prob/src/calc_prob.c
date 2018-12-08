#include <stdbool.h>
#include <stdio.h>
#include <THC/THC.h>
#include "calc_prob.h"
#include "calc_prob_kernel.h"

extern THCState *state;

int calc_prob_forward(THCudaTensor* prob_in, THCudaTensor* prob_out){
  int success = 0;
  success = calc_prob_forward_wrap(state, prob_in, prob_out);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}
int calc_prob_backward(THCudaTensor* prob_in, THCudaTensor* stop_prob_weighted, THCudaTensor*  grad_out){
  int success = 0;
  success = calc_prob_backward_wrap(state, prob_in, stop_prob_weighted, grad_out);
  // check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}
