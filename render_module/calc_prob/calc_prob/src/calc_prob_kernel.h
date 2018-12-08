#ifdef __cplusplus
extern "C" {
#endif
  int calc_prob_forward_wrap(THCState* state, THCudaTensor* prob_in, THCudaTensor* prob_out);
  int calc_prob_backward_wrap(THCState* state, THCudaTensor* prob_in, THCudaTensor* stop_prob_weighted, THCudaTensor*  grad_out);
#ifdef __cplusplus
}
#endif
