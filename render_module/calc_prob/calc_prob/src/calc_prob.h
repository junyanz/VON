int calc_prob_forward(THCudaTensor* prob_in, THCudaTensor* prob_out);
int calc_prob_backward(THCudaTensor* prob_in, THCudaTensor* stop_prob_weighted, THCudaTensor*  grad_out);
