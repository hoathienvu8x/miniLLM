#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

typedef struct {
  Tensor* gamma;
  Tensor* beta;
  int hidden_dim;
  float eps;
} LayerNorm;

LayerNorm* layernorm_create(int hidden_dim, float eps);

void layernorm_free(LayerNorm* ln);

void layernorm_init(LayerNorm* ln);

void layernorm_forward(LayerNorm* ln, Tensor* input, Tensor* output);

void layernorm_print_info(LayerNorm* ln);

int layernorm_num_params(LayerNorm* ln);

#endif
