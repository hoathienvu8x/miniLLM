#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

float cross_entropy_loss(
  Tensor* logits,
  int* targets,
  int seq_len,
  int vocab_size
);

float cross_entropy_loss_with_grad(
  Tensor* logits,
  int* targets,
  int seq_len,
  int vocab_size,
  Tensor* grad_logits
);

void cross_entropy_grad(
  Tensor* logits,
  int* targets,
  int seq_len,
  int vocab_size,
  Tensor* grad_logits
);

void softmax_cross_entropy_grad(
  float* logits,
  int target,
  int vocab_size,
  float* grad
);

#endif
