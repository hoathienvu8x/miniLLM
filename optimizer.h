#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "model.h"
#include "backward.h"

typedef struct {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;

  Gradients* m;
  Gradients* v;

  int t;
} AdamOptimizer;

typedef struct {
  float lr;
  float momentum;
  float weight_decay;

  Gradients* velocity;
} SGDOptimizer;

AdamOptimizer* adam_create(GPTModel* model, float lr);

AdamOptimizer* adam_create_full(
  GPTModel* model,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float weight_decay
);

void adam_free(AdamOptimizer* opt);

void adam_step(AdamOptimizer* opt, GPTModel* model, Gradients* grads);

void adam_reset(AdamOptimizer* opt);

SGDOptimizer* sgd_create(GPTModel* model, float lr, float momentum);

void sgd_free(SGDOptimizer* opt);

void sgd_step(SGDOptimizer* opt, GPTModel* model, Gradients* grads);

float cosine_lr(float lr_max, float lr_min, int step, int total_steps);

float warmup_lr(float lr_max, int step, int warmup_steps);

float warmup_cosine_lr(
  float lr_max,
  float lr_min,
  int step,
  int warmup_steps,
  int total_steps
);

#endif
