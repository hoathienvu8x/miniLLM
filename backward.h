#ifndef BACKWARD_H
#define BACKWARD_H

#include "tensor.h"
#include "model.h"

typedef struct {

  Tensor* d_token_embedding;
  Tensor* d_position_embedding;

  struct {

    Tensor* d_ln1_gamma;
    Tensor* d_ln1_beta;

    Tensor* d_W_q;
    Tensor* d_W_k;
    Tensor* d_W_v;
    Tensor* d_W_o;

    Tensor* d_ln2_gamma;
    Tensor* d_ln2_beta;

    Tensor* d_W1;
    Tensor* d_b1;
    Tensor* d_W2;
    Tensor* d_b2;
  }* layers;

  Tensor* d_final_ln_gamma;
  Tensor* d_final_ln_beta;

  Tensor* d_lm_head;

  int num_layers;
} Gradients;

typedef struct {

  Tensor* embed_output;
  Tensor** layer_inputs;
  Tensor** ln1_outputs;
  Tensor** attn_outputs;
  Tensor** ln2_outputs;
  Tensor** ffn_hidden;
  Tensor* final_ln_input;
  Tensor* final_ln_output;

  int num_layers;
  int seq_len;
  int hidden_dim;
} BackwardCache;

Gradients* gradients_create(GPTModel* model);

void gradients_free(Gradients* grads);

void gradients_zero(Gradients* grads);

float gradients_clip(Gradients* grads, float max_norm);

float gradients_norm(Gradients* grads);

BackwardCache* backward_cache_create(GPTModel* model, int seq_len);

void backward_cache_free(BackwardCache* cache);

void model_forward_with_cache(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  BackwardCache* cache,
  Tensor* logits
);

void model_backward(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  Tensor* grad_logits,
  BackwardCache* cache,
  Gradients* grads
);

void linear_backward(
  Tensor* grad_output,
  Tensor* input,
  Tensor* weight,
  Tensor* grad_weight,
  Tensor* grad_input
);

void layernorm_backward(
  Tensor* grad_output,
  Tensor* input,
  Tensor* gamma,
  Tensor* grad_gamma,
  Tensor* grad_beta,
  Tensor* grad_input,
  float eps
);

void gelu_backward(
  Tensor* grad_output,
  Tensor* input,
  Tensor* grad_input
);

void softmax_backward(
  Tensor* grad_output,
  Tensor* output,
  Tensor* grad_input
);

void embedding_backward(
  Tensor* grad_output,
  int* input_ids,
  int seq_len,
  Tensor* grad_token_emb,
  Tensor* grad_pos_emb
);

#endif
