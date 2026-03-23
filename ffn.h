#ifndef FFN_H
#define FFN_H

#include "tensor.h"

typedef struct {
  Tensor* W1;
  Tensor* b1;
  Tensor* W2;
  Tensor* b2;
  int hidden_dim;
  int ffn_dim;
} FFN;

typedef struct {
  Tensor* hidden;
  int seq_len;
  int ffn_dim;
} FFNCache;

FFN* ffn_create(int hidden_dim, int ffn_dim);

void ffn_free(FFN* ffn);

FFNCache* ffn_cache_create(int seq_len, int ffn_dim);

void ffn_cache_free(FFNCache* cache);

int ffn_cache_resize(FFNCache* cache, int new_seq_len);

void ffn_init_random(FFN* ffn, float std);

void ffn_forward(FFN* ffn, Tensor* input, FFNCache* cache, Tensor* output);

void ffn_print_info(FFN* ffn);

int ffn_num_params(FFN* ffn);

#endif
