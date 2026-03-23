#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "kv_cache.h"

typedef struct {
  Tensor* W_q;
  Tensor* W_k;
  Tensor* W_v;
  Tensor* W_o;

  int hidden_dim;
  int num_heads;
  int head_dim;
  float scale;
} MultiHeadAttention;

typedef struct {
  Tensor* Q;
  Tensor* K;
  Tensor* V;
  Tensor* scores;
  Tensor* attn;
  Tensor* attn_out;
  int seq_len;
  int hidden_dim;
  int num_heads;
} AttentionCache;

MultiHeadAttention* attention_create(int hidden_dim, int num_heads);

void attention_free(MultiHeadAttention* attn);

AttentionCache* attention_cache_create(int seq_len, int hidden_dim, int num_heads);

void attention_cache_free(AttentionCache* cache);

int attention_cache_resize(AttentionCache* cache, int new_seq_len);

void attention_init_random(MultiHeadAttention* attn, float std);

Tensor* create_causal_mask(int seq_len);

void attention_forward(
  MultiHeadAttention* attn,
  Tensor* input,
  Tensor* mask,
  AttentionCache* cache,
  Tensor* output
);

void single_head_attention(
  Tensor* Q, Tensor* K, Tensor* V,
  Tensor* mask, float scale,
  Tensor* output
);

void attention_print_info(MultiHeadAttention* attn);

int attention_num_params(MultiHeadAttention* attn);

Tensor* attention_get_weights(AttentionCache* cache);

void attention_forward_kv_cache(
  MultiHeadAttention* attn,
  Tensor* input,
  KVCache* kv_cache,
  int layer_idx,
  int start_pos,
  AttentionCache* cache,
  Tensor* output
);

void attention_prefill_kv_cache(
  MultiHeadAttention* attn,
  Tensor* input,
  KVCache* kv_cache,
  int layer_idx,
  Tensor* mask,
  AttentionCache* cache,
  Tensor* output
);

#endif
