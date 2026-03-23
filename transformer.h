#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "tensor.h"
#include "layernorm.h"
#include "attention.h"
#include "ffn.h"
#include "kv_cache.h"

typedef struct {
  LayerNorm* ln1;
  MultiHeadAttention* attn;
  LayerNorm* ln2;
  FFN* ffn;

  int hidden_dim;
  int num_heads;
  int ffn_dim;
} TransformerBlock;

typedef struct {
  Tensor* ln1_out;
  Tensor* attn_out;
  Tensor* ln2_out;
  Tensor* ffn_out;
  AttentionCache* attn_cache;
  FFNCache* ffn_cache;
  int seq_len;
  int hidden_dim;
} TransformerCache;

TransformerBlock* transformer_block_create(int hidden_dim, int num_heads, int ffn_dim);

void transformer_block_free(TransformerBlock* block);

TransformerCache* transformer_cache_create(int seq_len, int hidden_dim, int num_heads, int ffn_dim);

void transformer_cache_free(TransformerCache* cache);

int transformer_cache_resize(TransformerCache* cache, int new_seq_len);

void transformer_block_init(TransformerBlock* block, float std);

void transformer_block_forward(
  TransformerBlock* block,
  Tensor* input,
  Tensor* mask,
  TransformerCache* cache,
  Tensor* output
);

void transformer_block_print_info(TransformerBlock* block);

int transformer_block_num_params(TransformerBlock* block);

void transformer_block_forward_prefill(
  TransformerBlock* block,
  Tensor* input,
  KVCache* kv_cache,
  int layer_idx,
  Tensor* mask,
  TransformerCache* cache,
  Tensor* output
);

void transformer_block_forward_decode(
  TransformerBlock* block,
  Tensor* input,
  KVCache* kv_cache,
  int layer_idx,
  int pos,
  TransformerCache* cache,
  Tensor* output
);

#endif
