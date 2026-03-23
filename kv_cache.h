#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "tensor.h"

typedef struct {
  Tensor* k_cache;
  Tensor* v_cache;
} LayerKVCache;

typedef struct {
  LayerKVCache* layers;
  int num_layers;
  int max_seq_len;
  int hidden_dim;
  int current_len;
} KVCache;

KVCache* kv_cache_create(int num_layers, int max_seq_len, int hidden_dim);

void kv_cache_free(KVCache* cache);

int kv_cache_update(
  KVCache* cache, int layer_idx,
  Tensor* new_k, Tensor* new_v, int num_new_tokens
);

void kv_cache_update_pos(
  KVCache* cache, int layer_idx, int pos,
  float* k_data, float* v_data
);

Tensor* kv_cache_get_k(KVCache* cache, int layer_idx);

Tensor* kv_cache_get_v(KVCache* cache, int layer_idx);

void kv_cache_clear(KVCache* cache);

void kv_cache_set_len(KVCache* cache, int len);

void kv_cache_print_info(KVCache* cache);

size_t kv_cache_memory_size(KVCache* cache);

#endif
