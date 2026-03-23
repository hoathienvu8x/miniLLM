#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "config.h"
#include "embedding.h"
#include "transformer.h"
#include "layernorm.h"
#include "kv_cache.h"

typedef struct {
  ModelConfig config;
  Embedding* embedding;
  TransformerBlock** layers;
  LayerNorm* final_ln;
  Tensor* lm_head;
} GPTModel;

typedef struct {
  Tensor* hidden;
  Tensor* logits;
  TransformerCache** layer_caches;
  Tensor* mask;
  int seq_len;
  int num_layers;
} GPTCache;

GPTModel* model_create(ModelConfig config);

void model_free(GPTModel* model);

GPTCache* model_cache_create(GPTModel* model, int seq_len);

void model_cache_free(GPTCache* cache);

int model_cache_resize(GPTCache* cache, GPTModel* model, int new_seq_len);

void model_init_random(GPTModel* model, float std);

void model_forward(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  GPTCache* cache,
  Tensor* logits
);

void model_get_last_logits(Tensor* logits, int seq_len, Tensor* last_logits);

int model_save(GPTModel* model, const char* path);

GPTModel* model_load(const char* path);

int model_num_params(GPTModel* model);

void model_print_info(GPTModel* model);

size_t model_memory_size(GPTModel* model);

KVCache* model_create_kv_cache(GPTModel* model);

void model_forward_prefill(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  KVCache* kv_cache,
  GPTCache* gpt_cache,
  Tensor* logits
);

void model_forward_decode(
  GPTModel* model,
  int token_id,
  int pos,
  KVCache* kv_cache,
  GPTCache* gpt_cache,
  Tensor* logits
);

#endif
