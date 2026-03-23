#include "model.h"
#include "attention.h"
#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MODEL_MAGIC 0x4D4C4C4D
#define MODEL_VERSION 1

GPTModel* model_create(ModelConfig config) {
  if (!config_validate(&config)) {
    fprintf(stderr, "Error: invalid model config\n");
    return NULL;
  }

  GPTModel* model = (GPTModel*)malloc(sizeof(GPTModel));
  if (model == NULL) return NULL;

  model->config = config;

  model->embedding = embedding_create(config.vocab_size, config.hidden_dim, config.max_seq_len);
  if (model->embedding == NULL) {
    free(model);
    return NULL;
  }

  model->layers = (TransformerBlock**)malloc(config.num_layers * sizeof(TransformerBlock*));
  if (model->layers == NULL) {
    embedding_free(model->embedding);
    free(model);
    return NULL;
  }

  for (int i = 0; i < config.num_layers; i++) {
    model->layers[i] = transformer_block_create(config.hidden_dim, config.num_heads, config.ffn_dim);
    if (model->layers[i] == NULL) {

      for (int j = 0; j < i; j++) {
        transformer_block_free(model->layers[j]);
      }
      free(model->layers);
      embedding_free(model->embedding);
      free(model);
      return NULL;
    }
  }

  model->final_ln = layernorm_create(config.hidden_dim, 1e-5f);
  if (model->final_ln == NULL) {
    for (int i = 0; i < config.num_layers; i++) {
      transformer_block_free(model->layers[i]);
    }
    free(model->layers);
    embedding_free(model->embedding);
    free(model);
    return NULL;
  }

  int lm_shape[] = {config.hidden_dim, config.vocab_size};
  model->lm_head = tensor_zeros(2, lm_shape);
  if (model->lm_head == NULL) {
    layernorm_free(model->final_ln);
    for (int i = 0; i < config.num_layers; i++) {
      transformer_block_free(model->layers[i]);
    }
    free(model->layers);
    embedding_free(model->embedding);
    free(model);
    return NULL;
  }

  return model;
}

void model_free(GPTModel* model) {
  if (model == NULL) return;

  if (model->embedding) embedding_free(model->embedding);

  if (model->layers) {
    for (int i = 0; i < model->config.num_layers; i++) {
      if (model->layers[i]) transformer_block_free(model->layers[i]);
    }
    free(model->layers);
  }

  if (model->final_ln) layernorm_free(model->final_ln);
  if (model->lm_head) tensor_free(model->lm_head);

  free(model);
}

GPTCache* model_cache_create(GPTModel* model, int seq_len) {
  if (model == NULL) return NULL;

  GPTCache* cache = (GPTCache*)malloc(sizeof(GPTCache));
  if (cache == NULL) return NULL;

  cache->seq_len = seq_len;
  cache->num_layers = model->config.num_layers;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;

  int hidden_shape[] = {seq_len, hidden_dim};
  cache->hidden = tensor_zeros(2, hidden_shape);

  int logits_shape[] = {seq_len, vocab_size};
  cache->logits = tensor_zeros(2, logits_shape);

  cache->mask = create_causal_mask(seq_len);

  cache->layer_caches = (TransformerCache**)malloc(cache->num_layers * sizeof(TransformerCache*));
  if (cache->layer_caches == NULL) {
    tensor_free(cache->hidden);
    tensor_free(cache->logits);
    tensor_free(cache->mask);
    free(cache);
    return NULL;
  }

  for (int i = 0; i < cache->num_layers; i++) {
    cache->layer_caches[i] = transformer_cache_create(
      seq_len, hidden_dim, model->config.num_heads, model->config.ffn_dim
    );
    if (cache->layer_caches[i] == NULL) {
      for (int j = 0; j < i; j++) {
        transformer_cache_free(cache->layer_caches[j]);
      }
      free(cache->layer_caches);
      tensor_free(cache->hidden);
      tensor_free(cache->logits);
      tensor_free(cache->mask);
      free(cache);
      return NULL;
    }
  }

  return cache;
}

void model_cache_free(GPTCache* cache) {
  if (cache == NULL) return;

  if (cache->hidden) tensor_free(cache->hidden);
  if (cache->logits) tensor_free(cache->logits);
  if (cache->mask) tensor_free(cache->mask);

  if (cache->layer_caches) {
    for (int i = 0; i < cache->num_layers; i++) {
      if (cache->layer_caches[i]) {
        transformer_cache_free(cache->layer_caches[i]);
      }
    }
    free(cache->layer_caches);
  }

  free(cache);
}

int model_cache_resize(GPTCache* cache, GPTModel* model, int new_seq_len) {
  if (cache == NULL || model == NULL) return -1;
  if (new_seq_len == cache->seq_len) return 0;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;

  if (cache->hidden) tensor_free(cache->hidden);
  if (cache->logits) tensor_free(cache->logits);
  if (cache->mask) tensor_free(cache->mask);

  int hidden_shape[] = {new_seq_len, hidden_dim};
  int logits_shape[] = {new_seq_len, vocab_size};

  cache->hidden = tensor_zeros(2, hidden_shape);
  cache->logits = tensor_zeros(2, logits_shape);
  cache->mask = create_causal_mask(new_seq_len);

  for (int i = 0; i < cache->num_layers; i++) {
    transformer_cache_resize(cache->layer_caches[i], new_seq_len);
  }

  cache->seq_len = new_seq_len;

  return (cache->hidden && cache->logits && cache->mask) ? 0 : -1;
}

void model_init_random(GPTModel* model, float std) {
  if (model == NULL) return;

  embedding_init_random(model->embedding, std);
  embedding_init_sinusoidal_position(model->embedding);

  for (int i = 0; i < model->config.num_layers; i++) {
    transformer_block_init(model->layers[i], std);
  }

  layernorm_init(model->final_ln);

  for (int i = 0; i < model->lm_head->size; i++) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 < 1e-10f) u1 = 1e-10f;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    model->lm_head->data[i] = z * std;
  }
}

void model_forward(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  GPTCache* cache,
  Tensor* logits
) {
  if (model == NULL || input_ids == NULL || logits == NULL) return;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;

  if (cache != NULL && cache->seq_len != seq_len) {
    model_cache_resize(cache, model, seq_len);
  }

  Tensor* hidden;
  Tensor* mask;
  int need_free = 0;

  if (cache != NULL) {
    hidden = cache->hidden;
    mask = cache->mask;
  } else {
    int h_shape[] = {seq_len, hidden_dim};
    hidden = tensor_zeros(2, h_shape);
    mask = create_causal_mask(seq_len);
    need_free = 1;
  }

  embedding_forward(model->embedding, input_ids, seq_len, hidden);

  int h_shape[] = {seq_len, hidden_dim};
  Tensor* layer_output = tensor_zeros(2, h_shape);

  for (int i = 0; i < model->config.num_layers; i++) {
    TransformerCache* layer_cache = (cache != NULL) ? cache->layer_caches[i] : NULL;
    transformer_block_forward(model->layers[i], hidden, mask, layer_cache, layer_output);

    Tensor* temp = hidden;
    hidden = layer_output;
    layer_output = temp;
  }

  if (model->config.num_layers % 2 == 1) {

    if (cache != NULL) {
      memcpy(cache->hidden->data, hidden->data, seq_len * hidden_dim * sizeof(float));
      hidden = cache->hidden;
    }
  }

  tensor_free(layer_output);

  Tensor* ln_out = tensor_zeros(2, h_shape);
  layernorm_forward(model->final_ln, hidden, ln_out);

  for (int s = 0; s < seq_len; s++) {
    for (int v = 0; v < vocab_size; v++) {
      float sum = 0.0f;
      for (int h = 0; h < hidden_dim; h++) {
        sum += ln_out->data[s * hidden_dim + h] * model->lm_head->data[h * vocab_size + v];
      }
      logits->data[s * vocab_size + v] = sum;
    }
  }

  tensor_free(ln_out);

  if (need_free) {
    tensor_free(hidden);
    tensor_free(mask);
  }
}

void model_get_last_logits(Tensor* logits, int seq_len, Tensor* last_logits) {
  if (logits == NULL || last_logits == NULL) return;

  int vocab_size = logits->shape[1];
  int offset = (seq_len - 1) * vocab_size;

  memcpy(last_logits->data, &logits->data[offset], vocab_size * sizeof(float));
}

int model_save(GPTModel* model, const char* path) {
  if (model == NULL || path == NULL) return -1;

  FILE* f = fopen(path, "wb");
  if (f == NULL) {
    fprintf(stderr, "Error: cannot create file '%s'\n", path);
    return -1;
  }

  #define check_fwrite(ptr, size, nmemb, stream) \
    if (fwrite(ptr, size, nmemb, stream) < (size_t)(nmemb)) { \
      perror("fwrite()"); \
      fclose(f); \
      goto fail; \
    }

  int magic = MODEL_MAGIC;
  int version = MODEL_VERSION;
  check_fwrite(&magic, sizeof(int), 1, f);
  check_fwrite(&version, sizeof(int), 1, f);

  check_fwrite(&model->config, sizeof(ModelConfig), 1, f);

  check_fwrite(model->embedding->token_embedding->data,
       sizeof(float), model->embedding->token_embedding->size, f);
  check_fwrite(model->embedding->position_embedding->data,
       sizeof(float), model->embedding->position_embedding->size, f);

  for (int i = 0; i < model->config.num_layers; i++) {
    TransformerBlock* layer = model->layers[i];

    check_fwrite(layer->ln1->gamma->data, sizeof(float), layer->ln1->gamma->size, f);
    check_fwrite(layer->ln1->beta->data, sizeof(float), layer->ln1->beta->size, f);

    check_fwrite(layer->attn->W_q->data, sizeof(float), layer->attn->W_q->size, f);
    check_fwrite(layer->attn->W_k->data, sizeof(float), layer->attn->W_k->size, f);
    check_fwrite(layer->attn->W_v->data, sizeof(float), layer->attn->W_v->size, f);
    check_fwrite(layer->attn->W_o->data, sizeof(float), layer->attn->W_o->size, f);

    check_fwrite(layer->ln2->gamma->data, sizeof(float), layer->ln2->gamma->size, f);
    check_fwrite(layer->ln2->beta->data, sizeof(float), layer->ln2->beta->size, f);

    check_fwrite(layer->ffn->W1->data, sizeof(float), layer->ffn->W1->size, f);
    check_fwrite(layer->ffn->b1->data, sizeof(float), layer->ffn->b1->size, f);
    check_fwrite(layer->ffn->W2->data, sizeof(float), layer->ffn->W2->size, f);
    check_fwrite(layer->ffn->b2->data, sizeof(float), layer->ffn->b2->size, f);
  }

  check_fwrite(model->final_ln->gamma->data, sizeof(float), model->final_ln->gamma->size, f);
  check_fwrite(model->final_ln->beta->data, sizeof(float), model->final_ln->beta->size, f);

  check_fwrite(model->lm_head->data, sizeof(float), model->lm_head->size, f);

  fclose(f);
  #undef check_fwrite
  return 0;
fail:
  #undef check_fwrite
  return -1;
}

GPTModel* model_load(const char* path) {
  ModelConfig config;
  GPTModel* model = NULL;
  if (path == NULL) return NULL;

  FILE* f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "Error: cannot open file '%s'\n", path);
    return NULL;
  }

  #define check_fread(ptr, size, nmemb, stream) \
    if (fread(ptr, size, nmemb, stream) < (size_t)(nmemb)) { \
      perror("fread()"); \
      fclose(f); \
      goto fail; \
    }

  int magic, version;
  check_fread(&magic, sizeof(int), 1, f);
  check_fread(&version, sizeof(int), 1, f);

  if (magic != MODEL_MAGIC) {
    fprintf(stderr, "Error: invalid model file (bad magic number)\n");
    fclose(f);
    return NULL;
  }

  if (version != MODEL_VERSION) {
    fprintf(stderr, "Error: unsupported model version %d\n", version);
    fclose(f);
    return NULL;
  }

  check_fread(&config, sizeof(ModelConfig), 1, f);

  model = model_create(config);
  if (model == NULL) {
    fclose(f);
    return NULL;
  }

  check_fread(model->embedding->token_embedding->data,
      sizeof(float), model->embedding->token_embedding->size, f);
  check_fread(model->embedding->position_embedding->data,
      sizeof(float), model->embedding->position_embedding->size, f);

  for (int i = 0; i < config.num_layers; i++) {
    TransformerBlock* layer = model->layers[i];

    check_fread(layer->ln1->gamma->data, sizeof(float), layer->ln1->gamma->size, f);
    check_fread(layer->ln1->beta->data, sizeof(float), layer->ln1->beta->size, f);

    check_fread(layer->attn->W_q->data, sizeof(float), layer->attn->W_q->size, f);
    check_fread(layer->attn->W_k->data, sizeof(float), layer->attn->W_k->size, f);
    check_fread(layer->attn->W_v->data, sizeof(float), layer->attn->W_v->size, f);
    check_fread(layer->attn->W_o->data, sizeof(float), layer->attn->W_o->size, f);

    check_fread(layer->ln2->gamma->data, sizeof(float), layer->ln2->gamma->size, f);
    check_fread(layer->ln2->beta->data, sizeof(float), layer->ln2->beta->size, f);

    check_fread(layer->ffn->W1->data, sizeof(float), layer->ffn->W1->size, f);
    check_fread(layer->ffn->b1->data, sizeof(float), layer->ffn->b1->size, f);
    check_fread(layer->ffn->W2->data, sizeof(float), layer->ffn->W2->size, f);
    check_fread(layer->ffn->b2->data, sizeof(float), layer->ffn->b2->size, f);
  }

  check_fread(model->final_ln->gamma->data, sizeof(float), model->final_ln->gamma->size, f);
  check_fread(model->final_ln->beta->data, sizeof(float), model->final_ln->beta->size, f);

  check_fread(model->lm_head->data, sizeof(float), model->lm_head->size, f);

  fclose(f);
  #undef check_fread
  return model;
fail:
  #undef check_fread
  if (model) model_free(model);
  return NULL;
}

int model_num_params(GPTModel* model) {
  if (model == NULL) return 0;

  int total = 0;

  total += embedding_num_params(model->embedding);

  for (int i = 0; i < model->config.num_layers; i++) {
    total += transformer_block_num_params(model->layers[i]);
  }

  total += layernorm_num_params(model->final_ln);

  total += model->lm_head->size;

  return total;
}

void model_print_info(GPTModel* model) {
  if (model == NULL) {
    printf("GPTModel: NULL\n");
    return;
  }

  printf("========================================\n");
  printf("           GPT Model Info\n");
  printf("========================================\n");

  config_print(&model->config);

  printf("\nParameter Breakdown:\n");
  printf("  Embedding:\n");
  printf("    token_embedding: %d × %d = %d\n",
       model->config.vocab_size, model->config.hidden_dim,
       model->config.vocab_size * model->config.hidden_dim);
  printf("    position_embedding: %d × %d = %d\n",
       model->config.max_seq_len, model->config.hidden_dim,
       model->config.max_seq_len * model->config.hidden_dim);

  printf("  Transformer Layers (×%d):\n", model->config.num_layers);
  int layer_params = transformer_block_num_params(model->layers[0]);
  printf("    per layer: %d params\n", layer_params);
  printf("    total: %d params\n", layer_params * model->config.num_layers);

  printf("  Final LayerNorm: %d params\n", layernorm_num_params(model->final_ln));
  printf("  LM Head: %d × %d = %d params\n",
       model->config.hidden_dim, model->config.vocab_size,
       model->lm_head->size);

  printf("\nTotal Parameters: %d\n", model_num_params(model));
  printf("Memory Size: %.2f KB\n", model_memory_size(model) / 1024.0f);
  printf("========================================\n");
}

size_t model_memory_size(GPTModel* model) {
  if (model == NULL) return 0;
  return (size_t)model_num_params(model) * sizeof(float);
}

KVCache* model_create_kv_cache(GPTModel* model) {
  if (model == NULL) return NULL;

  return kv_cache_create(
    model->config.num_layers,
    model->config.max_seq_len,
    model->config.hidden_dim
  );
}

void model_forward_prefill(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  KVCache* kv_cache,
  GPTCache* gpt_cache,
  Tensor* logits
) {
  if (model == NULL || input_ids == NULL || kv_cache == NULL || logits == NULL) return;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;

  if (gpt_cache != NULL && gpt_cache->seq_len != seq_len) {
    model_cache_resize(gpt_cache, model, seq_len);
  }

  kv_cache_clear(kv_cache);

  Tensor* hidden;
  Tensor* mask;
  int need_free = 0;

  if (gpt_cache != NULL) {
    hidden = gpt_cache->hidden;
    mask = gpt_cache->mask;
  } else {
    int h_shape[] = {seq_len, hidden_dim};
    hidden = tensor_zeros(2, h_shape);
    mask = create_causal_mask(seq_len);
    need_free = 1;
  }

  embedding_forward(model->embedding, input_ids, seq_len, hidden);

  int h_shape[] = {seq_len, hidden_dim};
  Tensor* layer_output = tensor_zeros(2, h_shape);

  for (int i = 0; i < model->config.num_layers; i++) {
    TransformerCache* layer_cache = (gpt_cache != NULL) ? gpt_cache->layer_caches[i] : NULL;
    transformer_block_forward_prefill(model->layers[i], hidden, kv_cache, i,
                      mask, layer_cache, layer_output);

    Tensor* temp = hidden;
    hidden = layer_output;
    layer_output = temp;
  }

  if (model->config.num_layers % 2 == 1) {
    if (gpt_cache != NULL) {
      memcpy(gpt_cache->hidden->data, hidden->data, seq_len * hidden_dim * sizeof(float));
      hidden = gpt_cache->hidden;
    }
  }

  tensor_free(layer_output);

  kv_cache_set_len(kv_cache, seq_len);

  Tensor* ln_out = tensor_zeros(2, h_shape);
  layernorm_forward(model->final_ln, hidden, ln_out);

  for (int s = 0; s < seq_len; s++) {
    for (int v = 0; v < vocab_size; v++) {
      float sum = 0.0f;
      for (int h = 0; h < hidden_dim; h++) {
        sum += ln_out->data[s * hidden_dim + h] * model->lm_head->data[h * vocab_size + v];
      }
      logits->data[s * vocab_size + v] = sum;
    }
  }

  tensor_free(ln_out);

  if (need_free) {
    tensor_free(hidden);
    tensor_free(mask);
  }
}

void model_forward_decode(
  GPTModel* model,
  int token_id,
  int pos,
  KVCache* kv_cache,
  GPTCache* gpt_cache,
  Tensor* logits
) {
  if (model == NULL || kv_cache == NULL || logits == NULL) return;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;

  int hidden_shape[] = {1, hidden_dim};
  Tensor* hidden = tensor_zeros(2, hidden_shape);
  Tensor* layer_output = tensor_zeros(2, hidden_shape);

  for (int d = 0; d < hidden_dim; d++) {
    hidden->data[d] = model->embedding->token_embedding->data[token_id * hidden_dim + d];
  }

  for (int d = 0; d < hidden_dim; d++) {
    hidden->data[d] += model->embedding->position_embedding->data[pos * hidden_dim + d];
  }

  for (int i = 0; i < model->config.num_layers; i++) {
    TransformerCache* layer_cache = (gpt_cache != NULL) ? gpt_cache->layer_caches[i] : NULL;

    TransformerCache* temp_cache = NULL;
    if (layer_cache == NULL) {
      temp_cache = transformer_cache_create(1, hidden_dim,
                          model->config.num_heads,
                          model->config.ffn_dim);
      layer_cache = temp_cache;
    }

    transformer_block_forward_decode(model->layers[i], hidden, kv_cache, i,
                     pos, layer_cache, layer_output);

    if (temp_cache != NULL) {
      transformer_cache_free(temp_cache);
    }

    Tensor* temp = hidden;
    hidden = layer_output;
    layer_output = temp;
  }

  Tensor* ln_out = tensor_zeros(2, hidden_shape);
  layernorm_forward(model->final_ln, hidden, ln_out);

  for (int v = 0; v < vocab_size; v++) {
    float sum = 0.0f;
    for (int h = 0; h < hidden_dim; h++) {
      sum += ln_out->data[h] * model->lm_head->data[h * vocab_size + v];
    }
    logits->data[v] = sum;
  }

  tensor_free(hidden);
  tensor_free(layer_output);
  tensor_free(ln_out);
}
