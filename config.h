#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

typedef struct {
  int vocab_size;
  int hidden_dim;
  int num_heads;
  int num_layers;
  int ffn_dim;
  int max_seq_len;
} ModelConfig;

static inline ModelConfig default_config(void) {
  ModelConfig config;
  config.vocab_size = 260;
  config.hidden_dim = 64;
  config.num_heads = 4;
  config.num_layers = 4;
  config.ffn_dim = 256;
  config.max_seq_len = 128;
  return config;
}

static inline ModelConfig tiny_config(void) {
  ModelConfig config;
  config.vocab_size = 260;
  config.hidden_dim = 32;
  config.num_heads = 2;
  config.num_layers = 2;
  config.ffn_dim = 128;
  config.max_seq_len = 64;
  return config;
}

static inline ModelConfig medium_config(void) {
  ModelConfig config;
  config.vocab_size = 260;
  config.hidden_dim = 128;
  config.num_heads = 4;
  config.num_layers = 6;
  config.ffn_dim = 512;
  config.max_seq_len = 256;
  return config;
}

static inline ModelConfig bpe_config(int vocab_size) {
  ModelConfig config;
  config.vocab_size = vocab_size;
  config.hidden_dim = 128;
  config.num_heads = 4;
  config.num_layers = 4;
  config.ffn_dim = 512;
  config.max_seq_len = 256;
  return config;
}

static inline ModelConfig bpe_small_config(int vocab_size) {
  ModelConfig config;
  config.vocab_size = vocab_size;
  config.hidden_dim = 64;
  config.num_heads = 4;
  config.num_layers = 4;
  config.ffn_dim = 256;
  config.max_seq_len = 128;
  return config;
}

static inline void config_print(ModelConfig* config) {
  printf("ModelConfig:\n");
  printf("  vocab_size: %d\n", config->vocab_size);
  printf("  hidden_dim: %d\n", config->hidden_dim);
  printf("  num_heads: %d\n", config->num_heads);
  printf("  num_layers: %d\n", config->num_layers);
  printf("  ffn_dim: %d\n", config->ffn_dim);
  printf("  max_seq_len: %d\n", config->max_seq_len);
}

static inline int config_validate(ModelConfig* config) {
  if (config->vocab_size <= 0) return 0;
  if (config->hidden_dim <= 0) return 0;
  if (config->num_heads <= 0) return 0;
  if (config->num_layers <= 0) return 0;
  if (config->ffn_dim <= 0) return 0;
  if (config->max_seq_len <= 0) return 0;
  if (config->hidden_dim % config->num_heads != 0) return 0;
  return 1;
}

static inline int config_save(ModelConfig* config, const char* path) {
  FILE* f = fopen(path, "wb");
  if (f == NULL) return -1;
  fwrite(config, sizeof(ModelConfig), 1, f);
  fclose(f);
  return 0;
}

static inline int config_load(ModelConfig* config, const char* path) {
  FILE* f = fopen(path, "rb");
  if (f == NULL) return -1;
  size_t read = fread(config, sizeof(ModelConfig), 1, f);
  fclose(f);
  return (read == 1) ? 0 : -1;
}

#endif
