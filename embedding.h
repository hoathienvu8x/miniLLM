#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

typedef struct {
  Tensor* token_embedding;
  Tensor* position_embedding;
  int vocab_size;
  int hidden_dim;
  int max_seq_len;
} Embedding;

Embedding* embedding_create(int vocab_size, int hidden_dim, int max_seq_len);

void embedding_free(Embedding* emb);

void embedding_init_random(Embedding* emb, float std);

void embedding_init_sinusoidal_position(Embedding* emb);

void embedding_init_learned_position(Embedding* emb, float std);

void embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

void embedding_get_token(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

void embedding_get_position(Embedding* emb, int seq_len, Tensor* output);

void embedding_print_info(Embedding* emb);

float* embedding_get_vector(Embedding* emb, int token_id);

int embedding_num_params(Embedding* emb);

#endif
