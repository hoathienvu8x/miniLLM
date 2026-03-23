#include "embedding.h"
#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Embedding* embedding_create(int vocab_size, int hidden_dim, int max_seq_len) {
    if (vocab_size <= 0 || hidden_dim <= 0 || max_seq_len <= 0) {
        return NULL;
    }

    Embedding* emb = (Embedding*)malloc(sizeof(Embedding));
    if (emb == NULL) return NULL;

    emb->vocab_size = vocab_size;
    emb->hidden_dim = hidden_dim;
    emb->max_seq_len = max_seq_len;

    // 创建 token embedding [vocab_size, hidden_dim]
    int token_shape[] = {vocab_size, hidden_dim};
    emb->token_embedding = tensor_zeros(2, token_shape);
    if (emb->token_embedding == NULL) {
        free(emb);
        return NULL;
    }

    // 创建 position embedding [max_seq_len, hidden_dim]
    int pos_shape[] = {max_seq_len, hidden_dim};
    emb->position_embedding = tensor_zeros(2, pos_shape);
    if (emb->position_embedding == NULL) {
        tensor_free(emb->token_embedding);
        free(emb);
        return NULL;
    }

    return emb;
}

void embedding_free(Embedding* emb) {
    if (emb == NULL) return;

    if (emb->token_embedding != NULL) {
        tensor_free(emb->token_embedding);
    }
    if (emb->position_embedding != NULL) {
        tensor_free(emb->position_embedding);
    }
    free(emb);
}

void embedding_init_random(Embedding* emb, float std) {
    if (emb == NULL || emb->token_embedding == NULL) return;

    // 使用正态分布初始化 token embedding
    int size = emb->token_embedding->size;
    for (int i = 0; i < size; i++) {
        // Box-Muller 变换
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        emb->token_embedding->data[i] = z * std;
    }
}

void embedding_init_sinusoidal_position(Embedding* emb) {
    if (emb == NULL || emb->position_embedding == NULL) return;

    int max_len = emb->max_seq_len;
    int dim = emb->hidden_dim;

    // PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < dim; i++) {
            float div_term = powf(10000.0f, (float)(i / 2 * 2) / (float)dim);
            float angle = (float)pos / div_term;

            int idx = pos * dim + i;
            if (i % 2 == 0) {
                emb->position_embedding->data[idx] = sinf(angle);
            } else {
                emb->position_embedding->data[idx] = cosf(angle);
            }
        }
    }
}

void embedding_init_learned_position(Embedding* emb, float std) {
    if (emb == NULL || emb->position_embedding == NULL) return;

    int size = emb->position_embedding->size;
    for (int i = 0; i < size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        emb->position_embedding->data[i] = z * std;
    }
}

void embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output) {
    if (emb == NULL || token_ids == NULL || output == NULL) return;
    if (seq_len > emb->max_seq_len) {
        fprintf(stderr, "Error: seq_len (%d) > max_seq_len (%d)\n", seq_len, emb->max_seq_len);
        return;
    }

    int hidden_dim = emb->hidden_dim;

    // output = token_embedding[token_ids] + position_embedding[0:seq_len]
    for (int pos = 0; pos < seq_len; pos++) {
        int token_id = token_ids[pos];

        // 边界检查
        if (token_id < 0 || token_id >= emb->vocab_size) {
            token_id = 1;  // 使用 UNK token
        }

        int token_offset = token_id * hidden_dim;
        int pos_offset = pos * hidden_dim;
        int out_offset = pos * hidden_dim;

        for (int d = 0; d < hidden_dim; d++) {
            output->data[out_offset + d] =
                emb->token_embedding->data[token_offset + d] +
                emb->position_embedding->data[pos_offset + d];
        }
    }
}

void embedding_get_token(Embedding* emb, int* token_ids, int seq_len, Tensor* output) {
    if (emb == NULL || token_ids == NULL || output == NULL) return;

    int hidden_dim = emb->hidden_dim;

    for (int pos = 0; pos < seq_len; pos++) {
        int token_id = token_ids[pos];

        if (token_id < 0 || token_id >= emb->vocab_size) {
            token_id = 1;  // UNK
        }

        int token_offset = token_id * hidden_dim;
        int out_offset = pos * hidden_dim;

        for (int d = 0; d < hidden_dim; d++) {
            output->data[out_offset + d] = emb->token_embedding->data[token_offset + d];
        }
    }
}

void embedding_get_position(Embedding* emb, int seq_len, Tensor* output) {
    if (emb == NULL || output == NULL) return;
    if (seq_len > emb->max_seq_len) {
        seq_len = emb->max_seq_len;
    }

    int hidden_dim = emb->hidden_dim;

    for (int pos = 0; pos < seq_len; pos++) {
        int pos_offset = pos * hidden_dim;
        int out_offset = pos * hidden_dim;

        for (int d = 0; d < hidden_dim; d++) {
            output->data[out_offset + d] = emb->position_embedding->data[pos_offset + d];
        }
    }
}

void embedding_print_info(Embedding* emb) {
    if (emb == NULL) {
        printf("Embedding: NULL\n");
        return;
    }

    printf("Embedding Info:\n");
    printf("  vocab_size: %d\n", emb->vocab_size);
    printf("  hidden_dim: %d\n", emb->hidden_dim);
    printf("  max_seq_len: %d\n", emb->max_seq_len);
    printf("  token_embedding: [%d, %d]\n", emb->vocab_size, emb->hidden_dim);
    printf("  position_embedding: [%d, %d]\n", emb->max_seq_len, emb->hidden_dim);
    printf("  total params: %d\n", embedding_num_params(emb));
}

float* embedding_get_vector(Embedding* emb, int token_id) {
    if (emb == NULL || emb->token_embedding == NULL) return NULL;
    if (token_id < 0 || token_id >= emb->vocab_size) return NULL;

    return &emb->token_embedding->data[token_id * emb->hidden_dim];
}

int embedding_num_params(Embedding* emb) {
    if (emb == NULL) return 0;

    int token_params = emb->vocab_size * emb->hidden_dim;
    int pos_params = emb->max_seq_len * emb->hidden_dim;

    return token_params + pos_params;
}
