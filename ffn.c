#include "ffn.h"
#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

FFN* ffn_create(int hidden_dim, int ffn_dim) {
    if (hidden_dim <= 0 || ffn_dim <= 0) return NULL;

    FFN* ffn = (FFN*)malloc(sizeof(FFN));
    if (ffn == NULL) return NULL;

    ffn->hidden_dim = hidden_dim;
    ffn->ffn_dim = ffn_dim;

    // W1: [hidden_dim, ffn_dim]
    int w1_shape[] = {hidden_dim, ffn_dim};
    ffn->W1 = tensor_zeros(2, w1_shape);

    // b1: [ffn_dim]
    int b1_shape[] = {ffn_dim};
    ffn->b1 = tensor_zeros(1, b1_shape);

    // W2: [ffn_dim, hidden_dim]
    int w2_shape[] = {ffn_dim, hidden_dim};
    ffn->W2 = tensor_zeros(2, w2_shape);

    // b2: [hidden_dim]
    int b2_shape[] = {hidden_dim};
    ffn->b2 = tensor_zeros(1, b2_shape);

    if (ffn->W1 == NULL || ffn->b1 == NULL ||
        ffn->W2 == NULL || ffn->b2 == NULL) {
        ffn_free(ffn);
        return NULL;
    }

    return ffn;
}

void ffn_free(FFN* ffn) {
    if (ffn == NULL) return;

    if (ffn->W1) tensor_free(ffn->W1);
    if (ffn->b1) tensor_free(ffn->b1);
    if (ffn->W2) tensor_free(ffn->W2);
    if (ffn->b2) tensor_free(ffn->b2);
    free(ffn);
}

FFNCache* ffn_cache_create(int seq_len, int ffn_dim) {
    FFNCache* cache = (FFNCache*)malloc(sizeof(FFNCache));
    if (cache == NULL) return NULL;

    cache->seq_len = seq_len;
    cache->ffn_dim = ffn_dim;

    int shape[] = {seq_len, ffn_dim};
    cache->hidden = tensor_zeros(2, shape);

    if (cache->hidden == NULL) {
        free(cache);
        return NULL;
    }

    return cache;
}

void ffn_cache_free(FFNCache* cache) {
    if (cache == NULL) return;

    if (cache->hidden) tensor_free(cache->hidden);
    free(cache);
}

int ffn_cache_resize(FFNCache* cache, int new_seq_len) {
    if (cache == NULL) return -1;
    if (new_seq_len == cache->seq_len) return 0;

    if (cache->hidden) tensor_free(cache->hidden);

    int shape[] = {new_seq_len, cache->ffn_dim};
    cache->hidden = tensor_zeros(2, shape);
    cache->seq_len = new_seq_len;

    return (cache->hidden == NULL) ? -1 : 0;
}

void ffn_init_random(FFN* ffn, float std) {
    if (ffn == NULL) return;

    // 初始化 W1
    for (int i = 0; i < ffn->W1->size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        ffn->W1->data[i] = z * std;
    }

    // b1 初始化为 0 (已经是)

    // 初始化 W2
    for (int i = 0; i < ffn->W2->size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        ffn->W2->data[i] = z * std;
    }

    // b2 初始化为 0 (已经是)
}

// GELU 激活函数
static float gelu_scalar(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void ffn_forward(FFN* ffn, Tensor* input, FFNCache* cache, Tensor* output) {
    if (ffn == NULL || input == NULL || output == NULL) return;

    int seq_len = input->shape[0];
    int hidden_dim = ffn->hidden_dim;
    int ffn_dim = ffn->ffn_dim;

    // 检查并调整缓存大小
    if (cache != NULL && cache->seq_len != seq_len) {
        ffn_cache_resize(cache, seq_len);
    }

    // 分配临时张量 (如果没有缓存)
    Tensor* hidden;
    int need_free_hidden = 0;
    if (cache != NULL) {
        hidden = cache->hidden;
    } else {
        int h_shape[] = {seq_len, ffn_dim};
        hidden = tensor_zeros(2, h_shape);
        need_free_hidden = 1;
    }

    // Step 1: hidden = input @ W1 + b1
    // [seq_len, hidden_dim] @ [hidden_dim, ffn_dim] = [seq_len, ffn_dim]
    for (int s = 0; s < seq_len; s++) {
        for (int f = 0; f < ffn_dim; f++) {
            float sum = ffn->b1->data[f];
            for (int h = 0; h < hidden_dim; h++) {
                sum += input->data[s * hidden_dim + h] * ffn->W1->data[h * ffn_dim + f];
            }
            hidden->data[s * ffn_dim + f] = sum;
        }
    }

    // Step 2: hidden = GELU(hidden)
    for (int i = 0; i < hidden->size; i++) {
        hidden->data[i] = gelu_scalar(hidden->data[i]);
    }

    // Step 3: output = hidden @ W2 + b2
    // [seq_len, ffn_dim] @ [ffn_dim, hidden_dim] = [seq_len, hidden_dim]
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < hidden_dim; h++) {
            float sum = ffn->b2->data[h];
            for (int f = 0; f < ffn_dim; f++) {
                sum += hidden->data[s * ffn_dim + f] * ffn->W2->data[f * hidden_dim + h];
            }
            output->data[s * hidden_dim + h] = sum;
        }
    }

    if (need_free_hidden) {
        tensor_free(hidden);
    }
}

void ffn_print_info(FFN* ffn) {
    if (ffn == NULL) {
        printf("FFN: NULL\n");
        return;
    }

    printf("FFN Info:\n");
    printf("  hidden_dim: %d\n", ffn->hidden_dim);
    printf("  ffn_dim: %d\n", ffn->ffn_dim);
    printf("  W1: [%d, %d]\n", ffn->hidden_dim, ffn->ffn_dim);
    printf("  b1: [%d]\n", ffn->ffn_dim);
    printf("  W2: [%d, %d]\n", ffn->ffn_dim, ffn->hidden_dim);
    printf("  b2: [%d]\n", ffn->hidden_dim);
    printf("  total params: %d\n", ffn_num_params(ffn));
}

int ffn_num_params(FFN* ffn) {
    if (ffn == NULL) return 0;

    int w1_params = ffn->hidden_dim * ffn->ffn_dim;
    int b1_params = ffn->ffn_dim;
    int w2_params = ffn->ffn_dim * ffn->hidden_dim;
    int b2_params = ffn->hidden_dim;

    return w1_params + b1_params + w2_params + b2_params;
}
