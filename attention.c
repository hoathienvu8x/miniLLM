#include "attention.h"
#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

MultiHeadAttention* attention_create(int hidden_dim, int num_heads) {
    if (hidden_dim <= 0 || num_heads <= 0) return NULL;
    if (hidden_dim % num_heads != 0) {
        fprintf(stderr, "Error: hidden_dim (%d) must be divisible by num_heads (%d)\n",
                hidden_dim, num_heads);
        return NULL;
    }

    MultiHeadAttention* attn = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    if (attn == NULL) return NULL;

    attn->hidden_dim = hidden_dim;
    attn->num_heads = num_heads;
    attn->head_dim = hidden_dim / num_heads;
    attn->scale = 1.0f / sqrtf((float)attn->head_dim);

    // 创建权重矩阵 [hidden_dim, hidden_dim]
    int shape[] = {hidden_dim, hidden_dim};

    attn->W_q = tensor_zeros(2, shape);
    attn->W_k = tensor_zeros(2, shape);
    attn->W_v = tensor_zeros(2, shape);
    attn->W_o = tensor_zeros(2, shape);

    if (attn->W_q == NULL || attn->W_k == NULL ||
        attn->W_v == NULL || attn->W_o == NULL) {
        attention_free(attn);
        return NULL;
    }

    return attn;
}

void attention_free(MultiHeadAttention* attn) {
    if (attn == NULL) return;

    if (attn->W_q) tensor_free(attn->W_q);
    if (attn->W_k) tensor_free(attn->W_k);
    if (attn->W_v) tensor_free(attn->W_v);
    if (attn->W_o) tensor_free(attn->W_o);

    free(attn);
}

AttentionCache* attention_cache_create(int seq_len, int hidden_dim, int num_heads) {
    AttentionCache* cache = (AttentionCache*)malloc(sizeof(AttentionCache));
    if (cache == NULL) return NULL;

    cache->seq_len = seq_len;
    cache->hidden_dim = hidden_dim;
    cache->num_heads = num_heads;

    int qkv_shape[] = {seq_len, hidden_dim};
    int scores_shape[] = {num_heads, seq_len, seq_len};

    cache->Q = tensor_zeros(2, qkv_shape);
    cache->K = tensor_zeros(2, qkv_shape);
    cache->V = tensor_zeros(2, qkv_shape);
    cache->scores = tensor_zeros(3, scores_shape);
    cache->attn = tensor_zeros(3, scores_shape);
    cache->attn_out = tensor_zeros(2, qkv_shape);

    if (cache->Q == NULL || cache->K == NULL || cache->V == NULL ||
        cache->scores == NULL || cache->attn == NULL || cache->attn_out == NULL) {
        attention_cache_free(cache);
        return NULL;
    }

    return cache;
}

void attention_cache_free(AttentionCache* cache) {
    if (cache == NULL) return;

    if (cache->Q) tensor_free(cache->Q);
    if (cache->K) tensor_free(cache->K);
    if (cache->V) tensor_free(cache->V);
    if (cache->scores) tensor_free(cache->scores);
    if (cache->attn) tensor_free(cache->attn);
    if (cache->attn_out) tensor_free(cache->attn_out);

    free(cache);
}

int attention_cache_resize(AttentionCache* cache, int new_seq_len) {
    if (cache == NULL) return -1;
    if (new_seq_len == cache->seq_len) return 0;

    // 释放旧的
    if (cache->Q) tensor_free(cache->Q);
    if (cache->K) tensor_free(cache->K);
    if (cache->V) tensor_free(cache->V);
    if (cache->scores) tensor_free(cache->scores);
    if (cache->attn) tensor_free(cache->attn);
    if (cache->attn_out) tensor_free(cache->attn_out);

    // 重新分配
    int qkv_shape[] = {new_seq_len, cache->hidden_dim};
    int scores_shape[] = {cache->num_heads, new_seq_len, new_seq_len};

    cache->Q = tensor_zeros(2, qkv_shape);
    cache->K = tensor_zeros(2, qkv_shape);
    cache->V = tensor_zeros(2, qkv_shape);
    cache->scores = tensor_zeros(3, scores_shape);
    cache->attn = tensor_zeros(3, scores_shape);
    cache->attn_out = tensor_zeros(2, qkv_shape);
    cache->seq_len = new_seq_len;

    if (cache->Q == NULL || cache->K == NULL || cache->V == NULL ||
        cache->scores == NULL || cache->attn == NULL || cache->attn_out == NULL) {
        return -1;
    }

    return 0;
}

void attention_init_random(MultiHeadAttention* attn, float std) {
    if (attn == NULL) return;

    int size = attn->hidden_dim * attn->hidden_dim;

    // Box-Muller 初始化
    for (int i = 0; i < size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        attn->W_q->data[i] = z * std;
    }

    for (int i = 0; i < size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        attn->W_k->data[i] = z * std;
    }

    for (int i = 0; i < size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        attn->W_v->data[i] = z * std;
    }

    for (int i = 0; i < size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        attn->W_o->data[i] = z * std;
    }
}

Tensor* create_causal_mask(int seq_len) {
    int shape[] = {seq_len, seq_len};
    Tensor* mask = tensor_zeros(2, shape);
    if (mask == NULL) return NULL;

    // mask[i][j] = 0 if j <= i, else -inf
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                mask->data[i * seq_len + j] = -1e9f;  // 用大负数代替 -inf
            }
        }
    }

    return mask;
}

void single_head_attention(
    Tensor* Q, Tensor* K, Tensor* V,
    Tensor* mask, float scale,
    Tensor* output
) {
    int seq_len = Q->shape[0];
    int head_dim = Q->shape[1];

    // 分配临时 scores 和 attn
    int scores_shape[] = {seq_len, seq_len};
    Tensor* scores = tensor_zeros(2, scores_shape);
    Tensor* attn_weights = tensor_zeros(2, scores_shape);

    // scores = Q @ K^T
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += Q->data[i * head_dim + d] * K->data[j * head_dim + d];
            }
            scores->data[i * seq_len + j] = sum * scale;
        }
    }

    // scores = scores + mask
    if (mask != NULL) {
        for (int i = 0; i < seq_len * seq_len; i++) {
            scores->data[i] += mask->data[i];
        }
    }

    // attn = softmax(scores) - 对每行
    for (int i = 0; i < seq_len; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < seq_len; j++) {
            if (scores->data[i * seq_len + j] > max_val) {
                max_val = scores->data[i * seq_len + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            attn_weights->data[i * seq_len + j] = expf(scores->data[i * seq_len + j] - max_val);
            sum += attn_weights->data[i * seq_len + j];
        }

        for (int j = 0; j < seq_len; j++) {
            attn_weights->data[i * seq_len + j] /= sum;
        }
    }

    // output = attn @ V
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += attn_weights->data[i * seq_len + j] * V->data[j * head_dim + d];
            }
            output->data[i * head_dim + d] = sum;
        }
    }

    tensor_free(scores);
    tensor_free(attn_weights);
}

void attention_forward(
    MultiHeadAttention* attn,
    Tensor* input,
    Tensor* mask,
    AttentionCache* cache,
    Tensor* output
) {
    if (attn == NULL || input == NULL || cache == NULL || output == NULL) return;

    int seq_len = input->shape[0];
    int hidden_dim = attn->hidden_dim;
    int num_heads = attn->num_heads;
    int head_dim = attn->head_dim;

    // 检查并调整缓存大小
    if (cache->seq_len != seq_len) {
        attention_cache_resize(cache, seq_len);
    }

    // Step 1: 计算 Q, K, V
    // Q = input @ W_q, K = input @ W_k, V = input @ W_v
    matmul_inplace(cache->Q, input, attn->W_q);
    matmul_inplace(cache->K, input, attn->W_k);
    matmul_inplace(cache->V, input, attn->W_v);

    // Step 2-6: 多头注意力计算
    // 对每个头分别计算
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        // 对每个位置计算 scores
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                // Q[i, head_offset:head_offset+head_dim] @ K[j, head_offset:head_offset+head_dim]
                for (int d = 0; d < head_dim; d++) {
                    float q_val = cache->Q->data[i * hidden_dim + head_offset + d];
                    float k_val = cache->K->data[j * hidden_dim + head_offset + d];
                    sum += q_val * k_val;
                }
                // scores[h, i, j] = sum * scale
                cache->scores->data[h * seq_len * seq_len + i * seq_len + j] = sum * attn->scale;
            }
        }
    }

    // 添加掩码
    if (mask != NULL) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    int idx = h * seq_len * seq_len + i * seq_len + j;
                    cache->scores->data[idx] += mask->data[i * seq_len + j];
                }
            }
        }
    }

    // Softmax (对每个头的每行)
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            int row_start = h * seq_len * seq_len + i * seq_len;

            // 找最大值
            float max_val = -FLT_MAX;
            for (int j = 0; j < seq_len; j++) {
                if (cache->scores->data[row_start + j] > max_val) {
                    max_val = cache->scores->data[row_start + j];
                }
            }

            // exp 和 sum
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                cache->attn->data[row_start + j] = expf(cache->scores->data[row_start + j] - max_val);
                sum += cache->attn->data[row_start + j];
            }

            // 归一化
            for (int j = 0; j < seq_len; j++) {
                cache->attn->data[row_start + j] /= sum;
            }
        }
    }

    // attn @ V -> attn_out
    // 清零 attn_out
    memset(cache->attn_out->data, 0, cache->attn_out->size * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float attn_val = cache->attn->data[h * seq_len * seq_len + i * seq_len + j];
                    float v_val = cache->V->data[j * hidden_dim + head_offset + d];
                    sum += attn_val * v_val;
                }
                cache->attn_out->data[i * hidden_dim + head_offset + d] = sum;
            }
        }
    }

    // Step 7: output = attn_out @ W_o
    matmul_inplace(output, cache->attn_out, attn->W_o);
}

void attention_print_info(MultiHeadAttention* attn) {
    if (attn == NULL) {
        printf("MultiHeadAttention: NULL\n");
        return;
    }

    printf("MultiHeadAttention Info:\n");
    printf("  hidden_dim: %d\n", attn->hidden_dim);
    printf("  num_heads: %d\n", attn->num_heads);
    printf("  head_dim: %d\n", attn->head_dim);
    printf("  scale: %.6f\n", attn->scale);
    printf("  W_q: [%d, %d]\n", attn->W_q->shape[0], attn->W_q->shape[1]);
    printf("  W_k: [%d, %d]\n", attn->W_k->shape[0], attn->W_k->shape[1]);
    printf("  W_v: [%d, %d]\n", attn->W_v->shape[0], attn->W_v->shape[1]);
    printf("  W_o: [%d, %d]\n", attn->W_o->shape[0], attn->W_o->shape[1]);
    printf("  total params: %d\n", attention_num_params(attn));
}

int attention_num_params(MultiHeadAttention* attn) {
    if (attn == NULL) return 0;

    // 4 个权重矩阵，每个 [hidden_dim, hidden_dim]
    return 4 * attn->hidden_dim * attn->hidden_dim;
}

Tensor* attention_get_weights(AttentionCache* cache) {
    if (cache == NULL) return NULL;
    return cache->attn;
}

// ============ KV Cache 支持实现 ============

void attention_prefill_kv_cache(
    MultiHeadAttention* attn,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    Tensor* mask,
    AttentionCache* cache,
    Tensor* output
) {
    if (attn == NULL || input == NULL || kv_cache == NULL ||
        cache == NULL || output == NULL) return;

    int seq_len = input->shape[0];
    int hidden_dim = attn->hidden_dim;
    int num_heads = attn->num_heads;
    int head_dim = attn->head_dim;

    // 检查并调整缓存大小
    if (cache->seq_len != seq_len) {
        attention_cache_resize(cache, seq_len);
    }

    // Step 1: 计算 Q, K, V
    matmul_inplace(cache->Q, input, attn->W_q);
    matmul_inplace(cache->K, input, attn->W_k);
    matmul_inplace(cache->V, input, attn->W_v);

    // Step 2: 将 K, V 存入 KV Cache
    // 注意: 这里从位置 0 开始存储 (prefill 阶段)
    kv_cache_update(kv_cache, layer_idx, cache->K, cache->V, seq_len);

    // Step 3-6: 多头注意力计算 (使用刚存入的 K, V)
    // 这里可以直接使用 cache->K 和 cache->V，因为它们包含了所有需要的数据

    // 计算 scores
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = cache->Q->data[i * hidden_dim + head_offset + d];
                    float k_val = cache->K->data[j * hidden_dim + head_offset + d];
                    sum += q_val * k_val;
                }
                cache->scores->data[h * seq_len * seq_len + i * seq_len + j] = sum * attn->scale;
            }
        }
    }

    // 添加掩码
    if (mask != NULL) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    int idx = h * seq_len * seq_len + i * seq_len + j;
                    cache->scores->data[idx] += mask->data[i * seq_len + j];
                }
            }
        }
    }

    // Softmax
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            int row_start = h * seq_len * seq_len + i * seq_len;

            float max_val = -FLT_MAX;
            for (int j = 0; j < seq_len; j++) {
                if (cache->scores->data[row_start + j] > max_val) {
                    max_val = cache->scores->data[row_start + j];
                }
            }

            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                cache->attn->data[row_start + j] = expf(cache->scores->data[row_start + j] - max_val);
                sum += cache->attn->data[row_start + j];
            }

            for (int j = 0; j < seq_len; j++) {
                cache->attn->data[row_start + j] /= sum;
            }
        }
    }

    // attn @ V
    memset(cache->attn_out->data, 0, cache->attn_out->size * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float attn_val = cache->attn->data[h * seq_len * seq_len + i * seq_len + j];
                    float v_val = cache->V->data[j * hidden_dim + head_offset + d];
                    sum += attn_val * v_val;
                }
                cache->attn_out->data[i * hidden_dim + head_offset + d] = sum;
            }
        }
    }

    // output = attn_out @ W_o
    matmul_inplace(output, cache->attn_out, attn->W_o);
}

void attention_forward_kv_cache(
    MultiHeadAttention* attn,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    int start_pos,
    AttentionCache* cache,
    Tensor* output
) {
    if (attn == NULL || input == NULL || kv_cache == NULL ||
        cache == NULL || output == NULL) return;

    int cur_seq_len = input->shape[0];  // 通常为 1 (单 token 解码)
    int hidden_dim = attn->hidden_dim;
    int num_heads = attn->num_heads;
    int head_dim = attn->head_dim;
    int total_len = start_pos + cur_seq_len;  // 包括新 token 的总长度

    // 确保缓存大小足够
    if (cache->seq_len < cur_seq_len) {
        attention_cache_resize(cache, cur_seq_len);
    }

    // Step 1: 计算新 token 的 Q, K, V
    // 使用临时张量存储
    int qkv_shape[] = {cur_seq_len, hidden_dim};
    Tensor* new_Q = tensor_zeros(2, qkv_shape);
    Tensor* new_K = tensor_zeros(2, qkv_shape);
    Tensor* new_V = tensor_zeros(2, qkv_shape);

    matmul_inplace(new_Q, input, attn->W_q);
    matmul_inplace(new_K, input, attn->W_k);
    matmul_inplace(new_V, input, attn->W_v);

    // Step 2: 将新的 K, V 追加到 KV Cache
    // 直接存入指定位置
    for (int t = 0; t < cur_seq_len; t++) {
        kv_cache_update_pos(kv_cache, layer_idx, start_pos + t,
                           &new_K->data[t * hidden_dim],
                           &new_V->data[t * hidden_dim]);
    }

    // 更新 KV Cache 长度
    if (kv_cache->current_len < total_len) {
        kv_cache_set_len(kv_cache, total_len);
    }

    // Step 3: 从 KV Cache 获取完整的 K, V
    Tensor* cached_K = kv_cache_get_k(kv_cache, layer_idx);
    Tensor* cached_V = kv_cache_get_v(kv_cache, layer_idx);

    // Step 4: 计算注意力 scores
    // Q: [cur_seq_len, hidden_dim], K: [total_len, hidden_dim]
    // scores: [num_heads, cur_seq_len, total_len]

    // 分配临时 scores 和 attn
    int scores_shape[] = {num_heads, cur_seq_len, total_len};
    Tensor* scores = tensor_zeros(3, scores_shape);
    Tensor* attn_weights = tensor_zeros(3, scores_shape);

    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        for (int i = 0; i < cur_seq_len; i++) {
            for (int j = 0; j < total_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = new_Q->data[i * hidden_dim + head_offset + d];
                    float k_val = cached_K->data[j * hidden_dim + head_offset + d];
                    sum += q_val * k_val;
                }
                scores->data[h * cur_seq_len * total_len + i * total_len + j] = sum * attn->scale;
            }
        }
    }

    // Step 5: 应用因果掩码 (只对解码阶段有效)
    // 位置 start_pos + i 只能看到位置 0 到 start_pos + i
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < cur_seq_len; i++) {
            int cur_pos = start_pos + i;
            for (int j = 0; j < total_len; j++) {
                if (j > cur_pos) {
                    scores->data[h * cur_seq_len * total_len + i * total_len + j] = -1e9f;
                }
            }
        }
    }

    // Step 6: Softmax
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < cur_seq_len; i++) {
            int row_start = h * cur_seq_len * total_len + i * total_len;

            float max_val = -FLT_MAX;
            for (int j = 0; j < total_len; j++) {
                if (scores->data[row_start + j] > max_val) {
                    max_val = scores->data[row_start + j];
                }
            }

            float sum = 0.0f;
            for (int j = 0; j < total_len; j++) {
                attn_weights->data[row_start + j] = expf(scores->data[row_start + j] - max_val);
                sum += attn_weights->data[row_start + j];
            }

            for (int j = 0; j < total_len; j++) {
                attn_weights->data[row_start + j] /= sum;
            }
        }
    }

    // Step 7: attn @ V -> output_pre_proj
    int out_shape[] = {cur_seq_len, hidden_dim};
    Tensor* attn_out = tensor_zeros(2, out_shape);

    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        for (int i = 0; i < cur_seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < total_len; j++) {
                    float attn_val = attn_weights->data[h * cur_seq_len * total_len + i * total_len + j];
                    float v_val = cached_V->data[j * hidden_dim + head_offset + d];
                    sum += attn_val * v_val;
                }
                attn_out->data[i * hidden_dim + head_offset + d] = sum;
            }
        }
    }

    // Step 8: output = attn_out @ W_o
    matmul_inplace(output, attn_out, attn->W_o);

    // 清理临时张量
    tensor_free(new_Q);
    tensor_free(new_K);
    tensor_free(new_V);
    tensor_free(scores);
    tensor_free(attn_weights);
    tensor_free(attn_out);
}
