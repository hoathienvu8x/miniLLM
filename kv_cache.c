#include "kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

KVCache* kv_cache_create(int num_layers, int max_seq_len, int hidden_dim) {
    if (num_layers <= 0 || max_seq_len <= 0 || hidden_dim <= 0) {
        return NULL;
    }

    KVCache* cache = (KVCache*)malloc(sizeof(KVCache));
    if (cache == NULL) return NULL;

    cache->num_layers = num_layers;
    cache->max_seq_len = max_seq_len;
    cache->hidden_dim = hidden_dim;
    cache->current_len = 0;

    // 分配每层的缓存
    cache->layers = (LayerKVCache*)malloc(num_layers * sizeof(LayerKVCache));
    if (cache->layers == NULL) {
        free(cache);
        return NULL;
    }

    int shape[] = {max_seq_len, hidden_dim};

    for (int i = 0; i < num_layers; i++) {
        cache->layers[i].k_cache = tensor_zeros(2, shape);
        cache->layers[i].v_cache = tensor_zeros(2, shape);

        if (cache->layers[i].k_cache == NULL || cache->layers[i].v_cache == NULL) {
            // 清理已分配的
            for (int j = 0; j <= i; j++) {
                if (cache->layers[j].k_cache) tensor_free(cache->layers[j].k_cache);
                if (cache->layers[j].v_cache) tensor_free(cache->layers[j].v_cache);
            }
            free(cache->layers);
            free(cache);
            return NULL;
        }
    }

    return cache;
}

void kv_cache_free(KVCache* cache) {
    if (cache == NULL) return;

    if (cache->layers != NULL) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->layers[i].k_cache) tensor_free(cache->layers[i].k_cache);
            if (cache->layers[i].v_cache) tensor_free(cache->layers[i].v_cache);
        }
        free(cache->layers);
    }

    free(cache);
}

int kv_cache_update(KVCache* cache, int layer_idx,
                    Tensor* new_k, Tensor* new_v, int num_new_tokens) {
    if (cache == NULL || layer_idx < 0 || layer_idx >= cache->num_layers) {
        return -1;
    }
    if (new_k == NULL || new_v == NULL || num_new_tokens <= 0) {
        return -1;
    }

    // 检查是否超出最大长度
    if (cache->current_len + num_new_tokens > cache->max_seq_len) {
        fprintf(stderr, "KV Cache overflow: current=%d, new=%d, max=%d\n",
                cache->current_len, num_new_tokens, cache->max_seq_len);
        return -1;
    }

    LayerKVCache* layer = &cache->layers[layer_idx];
    int hidden_dim = cache->hidden_dim;
    int start_pos = cache->current_len;

    // 复制新的 K 值到缓存
    for (int t = 0; t < num_new_tokens; t++) {
        int cache_idx = (start_pos + t) * hidden_dim;
        int src_idx = t * hidden_dim;
        memcpy(&layer->k_cache->data[cache_idx],
               &new_k->data[src_idx],
               hidden_dim * sizeof(float));
    }

    // 复制新的 V 值到缓存
    for (int t = 0; t < num_new_tokens; t++) {
        int cache_idx = (start_pos + t) * hidden_dim;
        int src_idx = t * hidden_dim;
        memcpy(&layer->v_cache->data[cache_idx],
               &new_v->data[src_idx],
               hidden_dim * sizeof(float));
    }

    return 0;
}

void kv_cache_update_pos(KVCache* cache, int layer_idx, int pos,
                         float* k_data, float* v_data) {
    if (cache == NULL || layer_idx < 0 || layer_idx >= cache->num_layers) {
        return;
    }
    if (pos < 0 || pos >= cache->max_seq_len) {
        return;
    }

    LayerKVCache* layer = &cache->layers[layer_idx];
    int hidden_dim = cache->hidden_dim;
    int offset = pos * hidden_dim;

    if (k_data != NULL) {
        memcpy(&layer->k_cache->data[offset], k_data, hidden_dim * sizeof(float));
    }
    if (v_data != NULL) {
        memcpy(&layer->v_cache->data[offset], v_data, hidden_dim * sizeof(float));
    }
}

Tensor* kv_cache_get_k(KVCache* cache, int layer_idx) {
    if (cache == NULL || layer_idx < 0 || layer_idx >= cache->num_layers) {
        return NULL;
    }
    return cache->layers[layer_idx].k_cache;
}

Tensor* kv_cache_get_v(KVCache* cache, int layer_idx) {
    if (cache == NULL || layer_idx < 0 || layer_idx >= cache->num_layers) {
        return NULL;
    }
    return cache->layers[layer_idx].v_cache;
}

void kv_cache_clear(KVCache* cache) {
    if (cache == NULL) return;

    cache->current_len = 0;

    // 可选：清零数据 (不是必须的，因为 current_len 控制有效范围)
    // for (int i = 0; i < cache->num_layers; i++) {
    //     memset(cache->layers[i].k_cache->data, 0,
    //            cache->max_seq_len * cache->hidden_dim * sizeof(float));
    //     memset(cache->layers[i].v_cache->data, 0,
    //            cache->max_seq_len * cache->hidden_dim * sizeof(float));
    // }
}

void kv_cache_set_len(KVCache* cache, int len) {
    if (cache == NULL) return;
    if (len < 0) len = 0;
    if (len > cache->max_seq_len) len = cache->max_seq_len;
    cache->current_len = len;
}

void kv_cache_print_info(KVCache* cache) {
    if (cache == NULL) {
        printf("KVCache: NULL\n");
        return;
    }

    printf("KVCache Info:\n");
    printf("  num_layers: %d\n", cache->num_layers);
    printf("  max_seq_len: %d\n", cache->max_seq_len);
    printf("  hidden_dim: %d\n", cache->hidden_dim);
    printf("  current_len: %d\n", cache->current_len);
    printf("  memory: %.2f MB\n", kv_cache_memory_size(cache) / (1024.0 * 1024.0));
}

size_t kv_cache_memory_size(KVCache* cache) {
    if (cache == NULL) return 0;

    // 每层有 K 和 V 缓存，每个 [max_seq_len, hidden_dim]
    size_t per_layer = 2 * cache->max_seq_len * cache->hidden_dim * sizeof(float);
    return cache->num_layers * per_layer;
}
