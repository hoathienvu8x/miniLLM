#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "tensor.h"

/**
 * KV Cache - Key-Value 缓存用于高效推理
 *
 * 在自回归生成中，每次生成新 token 时:
 * - 不使用 KV Cache: 需要对所有 token 重新计算 K, V
 * - 使用 KV Cache: 只需计算新 token 的 K, V，复用之前缓存的值
 *
 * 这可以将推理复杂度从 O(n^2) 降低到 O(n)
 *
 * 内存布局:
 *   K_cache[layer][seq_pos][hidden_dim]
 *   V_cache[layer][seq_pos][hidden_dim]
 */

/**
 * 单层的 KV 缓存
 */
typedef struct {
    Tensor* k_cache;        // [max_seq_len, hidden_dim]
    Tensor* v_cache;        // [max_seq_len, hidden_dim]
} LayerKVCache;

/**
 * 完整模型的 KV 缓存
 */
typedef struct {
    LayerKVCache* layers;   // [num_layers]
    int num_layers;         // Transformer 层数
    int max_seq_len;        // 最大序列长度
    int hidden_dim;         // 隐藏维度
    int current_len;        // 当前缓存的序列长度
} KVCache;

// ============ 创建和销毁 ============

/**
 * 创建 KV 缓存
 * @param num_layers Transformer 层数
 * @param max_seq_len 最大序列长度
 * @param hidden_dim 隐藏维度
 * @return KV 缓存实例，失败返回 NULL
 */
KVCache* kv_cache_create(int num_layers, int max_seq_len, int hidden_dim);

/**
 * 释放 KV 缓存
 */
void kv_cache_free(KVCache* cache);

// ============ 缓存操作 ============

/**
 * 更新指定层的 KV 缓存 (追加新的 K, V)
 * @param cache KV 缓存
 * @param layer_idx 层索引
 * @param new_k 新的 K 值 [num_new_tokens, hidden_dim]
 * @param new_v 新的 V 值 [num_new_tokens, hidden_dim]
 * @param num_new_tokens 新 token 数量
 * @return 0 成功, -1 失败
 */
int kv_cache_update(KVCache* cache, int layer_idx,
                    Tensor* new_k, Tensor* new_v, int num_new_tokens);

/**
 * 更新指定层的单个位置的 KV 缓存
 * @param cache KV 缓存
 * @param layer_idx 层索引
 * @param pos 位置索引
 * @param k_data K 值数据 [hidden_dim]
 * @param v_data V 值数据 [hidden_dim]
 */
void kv_cache_update_pos(KVCache* cache, int layer_idx, int pos,
                         float* k_data, float* v_data);

/**
 * 获取指定层的 K 缓存 (到当前位置)
 * @param cache KV 缓存
 * @param layer_idx 层索引
 * @return K 缓存张量的指针
 */
Tensor* kv_cache_get_k(KVCache* cache, int layer_idx);

/**
 * 获取指定层的 V 缓存 (到当前位置)
 * @param cache KV 缓存
 * @param layer_idx 层索引
 * @return V 缓存张量的指针
 */
Tensor* kv_cache_get_v(KVCache* cache, int layer_idx);

/**
 * 清空 KV 缓存 (重置 current_len 为 0)
 */
void kv_cache_clear(KVCache* cache);

/**
 * 设置当前缓存长度
 * @param cache KV 缓存
 * @param len 新的长度
 */
void kv_cache_set_len(KVCache* cache, int len);

// ============ 工具函数 ============

/**
 * 打印 KV 缓存信息
 */
void kv_cache_print_info(KVCache* cache);

/**
 * 计算 KV 缓存内存占用 (字节)
 */
size_t kv_cache_memory_size(KVCache* cache);

#endif // KV_CACHE_H
