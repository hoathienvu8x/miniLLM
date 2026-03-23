#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "kv_cache.h"

/**
 * MultiHeadAttention - 多头自注意力层
 *
 * 计算流程:
 * 1. Q = input @ W_q, K = input @ W_k, V = input @ W_v
 * 2. 分成多个头: [seq_len, hidden_dim] -> [num_heads, seq_len, head_dim]
 * 3. scores = Q @ K^T / sqrt(head_dim)
 * 4. scores = scores + mask (因果掩码)
 * 5. attn = softmax(scores)
 * 6. out = attn @ V
 * 7. out = concat(heads) @ W_o
 */
typedef struct {
    Tensor* W_q;        // Query 权重 [hidden_dim, hidden_dim]
    Tensor* W_k;        // Key 权重 [hidden_dim, hidden_dim]
    Tensor* W_v;        // Value 权重 [hidden_dim, hidden_dim]
    Tensor* W_o;        // Output 权重 [hidden_dim, hidden_dim]

    int hidden_dim;     // 隐藏维度
    int num_heads;      // 注意力头数
    int head_dim;       // 每个头的维度 = hidden_dim / num_heads
    float scale;        // 缩放因子 = 1 / sqrt(head_dim)
} MultiHeadAttention;

/**
 * AttentionCache - 注意力计算的中间缓存
 * 用于避免重复内存分配
 */
typedef struct {
    Tensor* Q;          // [seq_len, hidden_dim]
    Tensor* K;          // [seq_len, hidden_dim]
    Tensor* V;          // [seq_len, hidden_dim]
    Tensor* scores;     // [num_heads, seq_len, seq_len]
    Tensor* attn;       // [num_heads, seq_len, seq_len]
    Tensor* attn_out;   // [seq_len, hidden_dim]
    int seq_len;
    int hidden_dim;
    int num_heads;
} AttentionCache;

// ============ 创建和销毁 ============

/**
 * 创建多头注意力层
 * @param hidden_dim 隐藏维度 (必须能被 num_heads 整除)
 * @param num_heads 注意力头数
 * @return 新创建的注意力层, 失败返回 NULL
 */
MultiHeadAttention* attention_create(int hidden_dim, int num_heads);

/**
 * 释放注意力层
 */
void attention_free(MultiHeadAttention* attn);

/**
 * 创建注意力缓存
 */
AttentionCache* attention_cache_create(int seq_len, int hidden_dim, int num_heads);

/**
 * 释放注意力缓存
 */
void attention_cache_free(AttentionCache* cache);

/**
 * 调整缓存大小 (如果 seq_len 变化)
 */
int attention_cache_resize(AttentionCache* cache, int new_seq_len);

// ============ 初始化 ============

/**
 * 随机初始化注意力权重
 * @param attn 注意力层
 * @param std 标准差 (推荐 0.02)
 */
void attention_init_random(MultiHeadAttention* attn, float std);

// ============ 掩码 ============

/**
 * 创建因果掩码 (下三角)
 * mask[i][j] = 0 if j <= i, else -inf
 * @param seq_len 序列长度
 * @return 掩码张量 [seq_len, seq_len]
 */
Tensor* create_causal_mask(int seq_len);

// ============ 前向传播 ============

/**
 * 多头注意力前向传播
 * @param attn 注意力层
 * @param input 输入张量 [seq_len, hidden_dim]
 * @param mask 注意力掩码 [seq_len, seq_len], 可为 NULL
 * @param cache 计算缓存
 * @param output 输出张量 [seq_len, hidden_dim] (需预先分配)
 */
void attention_forward(
    MultiHeadAttention* attn,
    Tensor* input,
    Tensor* mask,
    AttentionCache* cache,
    Tensor* output
);

/**
 * 单头注意力 (用于测试和理解)
 * @param Q Query [seq_len, head_dim]
 * @param K Key [seq_len, head_dim]
 * @param V Value [seq_len, head_dim]
 * @param mask 掩码 [seq_len, seq_len], 可为 NULL
 * @param scale 缩放因子
 * @param output 输出 [seq_len, head_dim]
 */
void single_head_attention(
    Tensor* Q, Tensor* K, Tensor* V,
    Tensor* mask, float scale,
    Tensor* output
);

// ============ 工具函数 ============

/**
 * 打印注意力层信息
 */
void attention_print_info(MultiHeadAttention* attn);

/**
 * 计算注意力层参数数量
 */
int attention_num_params(MultiHeadAttention* attn);

/**
 * 获取注意力权重 (用于可视化)
 * 需在 attention_forward 后调用
 * @return 指向 cache->attn 的指针
 */
Tensor* attention_get_weights(AttentionCache* cache);

// ============ KV Cache 支持 ============

/**
 * 带 KV Cache 的注意力前向传播
 *
 * 工作流程:
 * 1. 计算当前输入的 Q, K, V
 * 2. 将新的 K, V 追加到 KV Cache
 * 3. 使用缓存中的所有 K, V 计算注意力
 * 4. 输出结果
 *
 * @param attn 注意力层
 * @param input 输入张量 [seq_len, hidden_dim] (seq_len 通常为 1 用于解码)
 * @param kv_cache KV 缓存 (会被更新)
 * @param layer_idx 当前层索引 (用于更新对应层的 KV Cache)
 * @param start_pos 当前位置 (用于位置编码，等于之前的 token 数)
 * @param cache 计算缓存 (临时存储)
 * @param output 输出张量 [seq_len, hidden_dim]
 */
void attention_forward_kv_cache(
    MultiHeadAttention* attn,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    int start_pos,
    AttentionCache* cache,
    Tensor* output
);

/**
 * Prefill 阶段: 处理整个 prompt 并填充 KV Cache
 *
 * @param attn 注意力层
 * @param input 输入张量 [seq_len, hidden_dim]
 * @param kv_cache KV 缓存 (会被填充)
 * @param layer_idx 当前层索引
 * @param mask 因果掩码 [seq_len, seq_len]
 * @param cache 计算缓存
 * @param output 输出张量 [seq_len, hidden_dim]
 */
void attention_prefill_kv_cache(
    MultiHeadAttention* attn,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    Tensor* mask,
    AttentionCache* cache,
    Tensor* output
);

#endif // ATTENTION_H
