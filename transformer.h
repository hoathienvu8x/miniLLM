#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "tensor.h"
#include "layernorm.h"
#include "attention.h"
#include "ffn.h"
#include "kv_cache.h"

/**
 * TransformerBlock - Transformer 解码器块
 *
 * 使用 Pre-LN (Pre-LayerNorm) 结构:
 * x = x + Attention(LayerNorm(x))
 * x = x + FFN(LayerNorm(x))
 */
typedef struct {
    LayerNorm* ln1;             // 注意力前的 LayerNorm
    MultiHeadAttention* attn;   // 多头自注意力
    LayerNorm* ln2;             // FFN 前的 LayerNorm
    FFN* ffn;                   // 前馈网络

    int hidden_dim;
    int num_heads;
    int ffn_dim;
} TransformerBlock;

/**
 * TransformerBlock 计算缓存
 */
typedef struct {
    Tensor* ln1_out;        // LayerNorm1 输出
    Tensor* attn_out;       // Attention 输出
    Tensor* ln2_out;        // LayerNorm2 输出
    Tensor* ffn_out;        // FFN 输出
    AttentionCache* attn_cache;
    FFNCache* ffn_cache;
    int seq_len;
    int hidden_dim;
} TransformerCache;

// ============ 创建和销毁 ============

/**
 * 创建 Transformer Block
 * @param hidden_dim 隐藏维度
 * @param num_heads 注意力头数
 * @param ffn_dim FFN 中间层维度
 * @return 新创建的 TransformerBlock, 失败返回 NULL
 */
TransformerBlock* transformer_block_create(int hidden_dim, int num_heads, int ffn_dim);

/**
 * 释放 Transformer Block
 */
void transformer_block_free(TransformerBlock* block);

/**
 * 创建 Transformer 缓存
 */
TransformerCache* transformer_cache_create(int seq_len, int hidden_dim, int num_heads, int ffn_dim);

/**
 * 释放 Transformer 缓存
 */
void transformer_cache_free(TransformerCache* cache);

/**
 * 调整缓存大小
 */
int transformer_cache_resize(TransformerCache* cache, int new_seq_len);

// ============ 初始化 ============

/**
 * 初始化 Transformer Block 参数
 * @param block Transformer Block
 * @param std 权重标准差 (推荐 0.02)
 */
void transformer_block_init(TransformerBlock* block, float std);

// ============ 前向传播 ============

/**
 * Transformer Block 前向传播
 * @param block Transformer Block
 * @param input 输入张量 [seq_len, hidden_dim]
 * @param mask 注意力掩码 [seq_len, seq_len], 可为 NULL
 * @param cache 计算缓存
 * @param output 输出张量 [seq_len, hidden_dim] (需预先分配)
 */
void transformer_block_forward(
    TransformerBlock* block,
    Tensor* input,
    Tensor* mask,
    TransformerCache* cache,
    Tensor* output
);

// ============ 工具函数 ============

/**
 * 打印 Transformer Block 信息
 */
void transformer_block_print_info(TransformerBlock* block);

/**
 * 计算参数数量
 */
int transformer_block_num_params(TransformerBlock* block);

// ============ KV Cache 支持 ============

/**
 * Transformer Block 前向传播 (Prefill 阶段)
 * 处理整个序列并填充 KV Cache
 *
 * @param block Transformer Block
 * @param input 输入张量 [seq_len, hidden_dim]
 * @param kv_cache KV 缓存
 * @param layer_idx 层索引
 * @param mask 因果掩码
 * @param cache 计算缓存
 * @param output 输出张量 [seq_len, hidden_dim]
 */
void transformer_block_forward_prefill(
    TransformerBlock* block,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    Tensor* mask,
    TransformerCache* cache,
    Tensor* output
);

/**
 * Transformer Block 前向传播 (Decode 阶段)
 * 处理单个 token，使用 KV Cache
 *
 * @param block Transformer Block
 * @param input 输入张量 [1, hidden_dim]
 * @param kv_cache KV 缓存
 * @param layer_idx 层索引
 * @param pos 当前位置
 * @param cache 计算缓存
 * @param output 输出张量 [1, hidden_dim]
 */
void transformer_block_forward_decode(
    TransformerBlock* block,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    int pos,
    TransformerCache* cache,
    Tensor* output
);

#endif // TRANSFORMER_H
