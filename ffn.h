#ifndef FFN_H
#define FFN_H

#include "tensor.h"

/**
 * FFN - 前馈神经网络 (Feed-Forward Network)
 *
 * FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
 *
 * 通常 ffn_dim = 4 * hidden_dim
 */
typedef struct {
    Tensor* W1;         // 第一层权重 [hidden_dim, ffn_dim]
    Tensor* b1;         // 第一层偏置 [ffn_dim]
    Tensor* W2;         // 第二层权重 [ffn_dim, hidden_dim]
    Tensor* b2;         // 第二层偏置 [hidden_dim]
    int hidden_dim;
    int ffn_dim;
} FFN;

/**
 * FFN 计算缓存
 */
typedef struct {
    Tensor* hidden;     // 中间层输出 [seq_len, ffn_dim]
    int seq_len;
    int ffn_dim;
} FFNCache;

// ============ 创建和销毁 ============

/**
 * 创建 FFN 层
 * @param hidden_dim 输入/输出维度
 * @param ffn_dim 中间层维度 (通常为 4 * hidden_dim)
 * @return 新创建的 FFN, 失败返回 NULL
 */
FFN* ffn_create(int hidden_dim, int ffn_dim);

/**
 * 释放 FFN
 */
void ffn_free(FFN* ffn);

/**
 * 创建 FFN 缓存
 */
FFNCache* ffn_cache_create(int seq_len, int ffn_dim);

/**
 * 释放 FFN 缓存
 */
void ffn_cache_free(FFNCache* cache);

/**
 * 调整缓存大小
 */
int ffn_cache_resize(FFNCache* cache, int new_seq_len);

// ============ 初始化 ============

/**
 * 随机初始化 FFN 权重
 * @param ffn FFN 层
 * @param std 标准差 (推荐 0.02)
 */
void ffn_init_random(FFN* ffn, float std);

// ============ 前向传播 ============

/**
 * FFN 前向传播
 * @param ffn FFN 层
 * @param input 输入张量 [seq_len, hidden_dim]
 * @param cache 计算缓存
 * @param output 输出张量 [seq_len, hidden_dim] (需预先分配)
 */
void ffn_forward(FFN* ffn, Tensor* input, FFNCache* cache, Tensor* output);

// ============ 工具函数 ============

/**
 * 打印 FFN 信息
 */
void ffn_print_info(FFN* ffn);

/**
 * 计算参数数量
 */
int ffn_num_params(FFN* ffn);

#endif // FFN_H
