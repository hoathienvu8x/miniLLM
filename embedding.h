#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

/**
 * Embedding - 词嵌入层
 *
 * 将离散的 token ID 映射到连续的向量空间
 * 包含:
 * - token_embedding: 词嵌入矩阵 [vocab_size, hidden_dim]
 * - position_embedding: 位置编码 [max_seq_len, hidden_dim]
 */
typedef struct {
    Tensor* token_embedding;      // [vocab_size, hidden_dim]
    Tensor* position_embedding;   // [max_seq_len, hidden_dim]
    int vocab_size;
    int hidden_dim;
    int max_seq_len;
} Embedding;

// ============ 创建和销毁 ============

/**
 * 创建 Embedding 层
 * @param vocab_size 词汇表大小
 * @param hidden_dim 嵌入维度
 * @param max_seq_len 最大序列长度
 * @return 新创建的 Embedding, 失败返回 NULL
 */
Embedding* embedding_create(int vocab_size, int hidden_dim, int max_seq_len);

/**
 * 释放 Embedding
 */
void embedding_free(Embedding* emb);

// ============ 初始化 ============

/**
 * 使用正态分布随机初始化 token embedding
 * @param emb Embedding 层
 * @param std 标准差 (推荐 0.02)
 */
void embedding_init_random(Embedding* emb, float std);

/**
 * 使用正弦位置编码初始化 position embedding
 * PE(pos, 2i)   = sin(pos / 10000^(2i/d))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 */
void embedding_init_sinusoidal_position(Embedding* emb);

/**
 * 使用可学习的随机位置编码
 */
void embedding_init_learned_position(Embedding* emb, float std);

// ============ 前向传播 ============

/**
 * Embedding 前向传播
 * @param emb Embedding 层
 * @param token_ids 输入 token ID 数组 [seq_len]
 * @param seq_len 序列长度
 * @param output 输出张量 [seq_len, hidden_dim] (需预先分配)
 */
void embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

/**
 * 仅获取 token embedding (不加位置编码)
 */
void embedding_get_token(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

/**
 * 仅获取 position embedding
 */
void embedding_get_position(Embedding* emb, int seq_len, Tensor* output);

// ============ 工具函数 ============

/**
 * 打印 Embedding 信息
 */
void embedding_print_info(Embedding* emb);

/**
 * 获取单个 token 的嵌入向量
 * @return 指向嵌入向量的指针 (不需要 free)
 */
float* embedding_get_vector(Embedding* emb, int token_id);

/**
 * 计算 Embedding 层的参数数量
 */
int embedding_num_params(Embedding* emb);

#endif // EMBEDDING_H
