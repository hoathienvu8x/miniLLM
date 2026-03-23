#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

/**
 * LayerNorm - 层归一化
 *
 * 对每个样本的特征维度进行归一化:
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * 其中 mean 和 var 是在最后一个维度上计算的
 */
typedef struct {
    Tensor* gamma;      // 可学习缩放参数 [hidden_dim]
    Tensor* beta;       // 可学习偏移参数 [hidden_dim]
    int hidden_dim;
    float eps;          // 防止除零的小常数
} LayerNorm;

// ============ 创建和销毁 ============

/**
 * 创建 LayerNorm 层
 * @param hidden_dim 特征维度
 * @param eps 防止除零的小常数 (默认 1e-5)
 * @return 新创建的 LayerNorm, 失败返回 NULL
 */
LayerNorm* layernorm_create(int hidden_dim, float eps);

/**
 * 释放 LayerNorm
 */
void layernorm_free(LayerNorm* ln);

// ============ 初始化 ============

/**
 * 初始化 LayerNorm 参数
 * gamma 初始化为 1, beta 初始化为 0
 */
void layernorm_init(LayerNorm* ln);

// ============ 前向传播 ============

/**
 * LayerNorm 前向传播
 * @param ln LayerNorm 层
 * @param input 输入张量 [seq_len, hidden_dim] 或 [hidden_dim]
 * @param output 输出张量 (需预先分配, 形状与 input 相同)
 */
void layernorm_forward(LayerNorm* ln, Tensor* input, Tensor* output);

// ============ 工具函数 ============

/**
 * 打印 LayerNorm 信息
 */
void layernorm_print_info(LayerNorm* ln);

/**
 * 计算参数数量
 */
int layernorm_num_params(LayerNorm* ln);

#endif // LAYERNORM_H
