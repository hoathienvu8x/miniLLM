#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"

// ============ 矩阵运算 ============

/**
 * 矩阵乘法: C = A @ B
 * @param a 矩阵 A, 形状 [M, K]
 * @param b 矩阵 B, 形状 [K, N]
 * @return 结果矩阵 C, 形状 [M, N]
 */
Tensor* matmul(Tensor* a, Tensor* b);

/**
 * 矩阵乘法 (B 转置): C = A @ B^T
 * @param a 矩阵 A, 形状 [M, K]
 * @param b 矩阵 B, 形状 [N, K] (将被视为 [K, N] 的转置)
 * @return 结果矩阵 C, 形状 [M, N]
 */
Tensor* matmul_transposed_b(Tensor* a, Tensor* b);

/**
 * 矩阵乘法 (原地操作): out = A @ B
 * @param out 输出矩阵, 形状 [M, N]
 * @param a 矩阵 A, 形状 [M, K]
 * @param b 矩阵 B, 形状 [K, N]
 */
void matmul_inplace(Tensor* out, Tensor* a, Tensor* b);

/**
 * 矩阵-向量乘法: out = mat @ vec
 * @param out 输出向量, 形状 [M]
 * @param mat 矩阵, 形状 [M, N]
 * @param vec 向量, 形状 [N]
 */
void matvec(Tensor* out, Tensor* mat, Tensor* vec);

/**
 * 批量矩阵乘法: C = A @ B (支持 3D 张量)
 * @param a 形状 [batch, M, K]
 * @param b 形状 [batch, K, N]
 * @return 形状 [batch, M, N]
 */
Tensor* batched_matmul(Tensor* a, Tensor* b);

// ============ 逐元素运算 ============

/**
 * 逐元素加法: out = a + b
 */
void tensor_add(Tensor* out, Tensor* a, Tensor* b);

/**
 * 原地加法: a += b
 */
void tensor_add_inplace(Tensor* a, Tensor* b);

/**
 * 逐元素减法: out = a - b
 */
void tensor_sub(Tensor* out, Tensor* a, Tensor* b);

/**
 * 逐元素乘法 (Hadamard 积): out = a * b
 */
void tensor_mul(Tensor* out, Tensor* a, Tensor* b);

/**
 * 标量乘法: out = a * scalar
 */
void tensor_mul_scalar(Tensor* out, Tensor* a, float scalar);

/**
 * 原地标量乘法: a *= scalar
 */
void tensor_mul_scalar_inplace(Tensor* a, float scalar);

/**
 * 标量加法: out = a + scalar
 */
void tensor_add_scalar(Tensor* out, Tensor* a, float scalar);

/**
 * 原地标量加法: a += scalar
 */
void tensor_add_scalar_inplace(Tensor* a, float scalar);

// ============ 激活函数 ============

/**
 * ReLU 激活: out = max(0, in)
 */
void relu(Tensor* out, Tensor* in);

/**
 * 原地 ReLU
 */
void relu_inplace(Tensor* t);

/**
 * GELU 激活 (近似): out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void gelu(Tensor* out, Tensor* in);

/**
 * 原地 GELU
 */
void gelu_inplace(Tensor* t);

/**
 * Sigmoid 激活: out = 1 / (1 + exp(-in))
 */
void sigmoid(Tensor* out, Tensor* in);

/**
 * 原地 Sigmoid
 */
void sigmoid_inplace(Tensor* t);

/**
 * Tanh 激活
 */
void tanh_activation(Tensor* out, Tensor* in);

/**
 * Softmax: out[i] = exp(in[i]) / sum(exp(in))
 * 对最后一个维度进行 softmax
 * @param out 输出张量
 * @param in 输入张量
 */
void softmax(Tensor* out, Tensor* in);

/**
 * 原地 Softmax
 */
void softmax_inplace(Tensor* t);

// ============ 归约运算 ============

/**
 * 求和
 */
float tensor_sum(Tensor* t);

/**
 * 求均值
 */
float tensor_mean(Tensor* t);

/**
 * 求最大值
 */
float tensor_max(Tensor* t);

/**
 * 求最小值
 */
float tensor_min(Tensor* t);

/**
 * 求最大值索引
 */
int tensor_argmax(Tensor* t);

/**
 * 求最小值索引
 */
int tensor_argmin(Tensor* t);

/**
 * 求方差
 */
float tensor_var(Tensor* t);

/**
 * 求标准差
 */
float tensor_std(Tensor* t);

// ============ 按轴归约 ============

/**
 * 按轴求和
 * @param t 输入张量
 * @param axis 求和的轴
 * @return 结果张量 (去掉 axis 维度)
 */
Tensor* tensor_sum_axis(Tensor* t, int axis);

/**
 * 按轴求均值
 */
Tensor* tensor_mean_axis(Tensor* t, int axis);

/**
 * 按轴求最大值
 */
Tensor* tensor_max_axis(Tensor* t, int axis);

/**
 * 按轴求 argmax
 */
Tensor* tensor_argmax_axis(Tensor* t, int axis);

// ============ 其他数学运算 ============

/**
 * 逐元素取指数: out = exp(in)
 */
void tensor_exp(Tensor* out, Tensor* in);

/**
 * 逐元素取对数: out = log(in)
 */
void tensor_log(Tensor* out, Tensor* in);

/**
 * 逐元素平方: out = in^2
 */
void tensor_square(Tensor* out, Tensor* in);

/**
 * 逐元素开方: out = sqrt(in)
 */
void tensor_sqrt(Tensor* out, Tensor* in);

/**
 * 逐元素取绝对值
 */
void tensor_abs(Tensor* out, Tensor* in);

/**
 * 逐元素取负
 */
void tensor_neg(Tensor* out, Tensor* in);

/**
 * 裁剪值到指定范围
 */
void tensor_clip(Tensor* out, Tensor* in, float min_val, float max_val);

#endif // MATH_OPS_H
