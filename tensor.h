#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/**
 * Tensor - 多维数组数据结构
 *
 * 数据按行优先(row-major)顺序存储
 * 例如: shape=[2,3] 的张量存储顺序为:
 *   [0,0], [0,1], [0,2], [1,0], [1,1], [1,2]
 */
typedef struct {
    float* data;       // 数据指针
    int* shape;        // 形状数组, 长度为 ndim
    int* strides;      // 步长数组, 用于索引计算
    int ndim;          // 维度数
    int size;          // 总元素数
} Tensor;

// ============ 创建和销毁 ============

/**
 * 创建未初始化的张量
 * @param ndim 维度数
 * @param shape 形状数组
 * @return 新创建的张量, 失败返回 NULL
 */
Tensor* tensor_create(int ndim, int* shape);

/**
 * 创建全零张量
 */
Tensor* tensor_zeros(int ndim, int* shape);

/**
 * 创建全一张量
 */
Tensor* tensor_ones(int ndim, int* shape);

/**
 * 创建填充指定值的张量
 */
Tensor* tensor_full(int ndim, int* shape, float value);

/**
 * 创建随机张量 (正态分布)
 * @param mean 均值
 * @param std 标准差
 */
Tensor* tensor_randn(int ndim, int* shape, float mean, float std);

/**
 * 创建随机张量 (均匀分布 [0, 1))
 */
Tensor* tensor_rand(int ndim, int* shape);

/**
 * 释放张量内存
 */
void tensor_free(Tensor* t);

/**
 * 深拷贝张量
 */
Tensor* tensor_copy(Tensor* src);

// ============ 形状操作 ============

/**
 * 重塑张量 (不复制数据, 要求总元素数相同)
 * @param t 输入张量
 * @param new_ndim 新维度数
 * @param new_shape 新形状
 * @return 重塑后的张量 (共享数据), 失败返回 NULL
 */
Tensor* tensor_reshape(Tensor* t, int new_ndim, int* new_shape);

/**
 * 创建新张量并重塑 (复制数据)
 */
Tensor* tensor_reshape_copy(Tensor* t, int new_ndim, int* new_shape);

// ============ 索引和赋值 ============

/**
 * 获取元素值
 * @param t 张量
 * @param indices 索引数组, 长度必须等于 ndim
 */
float tensor_get(Tensor* t, int* indices);

/**
 * 设置元素值
 */
void tensor_set(Tensor* t, int* indices, float value);

/**
 * 获取一维索引对应的元素 (用于遍历)
 */
float tensor_get_flat(Tensor* t, int index);

/**
 * 设置一维索引对应的元素
 */
void tensor_set_flat(Tensor* t, int index, float value);

// ============ 工具函数 ============

/**
 * 打印张量信息和数据
 */
void tensor_print(Tensor* t);

/**
 * 仅打印张量形状信息
 */
void tensor_print_shape(Tensor* t);

/**
 * 检查两个张量形状是否相同
 */
int tensor_shape_equal(Tensor* a, Tensor* b);

/**
 * 用数据填充张量
 * @param t 张量
 * @param data 数据数组, 长度必须等于 t->size
 */
void tensor_fill_data(Tensor* t, float* data);

// ============ 便捷创建函数 ============

/**
 * 创建 1D 张量
 */
Tensor* tensor_create_1d(int size);

/**
 * 创建 2D 张量 (矩阵)
 */
Tensor* tensor_create_2d(int rows, int cols);

/**
 * 创建 3D 张量
 */
Tensor* tensor_create_3d(int d0, int d1, int d2);

#endif // TENSOR_H
