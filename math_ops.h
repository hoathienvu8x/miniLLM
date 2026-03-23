#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"

Tensor* matmul(Tensor* a, Tensor* b);

Tensor* matmul_transposed_b(Tensor* a, Tensor* b);

void matmul_inplace(Tensor* out, Tensor* a, Tensor* b);

void matvec(Tensor* out, Tensor* mat, Tensor* vec);

Tensor* batched_matmul(Tensor* a, Tensor* b);

void tensor_add(Tensor* out, Tensor* a, Tensor* b);

void tensor_add_inplace(Tensor* a, Tensor* b);

void tensor_sub(Tensor* out, Tensor* a, Tensor* b);

void tensor_mul(Tensor* out, Tensor* a, Tensor* b);

void tensor_mul_scalar(Tensor* out, Tensor* a, float scalar);

void tensor_mul_scalar_inplace(Tensor* a, float scalar);

void tensor_add_scalar(Tensor* out, Tensor* a, float scalar);

void tensor_add_scalar_inplace(Tensor* a, float scalar);

void relu(Tensor* out, Tensor* in);

void relu_inplace(Tensor* t);

void gelu(Tensor* out, Tensor* in);

void gelu_inplace(Tensor* t);

void sigmoid(Tensor* out, Tensor* in);

void sigmoid_inplace(Tensor* t);

void tanh_activation(Tensor* out, Tensor* in);

void softmax(Tensor* out, Tensor* in);

void softmax_inplace(Tensor* t);

float tensor_sum(Tensor* t);

float tensor_mean(Tensor* t);

float tensor_max(Tensor* t);

float tensor_min(Tensor* t);

int tensor_argmax(Tensor* t);

int tensor_argmin(Tensor* t);

float tensor_var(Tensor* t);

float tensor_std(Tensor* t);

Tensor* tensor_sum_axis(Tensor* t, int axis);

Tensor* tensor_mean_axis(Tensor* t, int axis);

Tensor* tensor_max_axis(Tensor* t, int axis);

Tensor* tensor_argmax_axis(Tensor* t, int axis);

void tensor_exp(Tensor* out, Tensor* in);

void tensor_log(Tensor* out, Tensor* in);

void tensor_square(Tensor* out, Tensor* in);

void tensor_sqrt(Tensor* out, Tensor* in);

void tensor_abs(Tensor* out, Tensor* in);

void tensor_neg(Tensor* out, Tensor* in);

void tensor_clip(Tensor* out, Tensor* in, float min_val, float max_val);

#endif
