#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
  float* data;
  int* shape;
  int* strides;
  int ndim;
  int size;
} Tensor;

Tensor* tensor_create(int ndim, int* shape);

Tensor* tensor_zeros(int ndim, int* shape);

Tensor* tensor_ones(int ndim, int* shape);

Tensor* tensor_full(int ndim, int* shape, float value);

Tensor* tensor_randn(int ndim, int* shape, float mean, float std);

Tensor* tensor_rand(int ndim, int* shape);

void tensor_free(Tensor* t);

Tensor* tensor_copy(Tensor* src);

Tensor* tensor_reshape(Tensor* t, int new_ndim, int* new_shape);

Tensor* tensor_reshape_copy(Tensor* t, int new_ndim, int* new_shape);

float tensor_get(Tensor* t, int* indices);

void tensor_set(Tensor* t, int* indices, float value);

float tensor_get_flat(Tensor* t, int index);

void tensor_set_flat(Tensor* t, int index, float value);

void tensor_print(Tensor* t);

void tensor_print_shape(Tensor* t);

int tensor_shape_equal(Tensor* a, Tensor* b);

void tensor_fill_data(Tensor* t, float* data);

Tensor* tensor_create_1d(int size);

Tensor* tensor_create_2d(int rows, int cols);

Tensor* tensor_create_3d(int d0, int d1, int d2);

#endif
