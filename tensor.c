#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 随机数生成器是否已初始化
static int rand_initialized = 0;

static void ensure_rand_initialized(void) {
    if (!rand_initialized) {
        srand((unsigned int)time(NULL));
        rand_initialized = 1;
    }
}

// Box-Muller 变换生成正态分布随机数
static float randn(void) {
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u, v, s;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = 1;
    return u * s;
}

// 计算步长 (strides)
static void compute_strides(int ndim, int* shape, int* strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// 计算总元素数
static int compute_size(int ndim, int* shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

Tensor* tensor_create(int ndim, int* shape) {
    if (ndim <= 0 || shape == NULL) {
        return NULL;
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) return NULL;

    t->ndim = ndim;
    t->size = compute_size(ndim, shape);

    // 分配形状数组
    t->shape = (int*)malloc(ndim * sizeof(int));
    if (t->shape == NULL) {
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    // 分配步长数组
    t->strides = (int*)malloc(ndim * sizeof(int));
    if (t->strides == NULL) {
        free(t->shape);
        free(t);
        return NULL;
    }
    compute_strides(ndim, shape, t->strides);

    // 分配数据
    t->data = (float*)malloc(t->size * sizeof(float));
    if (t->data == NULL) {
        free(t->strides);
        free(t->shape);
        free(t);
        return NULL;
    }

    return t;
}

Tensor* tensor_zeros(int ndim, int* shape) {
    Tensor* t = tensor_create(ndim, shape);
    if (t == NULL) return NULL;
    memset(t->data, 0, t->size * sizeof(float));
    return t;
}

Tensor* tensor_ones(int ndim, int* shape) {
    Tensor* t = tensor_create(ndim, shape);
    if (t == NULL) return NULL;
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f;
    }
    return t;
}

Tensor* tensor_full(int ndim, int* shape, float value) {
    Tensor* t = tensor_create(ndim, shape);
    if (t == NULL) return NULL;
    for (int i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
    return t;
}

Tensor* tensor_randn(int ndim, int* shape, float mean, float std) {
    ensure_rand_initialized();
    Tensor* t = tensor_create(ndim, shape);
    if (t == NULL) return NULL;
    for (int i = 0; i < t->size; i++) {
        t->data[i] = randn() * std + mean;
    }
    return t;
}

Tensor* tensor_rand(int ndim, int* shape) {
    ensure_rand_initialized();
    Tensor* t = tensor_create(ndim, shape);
    if (t == NULL) return NULL;
    for (int i = 0; i < t->size; i++) {
        t->data[i] = (float)rand() / RAND_MAX;
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (t == NULL) return;
    if (t->data != NULL) free(t->data);
    if (t->shape != NULL) free(t->shape);
    if (t->strides != NULL) free(t->strides);
    free(t);
}

Tensor* tensor_copy(Tensor* src) {
    if (src == NULL) return NULL;

    Tensor* dst = tensor_create(src->ndim, src->shape);
    if (dst == NULL) return NULL;
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}

Tensor* tensor_reshape(Tensor* t, int new_ndim, int* new_shape) {
    if (t == NULL || new_ndim <= 0 || new_shape == NULL) {
        return NULL;
    }

    int new_size = compute_size(new_ndim, new_shape);
    if (new_size != t->size) {
        fprintf(stderr, "Error: reshape size mismatch (%d vs %d)\n", t->size, new_size);
        return NULL;
    }

    Tensor* reshaped = (Tensor*)malloc(sizeof(Tensor));
    if (reshaped == NULL) return NULL;

    reshaped->data = t->data;  // 共享数据
    reshaped->ndim = new_ndim;
    reshaped->size = new_size;

    reshaped->shape = (int*)malloc(new_ndim * sizeof(int));
    reshaped->strides = (int*)malloc(new_ndim * sizeof(int));
    if (reshaped->shape == NULL || reshaped->strides == NULL) {
        if (reshaped->shape) free(reshaped->shape);
        if (reshaped->strides) free(reshaped->strides);
        free(reshaped);
        return NULL;
    }

    memcpy(reshaped->shape, new_shape, new_ndim * sizeof(int));
    compute_strides(new_ndim, new_shape, reshaped->strides);

    return reshaped;
}

Tensor* tensor_reshape_copy(Tensor* t, int new_ndim, int* new_shape) {
    Tensor* copy = tensor_copy(t);
    if (copy == NULL) return NULL;

    Tensor* reshaped = tensor_reshape(copy, new_ndim, new_shape);
    if (reshaped == NULL) {
        tensor_free(copy);
        return NULL;
    }

    // 释放 copy 的 shape 和 strides, 但保留 data
    free(copy->shape);
    free(copy->strides);
    free(copy);

    return reshaped;
}

float tensor_get(Tensor* t, int* indices) {
    int flat_idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        flat_idx += indices[i] * t->strides[i];
    }
    return t->data[flat_idx];
}

void tensor_set(Tensor* t, int* indices, float value) {
    int flat_idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        flat_idx += indices[i] * t->strides[i];
    }
    t->data[flat_idx] = value;
}

float tensor_get_flat(Tensor* t, int index) {
    return t->data[index];
}

void tensor_set_flat(Tensor* t, int index, float value) {
    t->data[index] = value;
}

void tensor_print_shape(Tensor* t) {
    if (t == NULL) {
        printf("Tensor: NULL\n");
        return;
    }
    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], size=%d)\n", t->size);
}

void tensor_print(Tensor* t) {
    if (t == NULL) {
        printf("Tensor: NULL\n");
        return;
    }

    tensor_print_shape(t);

    if (t->ndim == 1) {
        // 1D 张量
        printf("[");
        for (int i = 0; i < t->shape[0]; i++) {
            printf("%8.4f", t->data[i]);
            if (i < t->shape[0] - 1) printf(", ");
        }
        printf("]\n");
    } else if (t->ndim == 2) {
        // 2D 张量 (矩阵)
        printf("[\n");
        for (int i = 0; i < t->shape[0]; i++) {
            printf("  [");
            for (int j = 0; j < t->shape[1]; j++) {
                int idx = i * t->strides[0] + j * t->strides[1];
                printf("%8.4f", t->data[idx]);
                if (j < t->shape[1] - 1) printf(", ");
            }
            printf("]");
            if (i < t->shape[0] - 1) printf(",");
            printf("\n");
        }
        printf("]\n");
    } else {
        // 高维张量: 只打印前几个元素
        printf("data=[");
        int print_count = (t->size < 10) ? t->size : 10;
        for (int i = 0; i < print_count; i++) {
            printf("%8.4f", t->data[i]);
            if (i < print_count - 1) printf(", ");
        }
        if (t->size > 10) printf(", ...");
        printf("]\n");
    }
}

int tensor_shape_equal(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) return 0;
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

void tensor_fill_data(Tensor* t, float* data) {
    memcpy(t->data, data, t->size * sizeof(float));
}

Tensor* tensor_create_1d(int size) {
    int shape[] = {size};
    return tensor_create(1, shape);
}

Tensor* tensor_create_2d(int rows, int cols) {
    int shape[] = {rows, cols};
    return tensor_create(2, shape);
}

Tensor* tensor_create_3d(int d0, int d1, int d2) {
    int shape[] = {d0, d1, d2};
    return tensor_create(3, shape);
}
