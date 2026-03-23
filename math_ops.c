#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============ 矩阵运算 ============

Tensor* matmul(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "Error: matmul requires 2D tensors\n");
        return NULL;
    }
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error: matmul shape mismatch [%d,%d] @ [%d,%d]\n",
                a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        return NULL;
    }

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    int shape[] = {M, N};
    Tensor* c = tensor_zeros(2, shape);
    if (c == NULL) return NULL;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            c->data[i * N + j] = sum;
        }
    }

    return c;
}

Tensor* matmul_transposed_b(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "Error: matmul_transposed_b requires 2D tensors\n");
        return NULL;
    }
    if (a->shape[1] != b->shape[1]) {
        fprintf(stderr, "Error: matmul_transposed_b shape mismatch\n");
        return NULL;
    }

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[0];

    int shape[] = {M, N};
    Tensor* c = tensor_zeros(2, shape);
    if (c == NULL) return NULL;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[j * K + k];
            }
            c->data[i * N + j] = sum;
        }
    }

    return c;
}

void matmul_inplace(Tensor* out, Tensor* a, Tensor* b) {
    if (out == NULL || a == NULL || b == NULL) return;

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    memset(out->data, 0, out->size * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }
}

void matvec(Tensor* out, Tensor* mat, Tensor* vec) {
    if (out == NULL || mat == NULL || vec == NULL) return;

    int M = mat->shape[0];
    int N = mat->shape[1];

    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += mat->data[i * N + j] * vec->data[j];
        }
        out->data[i] = sum;
    }
}

Tensor* batched_matmul(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (a->ndim != 3 || b->ndim != 3) {
        fprintf(stderr, "Error: batched_matmul requires 3D tensors\n");
        return NULL;
    }
    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) {
        fprintf(stderr, "Error: batched_matmul shape mismatch\n");
        return NULL;
    }

    int batch = a->shape[0];
    int M = a->shape[1];
    int K = a->shape[2];
    int N = b->shape[2];

    int shape[] = {batch, M, N};
    Tensor* c = tensor_zeros(3, shape);
    if (c == NULL) return NULL;

    for (int bs = 0; bs < batch; bs++) {
        int a_offset = bs * M * K;
        int b_offset = bs * K * N;
        int c_offset = bs * M * N;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a->data[a_offset + i * K + k] * b->data[b_offset + k * N + j];
                }
                c->data[c_offset + i * N + j] = sum;
            }
        }
    }

    return c;
}

// ============ 逐元素运算 ============

void tensor_add(Tensor* out, Tensor* a, Tensor* b) {
    if (out == NULL || a == NULL || b == NULL) return;
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_add_inplace(Tensor* a, Tensor* b) {
    if (a == NULL || b == NULL) return;
    for (int i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}

void tensor_sub(Tensor* out, Tensor* a, Tensor* b) {
    if (out == NULL || a == NULL || b == NULL) return;
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void tensor_mul(Tensor* out, Tensor* a, Tensor* b) {
    if (out == NULL || a == NULL || b == NULL) return;
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_mul_scalar(Tensor* out, Tensor* a, float scalar) {
    if (out == NULL || a == NULL) return;
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * scalar;
    }
}

void tensor_mul_scalar_inplace(Tensor* a, float scalar) {
    if (a == NULL) return;
    for (int i = 0; i < a->size; i++) {
        a->data[i] *= scalar;
    }
}

void tensor_add_scalar(Tensor* out, Tensor* a, float scalar) {
    if (out == NULL || a == NULL) return;
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + scalar;
    }
}

void tensor_add_scalar_inplace(Tensor* a, float scalar) {
    if (a == NULL) return;
    for (int i = 0; i < a->size; i++) {
        a->data[i] += scalar;
    }
}

// ============ 激活函数 ============

void relu(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = (in->data[i] > 0) ? in->data[i] : 0.0f;
    }
}

void relu_inplace(Tensor* t) {
    if (t == NULL) return;
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < 0) t->data[i] = 0.0f;
    }
}

void gelu(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
    const float coeff = 0.044715f;

    for (int i = 0; i < in->size; i++) {
        float x = in->data[i];
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void gelu_inplace(Tensor* t) {
    if (t == NULL) return;
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int i = 0; i < t->size; i++) {
        float x = t->data[i];
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        t->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void sigmoid(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-in->data[i]));
    }
}

void sigmoid_inplace(Tensor* t) {
    if (t == NULL) return;
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
}

void tanh_activation(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = tanhf(in->data[i]);
    }
}

void softmax(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;

    if (in->ndim == 1) {
        // 1D: 对整个向量做 softmax
        float max_val = tensor_max(in);
        float sum = 0.0f;
        for (int i = 0; i < in->size; i++) {
            out->data[i] = expf(in->data[i] - max_val);
            sum += out->data[i];
        }
        for (int i = 0; i < in->size; i++) {
            out->data[i] /= sum;
        }
    } else if (in->ndim == 2) {
        // 2D: 对每行做 softmax
        int rows = in->shape[0];
        int cols = in->shape[1];
        for (int i = 0; i < rows; i++) {
            // 找最大值 (数值稳定性)
            float max_val = -FLT_MAX;
            for (int j = 0; j < cols; j++) {
                if (in->data[i * cols + j] > max_val) {
                    max_val = in->data[i * cols + j];
                }
            }
            // 计算 exp 和 sum
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                out->data[i * cols + j] = expf(in->data[i * cols + j] - max_val);
                sum += out->data[i * cols + j];
            }
            // 归一化
            for (int j = 0; j < cols; j++) {
                out->data[i * cols + j] /= sum;
            }
        }
    } else {
        // 高维: 对最后一个维度做 softmax
        int last_dim = in->shape[in->ndim - 1];
        int num_vectors = in->size / last_dim;

        for (int v = 0; v < num_vectors; v++) {
            int offset = v * last_dim;
            float max_val = -FLT_MAX;
            for (int i = 0; i < last_dim; i++) {
                if (in->data[offset + i] > max_val) {
                    max_val = in->data[offset + i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < last_dim; i++) {
                out->data[offset + i] = expf(in->data[offset + i] - max_val);
                sum += out->data[offset + i];
            }
            for (int i = 0; i < last_dim; i++) {
                out->data[offset + i] /= sum;
            }
        }
    }
}

void softmax_inplace(Tensor* t) {
    softmax(t, t);
}

// ============ 归约运算 ============

float tensor_sum(Tensor* t) {
    if (t == NULL) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    return sum;
}

float tensor_mean(Tensor* t) {
    if (t == NULL || t->size == 0) return 0.0f;
    return tensor_sum(t) / t->size;
}

float tensor_max(Tensor* t) {
    if (t == NULL || t->size == 0) return -FLT_MAX;
    float max_val = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] > max_val) max_val = t->data[i];
    }
    return max_val;
}

float tensor_min(Tensor* t) {
    if (t == NULL || t->size == 0) return FLT_MAX;
    float min_val = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] < min_val) min_val = t->data[i];
    }
    return min_val;
}

int tensor_argmax(Tensor* t) {
    if (t == NULL || t->size == 0) return -1;
    int max_idx = 0;
    float max_val = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] > max_val) {
            max_val = t->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int tensor_argmin(Tensor* t) {
    if (t == NULL || t->size == 0) return -1;
    int min_idx = 0;
    float min_val = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] < min_val) {
            min_val = t->data[i];
            min_idx = i;
        }
    }
    return min_idx;
}

float tensor_var(Tensor* t) {
    if (t == NULL || t->size <= 1) return 0.0f;
    float mean = tensor_mean(t);
    float var = 0.0f;
    for (int i = 0; i < t->size; i++) {
        float diff = t->data[i] - mean;
        var += diff * diff;
    }
    return var / t->size;
}

float tensor_std(Tensor* t) {
    return sqrtf(tensor_var(t));
}

// ============ 按轴归约 ============

Tensor* tensor_sum_axis(Tensor* t, int axis) {
    if (t == NULL || axis < 0 || axis >= t->ndim) return NULL;

    // 计算输出形状
    int new_ndim = t->ndim - 1;
    if (new_ndim == 0) {
        // 结果是标量, 返回 1D 张量
        int shape[] = {1};
        Tensor* result = tensor_zeros(1, shape);
        result->data[0] = tensor_sum(t);
        return result;
    }

    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    int idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[idx++] = t->shape[i];
    }

    Tensor* result = tensor_zeros(new_ndim, new_shape);
    free(new_shape);
    if (result == NULL) return NULL;

    // 计算归约
    int axis_size = t->shape[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= t->shape[i];
    int inner_size = 1;
    for (int i = axis + 1; i < t->ndim; i++) inner_size *= t->shape[i];

    for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
            float sum = 0.0f;
            for (int a = 0; a < axis_size; a++) {
                int idx_t = o * axis_size * inner_size + a * inner_size + i;
                sum += t->data[idx_t];
            }
            result->data[o * inner_size + i] = sum;
        }
    }

    return result;
}

Tensor* tensor_mean_axis(Tensor* t, int axis) {
    Tensor* sum = tensor_sum_axis(t, axis);
    if (sum == NULL) return NULL;
    float axis_size = (float)t->shape[axis];
    for (int i = 0; i < sum->size; i++) {
        sum->data[i] /= axis_size;
    }
    return sum;
}

Tensor* tensor_max_axis(Tensor* t, int axis) {
    if (t == NULL || axis < 0 || axis >= t->ndim) return NULL;

    int new_ndim = t->ndim - 1;
    if (new_ndim == 0) {
        int shape[] = {1};
        Tensor* result = tensor_create(1, shape);
        result->data[0] = tensor_max(t);
        return result;
    }

    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    int idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[idx++] = t->shape[i];
    }

    Tensor* result = tensor_create(new_ndim, new_shape);
    free(new_shape);
    if (result == NULL) return NULL;

    int axis_size = t->shape[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= t->shape[i];
    int inner_size = 1;
    for (int i = axis + 1; i < t->ndim; i++) inner_size *= t->shape[i];

    for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
            float max_val = -FLT_MAX;
            for (int a = 0; a < axis_size; a++) {
                int idx_t = o * axis_size * inner_size + a * inner_size + i;
                if (t->data[idx_t] > max_val) max_val = t->data[idx_t];
            }
            result->data[o * inner_size + i] = max_val;
        }
    }

    return result;
}

Tensor* tensor_argmax_axis(Tensor* t, int axis) {
    if (t == NULL || axis < 0 || axis >= t->ndim) return NULL;

    int new_ndim = t->ndim - 1;
    if (new_ndim == 0) {
        int shape[] = {1};
        Tensor* result = tensor_create(1, shape);
        result->data[0] = (float)tensor_argmax(t);
        return result;
    }

    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    int idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i != axis) new_shape[idx++] = t->shape[i];
    }

    Tensor* result = tensor_create(new_ndim, new_shape);
    free(new_shape);
    if (result == NULL) return NULL;

    int axis_size = t->shape[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= t->shape[i];
    int inner_size = 1;
    for (int i = axis + 1; i < t->ndim; i++) inner_size *= t->shape[i];

    for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
            float max_val = -FLT_MAX;
            int max_idx = 0;
            for (int a = 0; a < axis_size; a++) {
                int idx_t = o * axis_size * inner_size + a * inner_size + i;
                if (t->data[idx_t] > max_val) {
                    max_val = t->data[idx_t];
                    max_idx = a;
                }
            }
            result->data[o * inner_size + i] = (float)max_idx;
        }
    }

    return result;
}

// ============ 其他数学运算 ============

void tensor_exp(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = expf(in->data[i]);
    }
}

void tensor_log(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = logf(in->data[i]);
    }
}

void tensor_square(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = in->data[i] * in->data[i];
    }
}

void tensor_sqrt(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = sqrtf(in->data[i]);
    }
}

void tensor_abs(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = fabsf(in->data[i]);
    }
}

void tensor_neg(Tensor* out, Tensor* in) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        out->data[i] = -in->data[i];
    }
}

void tensor_clip(Tensor* out, Tensor* in, float min_val, float max_val) {
    if (out == NULL || in == NULL) return;
    for (int i = 0; i < in->size; i++) {
        float v = in->data[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        out->data[i] = v;
    }
}
