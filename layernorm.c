#include "layernorm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

LayerNorm* layernorm_create(int hidden_dim, float eps) {
    if (hidden_dim <= 0) return NULL;

    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    if (ln == NULL) return NULL;

    ln->hidden_dim = hidden_dim;
    ln->eps = eps;

    int shape[] = {hidden_dim};
    ln->gamma = tensor_ones(1, shape);
    ln->beta = tensor_zeros(1, shape);

    if (ln->gamma == NULL || ln->beta == NULL) {
        layernorm_free(ln);
        return NULL;
    }

    return ln;
}

void layernorm_free(LayerNorm* ln) {
    if (ln == NULL) return;

    if (ln->gamma) tensor_free(ln->gamma);
    if (ln->beta) tensor_free(ln->beta);
    free(ln);
}

void layernorm_init(LayerNorm* ln) {
    if (ln == NULL) return;

    // gamma = 1, beta = 0
    for (int i = 0; i < ln->hidden_dim; i++) {
        ln->gamma->data[i] = 1.0f;
        ln->beta->data[i] = 0.0f;
    }
}

void layernorm_forward(LayerNorm* ln, Tensor* input, Tensor* output) {
    if (ln == NULL || input == NULL || output == NULL) return;

    int hidden_dim = ln->hidden_dim;

    if (input->ndim == 1) {
        // 1D 输入: [hidden_dim]
        // 计算均值
        float mean = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            mean += input->data[i];
        }
        mean /= hidden_dim;

        // 计算方差
        float var = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float diff = input->data[i] - mean;
            var += diff * diff;
        }
        var /= hidden_dim;

        // 归一化
        float inv_std = 1.0f / sqrtf(var + ln->eps);
        for (int i = 0; i < hidden_dim; i++) {
            float normalized = (input->data[i] - mean) * inv_std;
            output->data[i] = ln->gamma->data[i] * normalized + ln->beta->data[i];
        }
    } else if (input->ndim == 2) {
        // 2D 输入: [seq_len, hidden_dim]
        int seq_len = input->shape[0];

        for (int s = 0; s < seq_len; s++) {
            int offset = s * hidden_dim;

            // 计算均值
            float mean = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                mean += input->data[offset + i];
            }
            mean /= hidden_dim;

            // 计算方差
            float var = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                float diff = input->data[offset + i] - mean;
                var += diff * diff;
            }
            var /= hidden_dim;

            // 归一化
            float inv_std = 1.0f / sqrtf(var + ln->eps);
            for (int i = 0; i < hidden_dim; i++) {
                float normalized = (input->data[offset + i] - mean) * inv_std;
                output->data[offset + i] = ln->gamma->data[i] * normalized + ln->beta->data[i];
            }
        }
    } else {
        fprintf(stderr, "Error: LayerNorm only supports 1D or 2D input\n");
    }
}

void layernorm_print_info(LayerNorm* ln) {
    if (ln == NULL) {
        printf("LayerNorm: NULL\n");
        return;
    }

    printf("LayerNorm Info:\n");
    printf("  hidden_dim: %d\n", ln->hidden_dim);
    printf("  eps: %.2e\n", ln->eps);
    printf("  gamma: [%d]\n", ln->hidden_dim);
    printf("  beta: [%d]\n", ln->hidden_dim);
    printf("  total params: %d\n", layernorm_num_params(ln));
}

int layernorm_num_params(LayerNorm* ln) {
    if (ln == NULL) return 0;
    return 2 * ln->hidden_dim;  // gamma + beta
}
