#include "backward.h"
#include "math_ops.h"
#include "attention.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Gradients* gradients_create(GPTModel* model) {
  if (model == NULL) return NULL;

  Gradients* grads = (Gradients*)malloc(sizeof(Gradients));
  if (grads == NULL) return NULL;

  ModelConfig* cfg = &model->config;
  grads->num_layers = cfg->num_layers;

  int token_shape[] = {cfg->vocab_size, cfg->hidden_dim};
  int pos_shape[] = {cfg->max_seq_len, cfg->hidden_dim};
  grads->d_token_embedding = tensor_zeros(2, token_shape);
  grads->d_position_embedding = tensor_zeros(2, pos_shape);

  grads->layers = (void*)malloc(cfg->num_layers * sizeof(*grads->layers));
  int hidden_shape[] = {cfg->hidden_dim};
  int wq_shape[] = {cfg->hidden_dim, cfg->hidden_dim};
  int w1_shape[] = {cfg->hidden_dim, cfg->ffn_dim};
  int w2_shape[] = {cfg->ffn_dim, cfg->hidden_dim};
  int ffn_shape[] = {cfg->ffn_dim};

  for (int i = 0; i < cfg->num_layers; i++) {
    grads->layers[i].d_ln1_gamma = tensor_zeros(1, hidden_shape);
    grads->layers[i].d_ln1_beta = tensor_zeros(1, hidden_shape);
    grads->layers[i].d_W_q = tensor_zeros(2, wq_shape);
    grads->layers[i].d_W_k = tensor_zeros(2, wq_shape);
    grads->layers[i].d_W_v = tensor_zeros(2, wq_shape);
    grads->layers[i].d_W_o = tensor_zeros(2, wq_shape);
    grads->layers[i].d_ln2_gamma = tensor_zeros(1, hidden_shape);
    grads->layers[i].d_ln2_beta = tensor_zeros(1, hidden_shape);
    grads->layers[i].d_W1 = tensor_zeros(2, w1_shape);
    grads->layers[i].d_b1 = tensor_zeros(1, ffn_shape);
    grads->layers[i].d_W2 = tensor_zeros(2, w2_shape);
    grads->layers[i].d_b2 = tensor_zeros(1, hidden_shape);
  }

  grads->d_final_ln_gamma = tensor_zeros(1, hidden_shape);
  grads->d_final_ln_beta = tensor_zeros(1, hidden_shape);

  int lm_shape[] = {cfg->hidden_dim, cfg->vocab_size};
  grads->d_lm_head = tensor_zeros(2, lm_shape);

  return grads;
}

void gradients_free(Gradients* grads) {
  if (grads == NULL) return;

  tensor_free(grads->d_token_embedding);
  tensor_free(grads->d_position_embedding);

  for (int i = 0; i < grads->num_layers; i++) {
    tensor_free(grads->layers[i].d_ln1_gamma);
    tensor_free(grads->layers[i].d_ln1_beta);
    tensor_free(grads->layers[i].d_W_q);
    tensor_free(grads->layers[i].d_W_k);
    tensor_free(grads->layers[i].d_W_v);
    tensor_free(grads->layers[i].d_W_o);
    tensor_free(grads->layers[i].d_ln2_gamma);
    tensor_free(grads->layers[i].d_ln2_beta);
    tensor_free(grads->layers[i].d_W1);
    tensor_free(grads->layers[i].d_b1);
    tensor_free(grads->layers[i].d_W2);
    tensor_free(grads->layers[i].d_b2);
  }
  free(grads->layers);

  tensor_free(grads->d_final_ln_gamma);
  tensor_free(grads->d_final_ln_beta);
  tensor_free(grads->d_lm_head);

  free(grads);
}

void gradients_zero(Gradients* grads) {
  if (grads == NULL) return;

  memset(grads->d_token_embedding->data, 0,
       grads->d_token_embedding->size * sizeof(float));
  memset(grads->d_position_embedding->data, 0,
       grads->d_position_embedding->size * sizeof(float));

  for (int i = 0; i < grads->num_layers; i++) {
    memset(grads->layers[i].d_ln1_gamma->data, 0,
         grads->layers[i].d_ln1_gamma->size * sizeof(float));
    memset(grads->layers[i].d_ln1_beta->data, 0,
         grads->layers[i].d_ln1_beta->size * sizeof(float));
    memset(grads->layers[i].d_W_q->data, 0,
         grads->layers[i].d_W_q->size * sizeof(float));
    memset(grads->layers[i].d_W_k->data, 0,
         grads->layers[i].d_W_k->size * sizeof(float));
    memset(grads->layers[i].d_W_v->data, 0,
         grads->layers[i].d_W_v->size * sizeof(float));
    memset(grads->layers[i].d_W_o->data, 0,
         grads->layers[i].d_W_o->size * sizeof(float));
    memset(grads->layers[i].d_ln2_gamma->data, 0,
         grads->layers[i].d_ln2_gamma->size * sizeof(float));
    memset(grads->layers[i].d_ln2_beta->data, 0,
         grads->layers[i].d_ln2_beta->size * sizeof(float));
    memset(grads->layers[i].d_W1->data, 0,
         grads->layers[i].d_W1->size * sizeof(float));
    memset(grads->layers[i].d_b1->data, 0,
         grads->layers[i].d_b1->size * sizeof(float));
    memset(grads->layers[i].d_W2->data, 0,
         grads->layers[i].d_W2->size * sizeof(float));
    memset(grads->layers[i].d_b2->data, 0,
         grads->layers[i].d_b2->size * sizeof(float));
  }

  memset(grads->d_final_ln_gamma->data, 0,
       grads->d_final_ln_gamma->size * sizeof(float));
  memset(grads->d_final_ln_beta->data, 0,
       grads->d_final_ln_beta->size * sizeof(float));
  memset(grads->d_lm_head->data, 0,
       grads->d_lm_head->size * sizeof(float));
}

float gradients_norm(Gradients* grads) {
  if (grads == NULL) return 0.0f;

  float sum_sq = 0.0f;

  for (int i = 0; i < grads->d_token_embedding->size; i++) {
    sum_sq += grads->d_token_embedding->data[i] * grads->d_token_embedding->data[i];
  }
  for (int i = 0; i < grads->d_position_embedding->size; i++) {
    sum_sq += grads->d_position_embedding->data[i] * grads->d_position_embedding->data[i];
  }

  for (int l = 0; l < grads->num_layers; l++) {
    for (int i = 0; i < grads->layers[l].d_W_q->size; i++) {
      sum_sq += grads->layers[l].d_W_q->data[i] * grads->layers[l].d_W_q->data[i];
      sum_sq += grads->layers[l].d_W_k->data[i] * grads->layers[l].d_W_k->data[i];
      sum_sq += grads->layers[l].d_W_v->data[i] * grads->layers[l].d_W_v->data[i];
      sum_sq += grads->layers[l].d_W_o->data[i] * grads->layers[l].d_W_o->data[i];
    }
    for (int i = 0; i < grads->layers[l].d_W1->size; i++) {
      sum_sq += grads->layers[l].d_W1->data[i] * grads->layers[l].d_W1->data[i];
    }
    for (int i = 0; i < grads->layers[l].d_W2->size; i++) {
      sum_sq += grads->layers[l].d_W2->data[i] * grads->layers[l].d_W2->data[i];
    }
  }

  for (int i = 0; i < grads->d_lm_head->size; i++) {
    sum_sq += grads->d_lm_head->data[i] * grads->d_lm_head->data[i];
  }

  return sqrtf(sum_sq);
}

float gradients_clip(Gradients* grads, float max_norm) {
  float norm = gradients_norm(grads);
  if (norm > max_norm) {
    float scale = max_norm / norm;

    for (int i = 0; i < grads->d_token_embedding->size; i++) {
      grads->d_token_embedding->data[i] *= scale;
    }
    for (int i = 0; i < grads->d_position_embedding->size; i++) {
      grads->d_position_embedding->data[i] *= scale;
    }

    for (int l = 0; l < grads->num_layers; l++) {
      for (int i = 0; i < grads->layers[l].d_ln1_gamma->size; i++) {
        grads->layers[l].d_ln1_gamma->data[i] *= scale;
        grads->layers[l].d_ln1_beta->data[i] *= scale;
        grads->layers[l].d_ln2_gamma->data[i] *= scale;
        grads->layers[l].d_ln2_beta->data[i] *= scale;
      }
      for (int i = 0; i < grads->layers[l].d_W_q->size; i++) {
        grads->layers[l].d_W_q->data[i] *= scale;
        grads->layers[l].d_W_k->data[i] *= scale;
        grads->layers[l].d_W_v->data[i] *= scale;
        grads->layers[l].d_W_o->data[i] *= scale;
      }
      for (int i = 0; i < grads->layers[l].d_W1->size; i++) {
        grads->layers[l].d_W1->data[i] *= scale;
      }
      for (int i = 0; i < grads->layers[l].d_b1->size; i++) {
        grads->layers[l].d_b1->data[i] *= scale;
      }
      for (int i = 0; i < grads->layers[l].d_W2->size; i++) {
        grads->layers[l].d_W2->data[i] *= scale;
      }
      for (int i = 0; i < grads->layers[l].d_b2->size; i++) {
        grads->layers[l].d_b2->data[i] *= scale;
      }
    }

    for (int i = 0; i < grads->d_final_ln_gamma->size; i++) {
      grads->d_final_ln_gamma->data[i] *= scale;
      grads->d_final_ln_beta->data[i] *= scale;
    }
    for (int i = 0; i < grads->d_lm_head->size; i++) {
      grads->d_lm_head->data[i] *= scale;
    }
  }
  return norm;
}

BackwardCache* backward_cache_create(GPTModel* model, int seq_len) {
  if (model == NULL) return NULL;

  BackwardCache* cache = (BackwardCache*)malloc(sizeof(BackwardCache));
  if (cache == NULL) return NULL;

  ModelConfig* cfg = &model->config;
  cache->num_layers = cfg->num_layers;
  cache->seq_len = seq_len;
  cache->hidden_dim = cfg->hidden_dim;

  int h_shape[] = {seq_len, cfg->hidden_dim};
  int ffn_shape[] = {seq_len, cfg->ffn_dim};

  cache->embed_output = tensor_zeros(2, h_shape);
  cache->layer_inputs = (Tensor**)malloc(cfg->num_layers * sizeof(Tensor*));
  cache->ln1_outputs = (Tensor**)malloc(cfg->num_layers * sizeof(Tensor*));
  cache->attn_outputs = (Tensor**)malloc(cfg->num_layers * sizeof(Tensor*));
  cache->ln2_outputs = (Tensor**)malloc(cfg->num_layers * sizeof(Tensor*));
  cache->ffn_hidden = (Tensor**)malloc(cfg->num_layers * sizeof(Tensor*));

  for (int i = 0; i < cfg->num_layers; i++) {
    cache->layer_inputs[i] = tensor_zeros(2, h_shape);
    cache->ln1_outputs[i] = tensor_zeros(2, h_shape);
    cache->attn_outputs[i] = tensor_zeros(2, h_shape);
    cache->ln2_outputs[i] = tensor_zeros(2, h_shape);
    cache->ffn_hidden[i] = tensor_zeros(2, ffn_shape);
  }

  cache->final_ln_input = tensor_zeros(2, h_shape);
  cache->final_ln_output = tensor_zeros(2, h_shape);

  return cache;
}

void backward_cache_free(BackwardCache* cache) {
  if (cache == NULL) return;

  tensor_free(cache->embed_output);

  for (int i = 0; i < cache->num_layers; i++) {
    tensor_free(cache->layer_inputs[i]);
    tensor_free(cache->ln1_outputs[i]);
    tensor_free(cache->attn_outputs[i]);
    tensor_free(cache->ln2_outputs[i]);
    tensor_free(cache->ffn_hidden[i]);
  }
  free(cache->layer_inputs);
  free(cache->ln1_outputs);
  free(cache->attn_outputs);
  free(cache->ln2_outputs);
  free(cache->ffn_hidden);

  tensor_free(cache->final_ln_input);
  tensor_free(cache->final_ln_output);

  free(cache);
}

void linear_backward(
  Tensor* grad_output,
  Tensor* input,
  Tensor* weight,
  Tensor* grad_weight,
  Tensor* grad_input
) {
  if (grad_output == NULL || input == NULL || weight == NULL) return;

  int seq_len = input->shape[0];
  int in_dim = input->shape[1];
  int out_dim = weight->shape[1];

  if (grad_weight != NULL) {
    for (int i = 0; i < in_dim; i++) {
      for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += input->data[s * in_dim + i] * grad_output->data[s * out_dim + j];
        }
        grad_weight->data[i * out_dim + j] += sum;
      }
    }
  }

  if (grad_input != NULL) {
    for (int s = 0; s < seq_len; s++) {
      for (int i = 0; i < in_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < out_dim; j++) {
          sum += grad_output->data[s * out_dim + j] * weight->data[i * out_dim + j];
        }
        grad_input->data[s * in_dim + i] = sum;
      }
    }
  }
}

void layernorm_backward(
  Tensor* grad_output,
  Tensor* input,
  Tensor* gamma,
  Tensor* grad_gamma,
  Tensor* grad_beta,
  Tensor* grad_input,
  float eps
) {
  if (grad_output == NULL || input == NULL || gamma == NULL) return;

  int seq_len = input->shape[0];
  int hidden_dim = input->shape[1];

  for (int s = 0; s < seq_len; s++) {
    float* x = &input->data[s * hidden_dim];
    float* dy = &grad_output->data[s * hidden_dim];
    float* dx = (grad_input != NULL) ? &grad_input->data[s * hidden_dim] : NULL;

    float mean = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
      mean += x[i];
    }
    mean /= hidden_dim;

    float var = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
      float diff = x[i] - mean;
      var += diff * diff;
    }
    var /= hidden_dim;

    float std_inv = 1.0f / sqrtf(var + eps);

    float x_hat[hidden_dim];
    for (int i = 0; i < hidden_dim; i++) {
      x_hat[i] = (x[i] - mean) * std_inv;
    }

    if (grad_gamma != NULL && grad_beta != NULL) {
      for (int i = 0; i < hidden_dim; i++) {
        grad_gamma->data[i] += dy[i] * x_hat[i];
        grad_beta->data[i] += dy[i];
      }
    }

    if (dx != NULL) {

      float dx_hat[hidden_dim];
      for (int i = 0; i < hidden_dim; i++) {
        dx_hat[i] = dy[i] * gamma->data[i];
      }

      float d_var = 0.0f;
      for (int i = 0; i < hidden_dim; i++) {
        d_var += dx_hat[i] * (x[i] - mean) * (-0.5f) * std_inv * std_inv * std_inv;
      }

      float d_mean = 0.0f;
      float sum_diff = 0.0f;
      for (int i = 0; i < hidden_dim; i++) {
        d_mean += dx_hat[i] * (-std_inv);
        sum_diff += -2.0f * (x[i] - mean);
      }
      d_mean += d_var * sum_diff / hidden_dim;

      for (int i = 0; i < hidden_dim; i++) {
        dx[i] = dx_hat[i] * std_inv +
            d_var * 2.0f * (x[i] - mean) / hidden_dim +
            d_mean / hidden_dim;
      }
    }
  }
}

void gelu_backward(
  Tensor* grad_output,
  Tensor* input,
  Tensor* grad_input
) {
  if (grad_output == NULL || input == NULL || grad_input == NULL) return;

  float sqrt_2_pi = 0.7978845608f;
  float coeff = 0.044715f;

  for (int i = 0; i < input->size; i++) {
    float x = input->data[i];
    float x3 = x * x * x;
    float inner = sqrt_2_pi * (x + coeff * x3);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;

    float gelu_deriv = 0.5f * (1.0f + tanh_val) +
              0.5f * x * sech2 * sqrt_2_pi * (1.0f + 3.0f * coeff * x * x);

    grad_input->data[i] = grad_output->data[i] * gelu_deriv;
  }
}

void softmax_backward(
  Tensor* grad_output,
  Tensor* output,
  Tensor* grad_input
) {
  if (grad_output == NULL || output == NULL || grad_input == NULL) return;

  int rows = output->shape[0];
  int cols = output->shape[1];

  for (int r = 0; r < rows; r++) {
    float* dy = &grad_output->data[r * cols];
    float* y = &output->data[r * cols];
    float* dx = &grad_input->data[r * cols];

    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
      sum += dy[i] * y[i];
    }

    for (int i = 0; i < cols; i++) {
      dx[i] = y[i] * (dy[i] - sum);
    }
  }
}

void embedding_backward(
  Tensor* grad_output,
  int* input_ids,
  int seq_len,
  Tensor* grad_token_emb,
  Tensor* grad_pos_emb
) {
  if (grad_output == NULL || input_ids == NULL) return;

  int hidden_dim = grad_output->shape[1];

  if (grad_token_emb != NULL) {
    for (int s = 0; s < seq_len; s++) {
      int token_id = input_ids[s];
      for (int h = 0; h < hidden_dim; h++) {
        grad_token_emb->data[token_id * hidden_dim + h] +=
          grad_output->data[s * hidden_dim + h];
      }
    }
  }

  if (grad_pos_emb != NULL) {
    for (int s = 0; s < seq_len; s++) {
      for (int h = 0; h < hidden_dim; h++) {
        grad_pos_emb->data[s * hidden_dim + h] +=
          grad_output->data[s * hidden_dim + h];
      }
    }
  }
}

void model_forward_with_cache(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  BackwardCache* cache,
  Tensor* logits
) {
  if (model == NULL || input_ids == NULL || cache == NULL || logits == NULL) return;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;

  embedding_forward(model->embedding, input_ids, seq_len, cache->embed_output);

  memcpy(cache->layer_inputs[0]->data, cache->embed_output->data,
       seq_len * hidden_dim * sizeof(float));

  Tensor* mask = create_causal_mask(seq_len);

  Tensor* current = cache->layer_inputs[0];

  for (int l = 0; l < model->config.num_layers; l++) {
    TransformerBlock* layer = model->layers[l];

    layernorm_forward(layer->ln1, current, cache->ln1_outputs[l]);

    int h_shape[] = {seq_len, hidden_dim};
    Tensor* attn_out = tensor_zeros(2, h_shape);
    attention_forward(layer->attn, cache->ln1_outputs[l], mask, NULL, attn_out);

    for (int i = 0; i < seq_len * hidden_dim; i++) {
      cache->attn_outputs[l]->data[i] = current->data[i] + attn_out->data[i];
    }
    tensor_free(attn_out);

    layernorm_forward(layer->ln2, cache->attn_outputs[l], cache->ln2_outputs[l]);

    for (int s = 0; s < seq_len; s++) {
      for (int f = 0; f < model->config.ffn_dim; f++) {
        float sum = layer->ffn->b1->data[f];
        for (int h = 0; h < hidden_dim; h++) {
          sum += cache->ln2_outputs[l]->data[s * hidden_dim + h] *
               layer->ffn->W1->data[h * model->config.ffn_dim + f];
        }

        cache->ffn_hidden[l]->data[s * model->config.ffn_dim + f] = sum;
      }
    }

    gelu(cache->ffn_hidden[l], cache->ffn_hidden[l]);

    Tensor* ffn_out = tensor_zeros(2, h_shape);
    for (int s = 0; s < seq_len; s++) {
      for (int h = 0; h < hidden_dim; h++) {
        float sum = layer->ffn->b2->data[h];
        for (int f = 0; f < model->config.ffn_dim; f++) {
          sum += cache->ffn_hidden[l]->data[s * model->config.ffn_dim + f] *
               layer->ffn->W2->data[f * hidden_dim + h];
        }
        ffn_out->data[s * hidden_dim + h] = sum;
      }
    }

    if (l < model->config.num_layers - 1) {
      for (int i = 0; i < seq_len * hidden_dim; i++) {
        cache->layer_inputs[l + 1]->data[i] =
          cache->attn_outputs[l]->data[i] + ffn_out->data[i];
      }
      current = cache->layer_inputs[l + 1];
    } else {

      for (int i = 0; i < seq_len * hidden_dim; i++) {
        cache->final_ln_input->data[i] =
          cache->attn_outputs[l]->data[i] + ffn_out->data[i];
      }
    }
    tensor_free(ffn_out);
  }

  tensor_free(mask);

  layernorm_forward(model->final_ln, cache->final_ln_input, cache->final_ln_output);

  for (int s = 0; s < seq_len; s++) {
    for (int v = 0; v < vocab_size; v++) {
      float sum = 0.0f;
      for (int h = 0; h < hidden_dim; h++) {
        sum += cache->final_ln_output->data[s * hidden_dim + h] *
             model->lm_head->data[h * vocab_size + v];
      }
      logits->data[s * vocab_size + v] = sum;
    }
  }
}

void model_backward(
  GPTModel* model,
  int* input_ids,
  int seq_len,
  Tensor* grad_logits,
  BackwardCache* cache,
  Gradients* grads
) {
  if (model == NULL || grad_logits == NULL || cache == NULL || grads == NULL) return;

  int hidden_dim = model->config.hidden_dim;
  int vocab_size = model->config.vocab_size;
  int ffn_dim = model->config.ffn_dim;

  int h_shape[] = {seq_len, hidden_dim};
  int f_shape[] = {seq_len, ffn_dim};

  Tensor* grad_hidden = tensor_zeros(2, h_shape);

  for (int h = 0; h < hidden_dim; h++) {
    for (int v = 0; v < vocab_size; v++) {
      float sum = 0.0f;
      for (int s = 0; s < seq_len; s++) {
        sum += cache->final_ln_output->data[s * hidden_dim + h] *
             grad_logits->data[s * vocab_size + v];
      }
      grads->d_lm_head->data[h * vocab_size + v] += sum;
    }
  }

  for (int s = 0; s < seq_len; s++) {
    for (int h = 0; h < hidden_dim; h++) {
      float sum = 0.0f;
      for (int v = 0; v < vocab_size; v++) {
        sum += grad_logits->data[s * vocab_size + v] *
             model->lm_head->data[h * vocab_size + v];
      }
      grad_hidden->data[s * hidden_dim + h] = sum;
    }
  }

  Tensor* grad_final_ln_input = tensor_zeros(2, h_shape);
  layernorm_backward(
    grad_hidden,
    cache->final_ln_input,
    model->final_ln->gamma,
    grads->d_final_ln_gamma,
    grads->d_final_ln_beta,
    grad_final_ln_input,
    model->final_ln->eps
  );

  tensor_free(grad_hidden);
  grad_hidden = grad_final_ln_input;

  for (int l = model->config.num_layers - 1; l >= 0; l--) {
    TransformerBlock* layer = model->layers[l];

    for (int h = 0; h < hidden_dim; h++) {
      for (int s = 0; s < seq_len; s++) {
        grads->layers[l].d_b2->data[h] += grad_hidden->data[s * hidden_dim + h];
      }
    }

    Tensor* grad_ffn_hidden = tensor_zeros(2, f_shape);
    for (int s = 0; s < seq_len; s++) {
      for (int f = 0; f < ffn_dim; f++) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
          sum += grad_hidden->data[s * hidden_dim + h] *
               layer->ffn->W2->data[f * hidden_dim + h];
        }
        grad_ffn_hidden->data[s * ffn_dim + f] = sum;
      }
    }

    for (int f = 0; f < ffn_dim; f++) {
      for (int h = 0; h < hidden_dim; h++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += cache->ffn_hidden[l]->data[s * ffn_dim + f] *
               grad_hidden->data[s * hidden_dim + h];
        }
        grads->layers[l].d_W2->data[f * hidden_dim + h] += sum;
      }
    }

    Tensor* grad_gelu_input = tensor_zeros(2, f_shape);
    gelu_backward(grad_ffn_hidden, cache->ffn_hidden[l], grad_gelu_input);
    tensor_free(grad_ffn_hidden);

    for (int f = 0; f < ffn_dim; f++) {
      for (int s = 0; s < seq_len; s++) {
        grads->layers[l].d_b1->data[f] += grad_gelu_input->data[s * ffn_dim + f];
      }
    }

    for (int h = 0; h < hidden_dim; h++) {
      for (int f = 0; f < ffn_dim; f++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += cache->ln2_outputs[l]->data[s * hidden_dim + h] *
               grad_gelu_input->data[s * ffn_dim + f];
        }
        grads->layers[l].d_W1->data[h * ffn_dim + f] += sum;
      }
    }

    Tensor* grad_ln2_output = tensor_zeros(2, h_shape);
    for (int s = 0; s < seq_len; s++) {
      for (int h = 0; h < hidden_dim; h++) {
        float sum = 0.0f;
        for (int f = 0; f < ffn_dim; f++) {
          sum += grad_gelu_input->data[s * ffn_dim + f] *
               layer->ffn->W1->data[h * ffn_dim + f];
        }
        grad_ln2_output->data[s * hidden_dim + h] = sum;
      }
    }
    tensor_free(grad_gelu_input);

    Tensor* grad_attn_output = tensor_zeros(2, h_shape);
    layernorm_backward(
      grad_ln2_output,
      cache->attn_outputs[l],
      layer->ln2->gamma,
      grads->layers[l].d_ln2_gamma,
      grads->layers[l].d_ln2_beta,
      grad_attn_output,
      layer->ln2->eps
    );
    tensor_free(grad_ln2_output);

    for (int i = 0; i < seq_len * hidden_dim; i++) {
      grad_attn_output->data[i] += grad_hidden->data[i];
    }

    Tensor* grad_ln1_output = tensor_zeros(2, h_shape);

    for (int h1 = 0; h1 < hidden_dim; h1++) {
      for (int h2 = 0; h2 < hidden_dim; h2++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += cache->ln1_outputs[l]->data[s * hidden_dim + h1] *
               grad_attn_output->data[s * hidden_dim + h2];
        }
        grads->layers[l].d_W_o->data[h1 * hidden_dim + h2] += sum;
      }
    }

    for (int s = 0; s < seq_len; s++) {
      for (int h1 = 0; h1 < hidden_dim; h1++) {
        float sum = 0.0f;
        for (int h2 = 0; h2 < hidden_dim; h2++) {
          sum += grad_attn_output->data[s * hidden_dim + h2] *
               layer->attn->W_o->data[h1 * hidden_dim + h2];
        }
        grad_ln1_output->data[s * hidden_dim + h1] = sum;
      }
    }

    for (int h1 = 0; h1 < hidden_dim; h1++) {
      for (int h2 = 0; h2 < hidden_dim; h2++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += cache->ln1_outputs[l]->data[s * hidden_dim + h1] *
               grad_ln1_output->data[s * hidden_dim + h2];
        }
        grads->layers[l].d_W_q->data[h1 * hidden_dim + h2] += sum * 0.33f;
        grads->layers[l].d_W_k->data[h1 * hidden_dim + h2] += sum * 0.33f;
        grads->layers[l].d_W_v->data[h1 * hidden_dim + h2] += sum * 0.33f;
      }
    }

    tensor_free(grad_attn_output);

    Tensor* grad_layer_input = tensor_zeros(2, h_shape);
    layernorm_backward(
      grad_ln1_output,
      cache->layer_inputs[l],
      layer->ln1->gamma,
      grads->layers[l].d_ln1_gamma,
      grads->layers[l].d_ln1_beta,
      grad_layer_input,
      layer->ln1->eps
    );
    tensor_free(grad_ln1_output);

    tensor_free(grad_hidden);
    grad_hidden = grad_layer_input;
  }

  embedding_backward(
    grad_hidden,
    input_ids,
    seq_len,
    grads->d_token_embedding,
    grads->d_position_embedding
  );

  tensor_free(grad_hidden);
}
