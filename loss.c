#include "loss.h"
#include <math.h>
#include <float.h>

void softmax_cross_entropy_grad(
  float* logits,
  int target,
  int vocab_size,
  float* grad
) {

  float max_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    grad[i] = expf(logits[i] - max_val);
    sum += grad[i];
  }

  for (int i = 0; i < vocab_size; i++) {
    grad[i] = grad[i] / sum;
    if (i == target) {
      grad[i] -= 1.0f;
    }
  }
}

float cross_entropy_loss(
  Tensor* logits,
  int* targets,
  int seq_len,
  int vocab_size
) {
  if (logits == NULL || targets == NULL) return 0.0f;

  float total_loss = 0.0f;

  for (int s = 0; s < seq_len; s++) {
    float* pos_logits = &logits->data[s * vocab_size];
    int target = targets[s];

    float max_val = pos_logits[0];
    for (int i = 1; i < vocab_size; i++) {
      if (pos_logits[i] > max_val) {
        max_val = pos_logits[i];
      }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
      sum_exp += expf(pos_logits[i] - max_val);
    }
    float log_sum_exp = max_val + logf(sum_exp);

    float loss = -pos_logits[target] + log_sum_exp;
    total_loss += loss;
  }

  return total_loss / seq_len;
}

void cross_entropy_grad(
  Tensor* logits,
  int* targets,
  int seq_len,
  int vocab_size,
  Tensor* grad_logits
) {
  if (logits == NULL || targets == NULL || grad_logits == NULL) return;

  for (int s = 0; s < seq_len; s++) {
    float* pos_logits = &logits->data[s * vocab_size];
    float* pos_grad = &grad_logits->data[s * vocab_size];
    int target = targets[s];

    softmax_cross_entropy_grad(pos_logits, target, vocab_size, pos_grad);

    for (int i = 0; i < vocab_size; i++) {
      pos_grad[i] /= seq_len;
    }
  }
}

float cross_entropy_loss_with_grad(
  Tensor* logits,
  int* targets,
  int seq_len,
  int vocab_size,
  Tensor* grad_logits
) {
  if (logits == NULL || targets == NULL) return 0.0f;

  float total_loss = 0.0f;

  for (int s = 0; s < seq_len; s++) {
    float* pos_logits = &logits->data[s * vocab_size];
    float* pos_grad = (grad_logits != NULL) ? &grad_logits->data[s * vocab_size] : NULL;
    int target = targets[s];

    float max_val = pos_logits[0];
    for (int i = 1; i < vocab_size; i++) {
      if (pos_logits[i] > max_val) {
        max_val = pos_logits[i];
      }
    }

    float sum_exp = 0.0f;
    float exp_vals[vocab_size];
    for (int i = 0; i < vocab_size; i++) {
      exp_vals[i] = expf(pos_logits[i] - max_val);
      sum_exp += exp_vals[i];
    }

    float log_sum_exp = max_val + logf(sum_exp);
    float loss = -pos_logits[target] + log_sum_exp;
    total_loss += loss;

    if (pos_grad != NULL) {
      for (int i = 0; i < vocab_size; i++) {
        pos_grad[i] = exp_vals[i] / sum_exp;
        if (i == target) {
          pos_grad[i] -= 1.0f;
        }

        pos_grad[i] /= seq_len;
      }
    }
  }

  return total_loss / seq_len;
}
