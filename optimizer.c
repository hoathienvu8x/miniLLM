#include "optimizer.h"
#include <stdlib.h>
#include <math.h>

AdamOptimizer* adam_create(GPTModel* model, float lr) {
  return adam_create_full(model, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
}

AdamOptimizer* adam_create_full(
  GPTModel* model,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float weight_decay
) {
  if (model == NULL) return NULL;

  AdamOptimizer* opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
  if (opt == NULL) return NULL;

  opt->lr = lr;
  opt->beta1 = beta1;
  opt->beta2 = beta2;
  opt->eps = eps;
  opt->weight_decay = weight_decay;
  opt->t = 0;

  opt->m = gradients_create(model);
  opt->v = gradients_create(model);

  if (opt->m == NULL || opt->v == NULL) {
    if (opt->m) gradients_free(opt->m);
    if (opt->v) gradients_free(opt->v);
    free(opt);
    return NULL;
  }

  return opt;
}

void adam_free(AdamOptimizer* opt) {
  if (opt == NULL) return;
  gradients_free(opt->m);
  gradients_free(opt->v);
  free(opt);
}

static void adam_update_tensor(
  Tensor* param,
  Tensor* grad,
  Tensor* m,
  Tensor* v,
  float lr,
  float beta1,
  float beta2,
  float eps,
  float weight_decay,
  float bias_correction1,
  float bias_correction2
) {
  if (param == NULL || grad == NULL || m == NULL || v == NULL) return;

  for (int i = 0; i < param->size; i++) {
    float g = grad->data[i];

    m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * g;

    v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * g * g;

    float m_hat = m->data[i] / bias_correction1;
    float v_hat = v->data[i] / bias_correction2;

    param->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);

    param->data[i] -= lr * weight_decay * param->data[i];
  }
}

void adam_step(AdamOptimizer* opt, GPTModel* model, Gradients* grads) {
  if (opt == NULL || model == NULL || grads == NULL) return;

  opt->t++;

  float bias_correction1 = 1.0f - powf(opt->beta1, (float)opt->t);
  float bias_correction2 = 1.0f - powf(opt->beta2, (float)opt->t);

  adam_update_tensor(
    model->embedding->token_embedding,
    grads->d_token_embedding,
    opt->m->d_token_embedding,
    opt->v->d_token_embedding,
    opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
    bias_correction1, bias_correction2
  );

  adam_update_tensor(
    model->embedding->position_embedding,
    grads->d_position_embedding,
    opt->m->d_position_embedding,
    opt->v->d_position_embedding,
    opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
    bias_correction1, bias_correction2
  );

  for (int l = 0; l < model->config.num_layers; l++) {
    TransformerBlock* layer = model->layers[l];

    adam_update_tensor(
      layer->ln1->gamma,
      grads->layers[l].d_ln1_gamma,
      opt->m->layers[l].d_ln1_gamma,
      opt->v->layers[l].d_ln1_gamma,
      opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->ln1->beta,
      grads->layers[l].d_ln1_beta,
      opt->m->layers[l].d_ln1_beta,
      opt->v->layers[l].d_ln1_beta,
      opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
      bias_correction1, bias_correction2
    );

    adam_update_tensor(
      layer->attn->W_q,
      grads->layers[l].d_W_q,
      opt->m->layers[l].d_W_q,
      opt->v->layers[l].d_W_q,
      opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->attn->W_k,
      grads->layers[l].d_W_k,
      opt->m->layers[l].d_W_k,
      opt->v->layers[l].d_W_k,
      opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->attn->W_v,
      grads->layers[l].d_W_v,
      opt->m->layers[l].d_W_v,
      opt->v->layers[l].d_W_v,
      opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->attn->W_o,
      grads->layers[l].d_W_o,
      opt->m->layers[l].d_W_o,
      opt->v->layers[l].d_W_o,
      opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
      bias_correction1, bias_correction2
    );

    adam_update_tensor(
      layer->ln2->gamma,
      grads->layers[l].d_ln2_gamma,
      opt->m->layers[l].d_ln2_gamma,
      opt->v->layers[l].d_ln2_gamma,
      opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->ln2->beta,
      grads->layers[l].d_ln2_beta,
      opt->m->layers[l].d_ln2_beta,
      opt->v->layers[l].d_ln2_beta,
      opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
      bias_correction1, bias_correction2
    );

    adam_update_tensor(
      layer->ffn->W1,
      grads->layers[l].d_W1,
      opt->m->layers[l].d_W1,
      opt->v->layers[l].d_W1,
      opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->ffn->b1,
      grads->layers[l].d_b1,
      opt->m->layers[l].d_b1,
      opt->v->layers[l].d_b1,
      opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->ffn->W2,
      grads->layers[l].d_W2,
      opt->m->layers[l].d_W2,
      opt->v->layers[l].d_W2,
      opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
      bias_correction1, bias_correction2
    );
    adam_update_tensor(
      layer->ffn->b2,
      grads->layers[l].d_b2,
      opt->m->layers[l].d_b2,
      opt->v->layers[l].d_b2,
      opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
      bias_correction1, bias_correction2
    );
  }

  adam_update_tensor(
    model->final_ln->gamma,
    grads->d_final_ln_gamma,
    opt->m->d_final_ln_gamma,
    opt->v->d_final_ln_gamma,
    opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
    bias_correction1, bias_correction2
  );
  adam_update_tensor(
    model->final_ln->beta,
    grads->d_final_ln_beta,
    opt->m->d_final_ln_beta,
    opt->v->d_final_ln_beta,
    opt->lr, opt->beta1, opt->beta2, opt->eps, 0.0f,
    bias_correction1, bias_correction2
  );

  adam_update_tensor(
    model->lm_head,
    grads->d_lm_head,
    opt->m->d_lm_head,
    opt->v->d_lm_head,
    opt->lr, opt->beta1, opt->beta2, opt->eps, opt->weight_decay,
    bias_correction1, bias_correction2
  );
}

void adam_reset(AdamOptimizer* opt) {
  if (opt == NULL) return;
  opt->t = 0;
  gradients_zero(opt->m);
  gradients_zero(opt->v);
}

SGDOptimizer* sgd_create(GPTModel* model, float lr, float momentum) {
  if (model == NULL) return NULL;

  SGDOptimizer* opt = (SGDOptimizer*)malloc(sizeof(SGDOptimizer));
  if (opt == NULL) return NULL;

  opt->lr = lr;
  opt->momentum = momentum;
  opt->weight_decay = 0.0f;

  opt->velocity = gradients_create(model);
  if (opt->velocity == NULL) {
    free(opt);
    return NULL;
  }

  return opt;
}

void sgd_free(SGDOptimizer* opt) {
  if (opt == NULL) return;
  gradients_free(opt->velocity);
  free(opt);
}

static void sgd_update_tensor(
  Tensor* param,
  Tensor* grad,
  Tensor* velocity,
  float lr,
  float momentum,
  float weight_decay
) {
  if (param == NULL || grad == NULL || velocity == NULL) return;

  for (int i = 0; i < param->size; i++) {

    velocity->data[i] = momentum * velocity->data[i] + grad->data[i];

    param->data[i] -= lr * velocity->data[i];

    param->data[i] -= lr * weight_decay * param->data[i];
  }
}

void sgd_step(SGDOptimizer* opt, GPTModel* model, Gradients* grads) {
  if (opt == NULL || model == NULL || grads == NULL) return;

  sgd_update_tensor(
    model->embedding->token_embedding,
    grads->d_token_embedding,
    opt->velocity->d_token_embedding,
    opt->lr, opt->momentum, opt->weight_decay
  );
  sgd_update_tensor(
    model->embedding->position_embedding,
    grads->d_position_embedding,
    opt->velocity->d_position_embedding,
    opt->lr, opt->momentum, 0.0f
  );

  for (int l = 0; l < model->config.num_layers; l++) {
    TransformerBlock* layer = model->layers[l];

    sgd_update_tensor(layer->ln1->gamma, grads->layers[l].d_ln1_gamma,
      opt->velocity->layers[l].d_ln1_gamma, opt->lr, opt->momentum, 0.0f);
    sgd_update_tensor(layer->ln1->beta, grads->layers[l].d_ln1_beta,
      opt->velocity->layers[l].d_ln1_beta, opt->lr, opt->momentum, 0.0f);

    sgd_update_tensor(layer->attn->W_q, grads->layers[l].d_W_q,
      opt->velocity->layers[l].d_W_q, opt->lr, opt->momentum, opt->weight_decay);
    sgd_update_tensor(layer->attn->W_k, grads->layers[l].d_W_k,
      opt->velocity->layers[l].d_W_k, opt->lr, opt->momentum, opt->weight_decay);
    sgd_update_tensor(layer->attn->W_v, grads->layers[l].d_W_v,
      opt->velocity->layers[l].d_W_v, opt->lr, opt->momentum, opt->weight_decay);
    sgd_update_tensor(layer->attn->W_o, grads->layers[l].d_W_o,
      opt->velocity->layers[l].d_W_o, opt->lr, opt->momentum, opt->weight_decay);

    sgd_update_tensor(layer->ln2->gamma, grads->layers[l].d_ln2_gamma,
      opt->velocity->layers[l].d_ln2_gamma, opt->lr, opt->momentum, 0.0f);
    sgd_update_tensor(layer->ln2->beta, grads->layers[l].d_ln2_beta,
      opt->velocity->layers[l].d_ln2_beta, opt->lr, opt->momentum, 0.0f);

    sgd_update_tensor(layer->ffn->W1, grads->layers[l].d_W1,
      opt->velocity->layers[l].d_W1, opt->lr, opt->momentum, opt->weight_decay);
    sgd_update_tensor(layer->ffn->b1, grads->layers[l].d_b1,
      opt->velocity->layers[l].d_b1, opt->lr, opt->momentum, 0.0f);
    sgd_update_tensor(layer->ffn->W2, grads->layers[l].d_W2,
      opt->velocity->layers[l].d_W2, opt->lr, opt->momentum, opt->weight_decay);
    sgd_update_tensor(layer->ffn->b2, grads->layers[l].d_b2,
      opt->velocity->layers[l].d_b2, opt->lr, opt->momentum, 0.0f);
  }

  sgd_update_tensor(model->final_ln->gamma, grads->d_final_ln_gamma,
    opt->velocity->d_final_ln_gamma, opt->lr, opt->momentum, 0.0f);
  sgd_update_tensor(model->final_ln->beta, grads->d_final_ln_beta,
    opt->velocity->d_final_ln_beta, opt->lr, opt->momentum, 0.0f);

  sgd_update_tensor(model->lm_head, grads->d_lm_head,
    opt->velocity->d_lm_head, opt->lr, opt->momentum, opt->weight_decay);
}

float cosine_lr(float lr_max, float lr_min, int step, int total_steps) {
  if (step >= total_steps) return lr_min;
  float progress = (float)step / (float)total_steps;
  return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159265f * progress));
}

float warmup_lr(float lr_max, int step, int warmup_steps) {
  if (step >= warmup_steps) return lr_max;
  return lr_max * (float)step / (float)warmup_steps;
}

float warmup_cosine_lr(
  float lr_max,
  float lr_min,
  int step,
  int warmup_steps,
  int total_steps
) {
  if (step < warmup_steps) {
    return warmup_lr(lr_max, step, warmup_steps);
  }
  int decay_steps = step - warmup_steps;
  int total_decay_steps = total_steps - warmup_steps;
  return cosine_lr(lr_max, lr_min, decay_steps, total_decay_steps);
}
