#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "bpe_tokenizer.h"
#include "model.h"
#include "config.h"
#include "loss.h"
#include "backward.h"
#include "optimizer.h"
#include "math_ops.h"
#include "tensor.h"

typedef struct {
  int* tokens;
  int num_tokens;
  int seq_len;
  int batch_size;
  int current_pos;
} BPEDataLoader;

BPEDataLoader* bpe_dataloader_create(BPETokenizer* tok, const char* filepath,
                    int seq_len, int batch_size) {
  FILE* f = fopen(filepath, "r");
  if (!f) {
    printf("Failed to open file: %s\n", filepath);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char* text = (char*)malloc(size + 1);
  size_t read_size = fread(text, 1, size, f);
  text[read_size] = '\0';
  fclose(f);

  int num_tokens;
  int* tokens = bpe_encode(tok, text, &num_tokens, 0, 0);
  free(text);

  if (tokens == NULL || num_tokens < seq_len + 1) {
    printf("Insufficient data! Required: %d tokens, Found: %d.\n", seq_len + 1, num_tokens);
    if (tokens) free(tokens);
    return NULL;
  }

  BPEDataLoader* dl = (BPEDataLoader*)malloc(sizeof(BPEDataLoader));
  dl->tokens = tokens;
  dl->num_tokens = num_tokens;
  dl->seq_len = seq_len;
  dl->batch_size = batch_size;
  dl->current_pos = 0;

  printf("Data has been successfully loaded: %d tokens, seq_len=%d, batch_size=%d\n",
       num_tokens, seq_len, batch_size);

  return dl;
}

void bpe_dataloader_free(BPEDataLoader* dl) {
  if (dl) {
    if (dl->tokens) free(dl->tokens);
    free(dl);
  }
}

int bpe_dataloader_next_batch(BPEDataLoader* dl, int* input_ids, int* targets) {
  int samples_available = (dl->num_tokens - dl->current_pos - 1) / dl->seq_len;
  if (samples_available <= 0) return 0;

  int actual_batch = (samples_available < dl->batch_size) ? samples_available : dl->batch_size;

  for (int b = 0; b < actual_batch; b++) {
    int start = dl->current_pos + b * dl->seq_len;
    for (int i = 0; i < dl->seq_len; i++) {
      input_ids[b * dl->seq_len + i] = dl->tokens[start + i];
      targets[b * dl->seq_len + i] = dl->tokens[start + i + 1];
    }
  }

  dl->current_pos += actual_batch * dl->seq_len;
  return actual_batch;
}

void bpe_dataloader_reset(BPEDataLoader* dl) {
  dl->current_pos = 0;
}

static void simple_softmax(float* data, int size) {
  float max_val = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] > max_val) max_val = data[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    data[i] = expf(data[i] - max_val);
    sum += data[i];
  }

  for (int i = 0; i < size; i++) {
    data[i] /= sum;
  }
}

char* generate_sample(GPTModel* model, BPETokenizer* tok, const char* prompt, int max_tokens) {
  int prompt_len;
  int* input_ids = bpe_encode(tok, prompt, &prompt_len, 0, 0);
  if (!input_ids || prompt_len == 0) return strdup(prompt);

  int max_len = prompt_len + max_tokens;
  if (max_len > model->config.max_seq_len) max_len = model->config.max_seq_len;

  int* tokens = (int*)malloc(max_len * sizeof(int));
  memcpy(tokens, input_ids, prompt_len * sizeof(int));
  free(input_ids);

  int current_len = prompt_len;
  int vocab_size = model->config.vocab_size;

  GPTCache* cache = model_cache_create(model, max_len);
  Tensor* logits = tensor_create_2d(max_len, vocab_size);

  for (int i = 0; i < max_tokens && current_len < max_len; i++) {
    model_forward(model, tokens, current_len, cache, logits);
    float* last_logits = logits->data + (current_len - 1) * vocab_size;

    float* probs = (float*)malloc(vocab_size * sizeof(float));
    for (int j = 0; j < vocab_size; j++) {
      probs[j] = last_logits[j] / 0.8f;
    }
    simple_softmax(probs, vocab_size);

    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int next_token = vocab_size - 1;
    for (int j = 0; j < vocab_size; j++) {
      cumsum += probs[j];
      if (cumsum > r) { next_token = j; break; }
    }
    free(probs);

    if (next_token == tok->eos_id) break;
    tokens[current_len++] = next_token;
  }

  model_cache_free(cache);
  tensor_free(logits);

  char* result = bpe_decode(tok, tokens, current_len);
  free(tokens);
  return result;
}

char* bpe_generate(GPTModel* model, BPETokenizer* tok,
           const char* prompt, int max_new_tokens,
           float temperature, int top_k) {
  int prompt_len;
  int* input_ids = bpe_encode(tok, prompt, &prompt_len, 0, 0);
  if (!input_ids || prompt_len == 0) {
    return strdup(prompt);
  }

  int max_len = prompt_len + max_new_tokens;
  if (max_len > model->config.max_seq_len) {
    max_len = model->config.max_seq_len;
  }

  int* tokens = (int*)malloc(max_len * sizeof(int));
  memcpy(tokens, input_ids, prompt_len * sizeof(int));
  free(input_ids);

  int current_len = prompt_len;
  int vocab_size = model->config.vocab_size;
  srand(time(NULL));

  GPTCache* cache = model_cache_create(model, max_len);
  Tensor* logits = tensor_create_2d(max_len, vocab_size);

  for (int i = 0; i < max_new_tokens && current_len < max_len; i++) {

    model_forward(model, tokens, current_len, cache, logits);

    float* last_logits = logits->data + (current_len - 1) * vocab_size;

    int next_token;
    if (temperature <= 0.0f) {

      next_token = 0;
      float max_val = last_logits[0];
      for (int j = 1; j < vocab_size; j++) {
        if (last_logits[j] > max_val) {
          max_val = last_logits[j];
          next_token = j;
        }
      }
    } else {

      float* probs = (float*)malloc(vocab_size * sizeof(float));
      for (int j = 0; j < vocab_size; j++) {
        probs[j] = last_logits[j] / temperature;
      }
      simple_softmax(probs, vocab_size);

      if (top_k > 0 && top_k < vocab_size) {
        float* sorted = (float*)malloc(vocab_size * sizeof(float));
        memcpy(sorted, probs, vocab_size * sizeof(float));
        for (int a = 0; a < top_k; a++) {
          for (int b = a + 1; b < vocab_size; b++) {
            if (sorted[b] > sorted[a]) {
              float tmp = sorted[a];
              sorted[a] = sorted[b];
              sorted[b] = tmp;
            }
          }
        }
        float threshold = sorted[top_k - 1];
        free(sorted);

        float sum = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
          if (probs[j] < threshold) {
            probs[j] = 0.0f;
          } else {
            sum += probs[j];
          }
        }
        if (sum > 0) {
          for (int j = 0; j < vocab_size; j++) {
            probs[j] /= sum;
          }
        }
      }

      float r = (float)rand() / (float)RAND_MAX;
      float cumsum = 0.0f;
      next_token = vocab_size - 1;
      for (int j = 0; j < vocab_size; j++) {
        cumsum += probs[j];
        if (cumsum > r) {
          next_token = j;
          break;
        }
      }
      free(probs);
    }

    if (next_token == tok->eos_id) break;
    tokens[current_len++] = next_token;
  }

  model_cache_free(cache);
  tensor_free(logits);

  char* result = bpe_decode(tok, tokens, current_len);
  free(tokens);

  return result;
}
#ifdef LLM_CHAT
void print_help() {
  printf("\n");
  printf("Commands:\n");
  printf("  /help        - Show help\n");
  printf("  /temp <val>  - Set temperature (0-2, default 0.8)\n");
  printf("  /len <val>   - Set max generation length (default 50)\n");
  printf("  /topk <val>  - Set Top-K (default 20)\n");
  printf("  /quit        - Exit\n");
  printf("\n");
}

int main(int argc, char** argv) {
  const char* model_path = "model_bpe.bin";
  const char* vocab_path = "bpe_vocab.txt";

  if (argc > 1) model_path = argv[1];
  if (argc > 2) vocab_path = argv[2];

  printf("\n");
  printf("╔══════════════════════════════════════════════════╗\n");
  printf("║        miniLLM BPE Interactive Chat              ║\n");
  printf("╚══════════════════════════════════════════════════╝\n\n");

  printf("Loading model: %s\n", model_path);
  GPTModel* model = model_load(model_path);
  if (!model) {
    printf("Error: Failed to load model!\n");
    return 1;
  }

  printf("Loading vocabulary: %s\n", vocab_path);
  BPETokenizer* tok = bpe_tokenizer_create();
  if (bpe_load_vocab(tok, vocab_path) != 0) {
    printf("Error: Failed to load vocabulary!\n");
    return 1;
  }

  printf("\nModel parameters: %d\n", model_num_params(model));
  printf("Vocabulary size: %d\n", bpe_vocab_size(tok));

  print_help();

  float temperature = 0.8f;
  int max_tokens = 50;
  int top_k = 20;

  srand(time(NULL));

  char input[1024];

  while (1) {
    printf("\n[You] > ");
    fflush(stdout);

    if (fgets(input, sizeof(input), stdin) == NULL) break;

    int len = strlen(input);
    if (len > 0 && input[len-1] == '\n') input[len-1] = '\0';

    if (strlen(input) == 0) continue;

    if (input[0] == '/') {
      if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
        printf("Goodbye!\n");
        break;
      } else if (strcmp(input, "/help") == 0) {
        print_help();
      } else if (strncmp(input, "/temp ", 6) == 0) {
        temperature = atof(input + 6);
        printf("Temperature set to: %.2f\n", temperature);
      } else if (strncmp(input, "/len ", 5) == 0) {
        max_tokens = atoi(input + 5);
        printf("Max tokens set to: %d\n", max_tokens);
      } else if (strncmp(input, "/topk ", 6) == 0) {
        top_k = atoi(input + 6);
        printf("Top-K set to: %d\n", top_k);
      } else {
        printf("Unknown command. Type /help for assistance.\n");
      }
      continue;
    }

    char* response = bpe_generate(model, tok, input, max_tokens, temperature, top_k);

    printf("\n[Model] > %s\n", response);

    free(response);
  }

  model_free(model);
  bpe_tokenizer_free(tok);

  return 0;
}
#else
/*
int main(int argc, char** argv) {
  printf("\n=== Continue Training BPE Model ===\n\n");

  if (argc < 2) {
    printf("Usage: %s <training_data_file> [epochs] [learning_rate]\n", argv[0]);
    printf("Example: %s train_data.txt 50 0.0005\n", argv[0]);
    return 1;
  }

  const char* data_path = argv[1];
  int num_epochs = (argc > 2) ? atoi(argv[2]) : 50;
  float learning_rate = (argc > 3) ? atof(argv[3]) : 0.0005f;

  const char* model_path = "model_bpe.bin";
  const char* vocab_path = "bpe_vocab.txt";

  // 1. Load Vocabulary
  printf("1. Loading vocabulary: %s\n", vocab_path);
  BPETokenizer* tok = bpe_tokenizer_create();
  if (bpe_load_vocab(tok, vocab_path) != 0) {
    printf("Error: Failed to load vocabulary! Please run train_bpe first.\n");
    return 1;
  }
  printf("   Vocab size: %d\n", bpe_vocab_size(tok));

  // 2. Load Model
  printf("\n2. Loading model: %s\n", model_path);
  GPTModel* model = model_load(model_path);
  if (!model) {
    printf("Error: Failed to load model! Please run train_bpe first.\n");
    return 1;
  }
  printf("   Model parameters: %d\n", model_num_params(model));

  int seq_len = model->config.max_seq_len - 16;
  int batch_size = 4;

  // 3. Create Data Loader
  printf("\n3. Loading training data: %s\n", data_path);
  BPEDataLoader* dl = dataloader_create(tok, data_path, seq_len, batch_size);
  if (!dl) {
      printf("Failed to create data loader!\n");
      return 1;
  }
  printf("   Tokens: %d, seq_len: %d, batch_size: %d\n",
         dl->num_tokens, seq_len, batch_size);

  // 4. Initialize Optimizer
  printf("\n4. Initializing training (lr=%.6f, epochs=%d)\n", learning_rate, num_epochs);
  AdamOptimizer* opt = adam_create(model, learning_rate);
  Gradients* grads = gradients_create(model);
  BackwardCache* cache = backward_cache_create(model, seq_len);

  int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
  int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));
  Tensor* logits = tensor_create_2d(seq_len, model->config.vocab_size);
  Tensor* grad_logits = tensor_create_2d(seq_len, model->config.vocab_size);

  // 5. Training
  printf("\n5. Starting training resume...\n\n");

  srand(time(NULL));
  float best_loss = 1e9f;
  int total_steps = 0;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    dataloader_reset(dl);
    float epoch_loss = 0.0f;
    int epoch_steps = 0;

    int actual_batch;
    while ((actual_batch = dataloader_next_batch(dl, input_ids, targets)) > 0) {
      for (int b = 0; b < actual_batch; b++) {
        int* batch_input = input_ids + b * seq_len;
        int* batch_target = targets + b * seq_len;

        model_forward_with_cache(model, batch_input, seq_len, cache, logits);

        float loss = cross_entropy_loss_with_grad(
          logits, batch_target, seq_len, model->config.vocab_size, grad_logits);

        if (b == 0) gradients_zero(grads);
        model_backward(model, batch_input, seq_len, grad_logits, cache, grads);

        epoch_loss += loss;
      }

      adam_step(opt, model, grads);
      epoch_steps++;
      total_steps++;
    }

    float avg_loss = epoch_loss / (epoch_steps * batch_size);

    if ((epoch + 1) % 10 == 0 || epoch == 0) {
      printf("Epoch %3d/%d | Loss: %.4f | PPL: %.2f\n",
             epoch + 1, num_epochs, avg_loss, expf(avg_loss));
    }

    if (avg_loss < best_loss) {
      best_loss = avg_loss;
      model_save(model, model_path);
    }

    // Show sample generation every 20 epochs
    if ((epoch + 1) % 20 == 0) {
      printf("\n  Generation samples:\n");
      const char* prompts[] = {"hello", "the", "good"};
      for (int i = 0; i < 3; i++) {
        char* output = generate_sample(model, tok, prompts[i], 30);
        printf("    '%s' -> '%.60s...'\n", prompts[i], output);
        free(output);
      }
      printf("\n");
    }
  }

  // 6. Save
  printf("\n6. Saving model...\n");
  model_save(model, model_path);
  printf("   Model saved to: %s\n", model_path);
  printf("   Best Loss: %.4f\n", best_loss);

  // Cleanup
  free(input_ids);
  free(targets);
  tensor_free(logits);
  tensor_free(grad_logits);
  backward_cache_free(cache);
  gradients_free(grads);
  adam_free(opt);
  dataloader_free(dl);
  model_free(model);
  bpe_tokenizer_free(tok);

  printf("\n=== Training complete! ===\n");
  return 0;
}
*/
int main(int argc, char** argv) {
  printf("=== miniLLM BPE Training ===\n\n");

  const char* data_path = "train_data.txt";
  const char* vocab_path = "bpe_vocab.txt";
  const char* model_path = "model_bpe.bin";
  int num_merges = 500;
  int num_epochs = 10;
  int seq_len = 64;
  int batch_size = 4;
  float learning_rate = 0.0003f;

  if (argc > 1) data_path = argv[1];
  if (argc > 2) num_epochs = atoi(argv[2]);
  if (argc > 3) learning_rate = atof(argv[3]);

  printf("1. Training BPE tokenizer...\n");
  BPETokenizer* tok = bpe_tokenizer_create();

  FILE* f = fopen(data_path, "r");
  if (!f) {
    printf("Failed to open training data: %s\n", data_path);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* text = (char*)malloc(size + 1);
  size_t read_size = fread(text, 1, size, f);
  text[read_size] = '\0';
  fclose(f);

  printf("   Training set size: %.2f KB\n", size / 1024.0f);
  bpe_train(tok, text, num_merges);
  free(text);

  bpe_save_vocab(tok, vocab_path);
  printf("   Vocabulary saved to: %s\n", vocab_path);
  printf("   Vocabulary size: %d\n\n", bpe_vocab_size(tok));

  printf("2. Create model ...\n");
  ModelConfig config = bpe_config(bpe_vocab_size(tok));
  config.max_seq_len = seq_len + 16;
  config_print(&config);

  GPTModel* model = model_create(config);
  model_init_random(model, 0.02f);

  int num_params = model_num_params(model);
  printf("   Number of parameters: %d (%.2f K)\n\n", num_params, num_params / 1000.0f);

  printf("3. Creating data loader ...\n");
  BPEDataLoader* dl = bpe_dataloader_create(tok, data_path, seq_len, batch_size);
  if (!dl) {
    printf("Failed to create data loader!\n");
    return 1;
  }

  printf("4. Initializing training ...\n");
  AdamOptimizer* opt = adam_create(model, learning_rate);
  Gradients* grads = gradients_create(model);
  BackwardCache* cache = backward_cache_create(model, seq_len);

  int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
  int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));
  Tensor* logits = tensor_create_2d(seq_len, model->config.vocab_size);
  Tensor* grad_logits = tensor_create_2d(seq_len, model->config.vocab_size);

  printf("\n5. Start training (%d epochs)...\n\n", num_epochs);

  float best_loss = 1e9f;
  int total_steps = 0;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    bpe_dataloader_reset(dl);
    float epoch_loss = 0.0f;
    int epoch_steps = 0;

    int actual_batch;
    while ((actual_batch = bpe_dataloader_next_batch(dl, input_ids, targets)) > 0) {

      for (int b = 0; b < actual_batch; b++) {
        int* batch_input = input_ids + b * seq_len;
        int* batch_target = targets + b * seq_len;

        model_forward_with_cache(model, batch_input, seq_len, cache, logits);

        float loss = cross_entropy_loss_with_grad(
          logits, batch_target, seq_len, model->config.vocab_size, grad_logits);

        if (b == 0) gradients_zero(grads);
        model_backward(model, batch_input, seq_len, grad_logits, cache, grads);

        epoch_loss += loss;
      }

      adam_step(opt, model, grads);
      epoch_steps++;
      total_steps++;

      if (total_steps % 10 == 0) {
        printf("\r   Epoch %d, Step %d, Loss: %.4f",
             epoch + 1, total_steps, epoch_loss / (epoch_steps * actual_batch));
        fflush(stdout);
      }
    }

    float avg_loss = epoch_loss / (epoch_steps * batch_size);
    printf("\n   Epoch %d complete, Average Loss: %.4f, PPL: %.2f\n",
         epoch + 1, avg_loss, expf(avg_loss));

    if (avg_loss < best_loss) {
      best_loss = avg_loss;
      model_save(model, model_path);
      printf("   -> Saving best model (loss=%.4f)\n", best_loss);
    }

    if ((epoch + 1) % 2 == 0 || epoch == num_epochs - 1) {
      printf("\n   Generation example:\n");
      const char* prompts[] = {"hello", "the", "how"};
      for (int i = 0; i < 3; i++) {
        char* output = bpe_generate(model, tok, prompts[i], 30, 0.8f, 20);
        printf("     '%s' -> '%s'\n", prompts[i], output);
        free(output);
      }
      printf("\n");
    }
  }

  printf("\n6. Final generation test:\n\n");
  const char* test_prompts[] = {"hello ", "the ", "how are ", "good ", "one "};

  for (int i = 0; i < 5; i++) {
    char* output = bpe_generate(model, tok, test_prompts[i], 40, 0.7f, 30);
    printf("   '%s' -> '%s'\n", test_prompts[i], output);
    free(output);
  }

  free(input_ids);
  free(targets);
  tensor_free(logits);
  tensor_free(grad_logits);
  backward_cache_free(cache);
  gradients_free(grads);
  adam_free(opt);
  bpe_dataloader_free(dl);
  model_free(model);
  bpe_tokenizer_free(tok);

  printf("\n=== Training complete! ===\n");
  printf("Model saved to: %s\n", model_path);
  printf("Vocabulary saved to: %s\n", vocab_path);

  return 0;
}
#endif
