#include "transformer.h"
#include "math_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TransformerBlock* transformer_block_create(int hidden_dim, int num_heads, int ffn_dim) {
    if (hidden_dim <= 0 || num_heads <= 0 || ffn_dim <= 0) return NULL;
    if (hidden_dim % num_heads != 0) {
        fprintf(stderr, "Error: hidden_dim must be divisible by num_heads\n");
        return NULL;
    }

    TransformerBlock* block = (TransformerBlock*)malloc(sizeof(TransformerBlock));
    if (block == NULL) return NULL;

    block->hidden_dim = hidden_dim;
    block->num_heads = num_heads;
    block->ffn_dim = ffn_dim;

    // 创建各组件
    block->ln1 = layernorm_create(hidden_dim, 1e-5f);
    block->attn = attention_create(hidden_dim, num_heads);
    block->ln2 = layernorm_create(hidden_dim, 1e-5f);
    block->ffn = ffn_create(hidden_dim, ffn_dim);

    if (block->ln1 == NULL || block->attn == NULL ||
        block->ln2 == NULL || block->ffn == NULL) {
        transformer_block_free(block);
        return NULL;
    }

    return block;
}

void transformer_block_free(TransformerBlock* block) {
    if (block == NULL) return;

    if (block->ln1) layernorm_free(block->ln1);
    if (block->attn) attention_free(block->attn);
    if (block->ln2) layernorm_free(block->ln2);
    if (block->ffn) ffn_free(block->ffn);
    free(block);
}

TransformerCache* transformer_cache_create(int seq_len, int hidden_dim, int num_heads, int ffn_dim) {
    TransformerCache* cache = (TransformerCache*)malloc(sizeof(TransformerCache));
    if (cache == NULL) return NULL;

    cache->seq_len = seq_len;
    cache->hidden_dim = hidden_dim;

    int shape[] = {seq_len, hidden_dim};

    cache->ln1_out = tensor_zeros(2, shape);
    cache->attn_out = tensor_zeros(2, shape);
    cache->ln2_out = tensor_zeros(2, shape);
    cache->ffn_out = tensor_zeros(2, shape);
    cache->attn_cache = attention_cache_create(seq_len, hidden_dim, num_heads);
    cache->ffn_cache = ffn_cache_create(seq_len, ffn_dim);

    if (cache->ln1_out == NULL || cache->attn_out == NULL ||
        cache->ln2_out == NULL || cache->ffn_out == NULL ||
        cache->attn_cache == NULL || cache->ffn_cache == NULL) {
        transformer_cache_free(cache);
        return NULL;
    }

    return cache;
}

void transformer_cache_free(TransformerCache* cache) {
    if (cache == NULL) return;

    if (cache->ln1_out) tensor_free(cache->ln1_out);
    if (cache->attn_out) tensor_free(cache->attn_out);
    if (cache->ln2_out) tensor_free(cache->ln2_out);
    if (cache->ffn_out) tensor_free(cache->ffn_out);
    if (cache->attn_cache) attention_cache_free(cache->attn_cache);
    if (cache->ffn_cache) ffn_cache_free(cache->ffn_cache);
    free(cache);
}

int transformer_cache_resize(TransformerCache* cache, int new_seq_len) {
    if (cache == NULL) return -1;
    if (new_seq_len == cache->seq_len) return 0;

    int hidden_dim = cache->hidden_dim;

    // 重新分配张量
    if (cache->ln1_out) tensor_free(cache->ln1_out);
    if (cache->attn_out) tensor_free(cache->attn_out);
    if (cache->ln2_out) tensor_free(cache->ln2_out);
    if (cache->ffn_out) tensor_free(cache->ffn_out);

    int shape[] = {new_seq_len, hidden_dim};
    cache->ln1_out = tensor_zeros(2, shape);
    cache->attn_out = tensor_zeros(2, shape);
    cache->ln2_out = tensor_zeros(2, shape);
    cache->ffn_out = tensor_zeros(2, shape);

    // 调整子缓存
    attention_cache_resize(cache->attn_cache, new_seq_len);
    ffn_cache_resize(cache->ffn_cache, new_seq_len);

    cache->seq_len = new_seq_len;

    if (cache->ln1_out == NULL || cache->attn_out == NULL ||
        cache->ln2_out == NULL || cache->ffn_out == NULL) {
        return -1;
    }

    return 0;
}

void transformer_block_init(TransformerBlock* block, float std) {
    if (block == NULL) return;

    layernorm_init(block->ln1);
    attention_init_random(block->attn, std);
    layernorm_init(block->ln2);
    ffn_init_random(block->ffn, std);
}

void transformer_block_forward(
    TransformerBlock* block,
    Tensor* input,
    Tensor* mask,
    TransformerCache* cache,
    Tensor* output
) {
    if (block == NULL || input == NULL || output == NULL) return;

    int seq_len = input->shape[0];
    int hidden_dim = block->hidden_dim;

    // 检查并调整缓存大小
    if (cache != NULL && cache->seq_len != seq_len) {
        transformer_cache_resize(cache, seq_len);
    }

    // 分配临时张量 (如果没有缓存)
    Tensor *ln1_out, *attn_out, *ln2_out, *ffn_out;
    AttentionCache* attn_cache;
    FFNCache* ffn_cache;
    int need_free = 0;

    if (cache != NULL) {
        ln1_out = cache->ln1_out;
        attn_out = cache->attn_out;
        ln2_out = cache->ln2_out;
        ffn_out = cache->ffn_out;
        attn_cache = cache->attn_cache;
        ffn_cache = cache->ffn_cache;
    } else {
        int shape[] = {seq_len, hidden_dim};
        ln1_out = tensor_zeros(2, shape);
        attn_out = tensor_zeros(2, shape);
        ln2_out = tensor_zeros(2, shape);
        ffn_out = tensor_zeros(2, shape);
        attn_cache = attention_cache_create(seq_len, hidden_dim, block->num_heads);
        ffn_cache = ffn_cache_create(seq_len, block->ffn_dim);
        need_free = 1;
    }

    // Pre-LN Transformer Block:
    // x = x + Attention(LayerNorm(x))
    // x = x + FFN(LayerNorm(x))

    // Step 1: ln1_out = LayerNorm(input)
    layernorm_forward(block->ln1, input, ln1_out);

    // Step 2: attn_out = Attention(ln1_out)
    attention_forward(block->attn, ln1_out, mask, attn_cache, attn_out);

    // Step 3: residual1 = input + attn_out (存到 output 作为中间结果)
    for (int i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] + attn_out->data[i];
    }

    // Step 4: ln2_out = LayerNorm(residual1)
    layernorm_forward(block->ln2, output, ln2_out);

    // Step 5: ffn_out = FFN(ln2_out)
    ffn_forward(block->ffn, ln2_out, ffn_cache, ffn_out);

    // Step 6: output = residual1 + ffn_out
    for (int i = 0; i < output->size; i++) {
        output->data[i] = output->data[i] + ffn_out->data[i];
    }

    if (need_free) {
        tensor_free(ln1_out);
        tensor_free(attn_out);
        tensor_free(ln2_out);
        tensor_free(ffn_out);
        attention_cache_free(attn_cache);
        ffn_cache_free(ffn_cache);
    }
}

void transformer_block_print_info(TransformerBlock* block) {
    if (block == NULL) {
        printf("TransformerBlock: NULL\n");
        return;
    }

    printf("TransformerBlock Info:\n");
    printf("  hidden_dim: %d\n", block->hidden_dim);
    printf("  num_heads: %d\n", block->num_heads);
    printf("  head_dim: %d\n", block->hidden_dim / block->num_heads);
    printf("  ffn_dim: %d\n", block->ffn_dim);
    printf("  total params: %d\n", transformer_block_num_params(block));
    printf("\n  Components:\n");
    printf("    ln1: %d params\n", layernorm_num_params(block->ln1));
    printf("    attention: %d params\n", attention_num_params(block->attn));
    printf("    ln2: %d params\n", layernorm_num_params(block->ln2));
    printf("    ffn: %d params\n", ffn_num_params(block->ffn));
}

int transformer_block_num_params(TransformerBlock* block) {
    if (block == NULL) return 0;

    int ln1_params = layernorm_num_params(block->ln1);
    int attn_params = attention_num_params(block->attn);
    int ln2_params = layernorm_num_params(block->ln2);
    int ffn_params = ffn_num_params(block->ffn);

    return ln1_params + attn_params + ln2_params + ffn_params;
}

// ============ KV Cache 支持实现 ============

void transformer_block_forward_prefill(
    TransformerBlock* block,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    Tensor* mask,
    TransformerCache* cache,
    Tensor* output
) {
    if (block == NULL || input == NULL || kv_cache == NULL ||
        cache == NULL || output == NULL) return;

    int seq_len = input->shape[0];
    int hidden_dim = block->hidden_dim;

    // 检查并调整缓存大小
    if (cache->seq_len != seq_len) {
        transformer_cache_resize(cache, seq_len);
    }

    // Pre-LN Transformer Block with KV Cache:
    // Step 1: ln1_out = LayerNorm(input)
    layernorm_forward(block->ln1, input, cache->ln1_out);

    // Step 2: attn_out = Attention(ln1_out) with KV Cache
    attention_prefill_kv_cache(block->attn, cache->ln1_out, kv_cache, layer_idx,
                               mask, cache->attn_cache, cache->attn_out);

    // Step 3: residual1 = input + attn_out
    for (int i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] + cache->attn_out->data[i];
    }

    // Step 4: ln2_out = LayerNorm(residual1)
    layernorm_forward(block->ln2, output, cache->ln2_out);

    // Step 5: ffn_out = FFN(ln2_out)
    ffn_forward(block->ffn, cache->ln2_out, cache->ffn_cache, cache->ffn_out);

    // Step 6: output = residual1 + ffn_out
    for (int i = 0; i < output->size; i++) {
        output->data[i] = output->data[i] + cache->ffn_out->data[i];
    }
}

void transformer_block_forward_decode(
    TransformerBlock* block,
    Tensor* input,
    KVCache* kv_cache,
    int layer_idx,
    int pos,
    TransformerCache* cache,
    Tensor* output
) {
    if (block == NULL || input == NULL || kv_cache == NULL ||
        cache == NULL || output == NULL) return;

    int seq_len = input->shape[0];  // 通常为 1
    int hidden_dim = block->hidden_dim;

    // 确保缓存足够大
    if (cache->seq_len < seq_len) {
        transformer_cache_resize(cache, seq_len);
    }

    // 分配临时张量
    int shape[] = {seq_len, hidden_dim};
    Tensor* ln1_out = tensor_zeros(2, shape);
    Tensor* attn_out = tensor_zeros(2, shape);
    Tensor* ln2_out = tensor_zeros(2, shape);
    Tensor* ffn_out = tensor_zeros(2, shape);

    // Pre-LN Transformer Block with KV Cache (Decode):
    // Step 1: ln1_out = LayerNorm(input)
    layernorm_forward(block->ln1, input, ln1_out);

    // Step 2: attn_out = Attention(ln1_out) with KV Cache
    attention_forward_kv_cache(block->attn, ln1_out, kv_cache, layer_idx,
                               pos, cache->attn_cache, attn_out);

    // Step 3: residual1 = input + attn_out
    for (int i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] + attn_out->data[i];
    }

    // Step 4: ln2_out = LayerNorm(residual1)
    layernorm_forward(block->ln2, output, ln2_out);

    // Step 5: ffn_out = FFN(ln2_out)
    ffn_forward(block->ffn, ln2_out, cache->ffn_cache, ffn_out);

    // Step 6: output = residual1 + ffn_out
    for (int i = 0; i < output->size; i++) {
        output->data[i] = output->data[i] + ffn_out->data[i];
    }

    // 清理临时张量
    tensor_free(ln1_out);
    tensor_free(attn_out);
    tensor_free(ln2_out);
    tensor_free(ffn_out);
}
