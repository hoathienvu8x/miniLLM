#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "config.h"
#include "embedding.h"
#include "transformer.h"
#include "layernorm.h"
#include "kv_cache.h"

/**
 * GPTModel - 完整的 GPT 模型
 *
 * 结构:
 * 1. Token Embedding + Position Embedding
 * 2. N 层 Transformer Block
 * 3. Final LayerNorm
 * 4. LM Head (输出投影到词汇表)
 */
typedef struct {
    ModelConfig config;
    Embedding* embedding;           // 词嵌入层
    TransformerBlock** layers;      // Transformer 层数组
    LayerNorm* final_ln;            // 最终 LayerNorm
    Tensor* lm_head;                // 输出投影 [hidden_dim, vocab_size]
} GPTModel;

/**
 * GPTModel 计算缓存
 */
typedef struct {
    Tensor* hidden;                 // 隐藏状态 [seq_len, hidden_dim]
    Tensor* logits;                 // 输出 logits [seq_len, vocab_size]
    TransformerCache** layer_caches; // 每层的缓存
    Tensor* mask;                   // 因果掩码
    int seq_len;
    int num_layers;
} GPTCache;

// ============ 创建和销毁 ============

/**
 * 创建 GPT 模型
 * @param config 模型配置
 * @return 新创建的模型, 失败返回 NULL
 */
GPTModel* model_create(ModelConfig config);

/**
 * 释放模型
 */
void model_free(GPTModel* model);

/**
 * 创建模型缓存
 */
GPTCache* model_cache_create(GPTModel* model, int seq_len);

/**
 * 释放模型缓存
 */
void model_cache_free(GPTCache* cache);

/**
 * 调整缓存大小
 */
int model_cache_resize(GPTCache* cache, GPTModel* model, int new_seq_len);

// ============ 初始化 ============

/**
 * 随机初始化模型参数
 * @param model 模型
 * @param std 权重标准差 (推荐 0.02)
 */
void model_init_random(GPTModel* model, float std);

// ============ 前向传播 ============

/**
 * 模型前向传播
 * @param model 模型
 * @param input_ids 输入 token ID 数组 [seq_len]
 * @param seq_len 序列长度
 * @param cache 计算缓存
 * @param logits 输出 logits [seq_len, vocab_size] (需预先分配)
 */
void model_forward(
    GPTModel* model,
    int* input_ids,
    int seq_len,
    GPTCache* cache,
    Tensor* logits
);

/**
 * 获取最后一个位置的 logits (用于生成)
 * @param logits 完整 logits [seq_len, vocab_size]
 * @param seq_len 序列长度
 * @param last_logits 输出 [vocab_size] (需预先分配)
 */
void model_get_last_logits(Tensor* logits, int seq_len, Tensor* last_logits);

// ============ 保存和加载 ============

/**
 * 保存模型到文件
 * @param model 模型
 * @param path 文件路径
 * @return 0 成功, -1 失败
 */
int model_save(GPTModel* model, const char* path);

/**
 * 从文件加载模型
 * @param path 文件路径
 * @return 加载的模型, 失败返回 NULL
 */
GPTModel* model_load(const char* path);

// ============ 工具函数 ============

/**
 * 计算模型参数数量
 */
int model_num_params(GPTModel* model);

/**
 * 打印模型信息
 */
void model_print_info(GPTModel* model);

/**
 * 估计模型内存占用 (字节)
 */
size_t model_memory_size(GPTModel* model);

// ============ KV Cache 支持 ============

/**
 * 创建模型的 KV Cache
 * @param model 模型
 * @return KV Cache 实例
 */
KVCache* model_create_kv_cache(GPTModel* model);

/**
 * Prefill 阶段: 处理整个 prompt 并填充 KV Cache
 *
 * @param model 模型
 * @param input_ids 输入 token ID 数组 [seq_len]
 * @param seq_len 序列长度
 * @param kv_cache KV 缓存 (会被填充)
 * @param gpt_cache 计算缓存
 * @param logits 输出 logits [seq_len, vocab_size]
 */
void model_forward_prefill(
    GPTModel* model,
    int* input_ids,
    int seq_len,
    KVCache* kv_cache,
    GPTCache* gpt_cache,
    Tensor* logits
);

/**
 * Decode 阶段: 处理单个新 token
 *
 * @param model 模型
 * @param token_id 新 token ID
 * @param pos 当前位置 (已生成的 token 数)
 * @param kv_cache KV 缓存 (会被更新)
 * @param gpt_cache 计算缓存
 * @param logits 输出 logits [vocab_size]
 */
void model_forward_decode(
    GPTModel* model,
    int token_id,
    int pos,
    KVCache* kv_cache,
    GPTCache* gpt_cache,
    Tensor* logits
);

#endif // MODEL_H
