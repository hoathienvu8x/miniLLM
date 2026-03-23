#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

/**
 * ModelConfig - GPT 模型配置
 */
typedef struct {
    int vocab_size;     // 词汇表大小 (默认 260)
    int hidden_dim;     // 隐藏层维度 (默认 64)
    int num_heads;      // 注意力头数 (默认 4)
    int num_layers;     // Transformer 层数 (默认 4)
    int ffn_dim;        // FFN 中间层维度 (默认 256, 通常是 hidden_dim * 4)
    int max_seq_len;    // 最大序列长度 (默认 128)
} ModelConfig;

/**
 * 创建默认配置
 * 适合教学的小模型，约 200K 参数
 */
static inline ModelConfig default_config(void) {
    ModelConfig config;
    config.vocab_size = 260;
    config.hidden_dim = 64;
    config.num_heads = 4;
    config.num_layers = 4;
    config.ffn_dim = 256;
    config.max_seq_len = 128;
    return config;
}

/**
 * 创建小型配置
 * 用于快速测试
 */
static inline ModelConfig tiny_config(void) {
    ModelConfig config;
    config.vocab_size = 260;
    config.hidden_dim = 32;
    config.num_heads = 2;
    config.num_layers = 2;
    config.ffn_dim = 128;
    config.max_seq_len = 64;
    return config;
}

/**
 * 创建中型配置
 * 约 1M 参数
 */
static inline ModelConfig medium_config(void) {
    ModelConfig config;
    config.vocab_size = 260;
    config.hidden_dim = 128;
    config.num_heads = 4;
    config.num_layers = 6;
    config.ffn_dim = 512;
    config.max_seq_len = 256;
    return config;
}

/**
 * 创建 BPE 配置 (使用 BPE 分词器)
 * vocab_size 应该与 BPE 训练后的词汇表大小匹配
 */
static inline ModelConfig bpe_config(int vocab_size) {
    ModelConfig config;
    config.vocab_size = vocab_size;
    config.hidden_dim = 128;
    config.num_heads = 4;
    config.num_layers = 4;
    config.ffn_dim = 512;
    config.max_seq_len = 256;
    return config;
}

/**
 * 创建小型 BPE 配置
 */
static inline ModelConfig bpe_small_config(int vocab_size) {
    ModelConfig config;
    config.vocab_size = vocab_size;
    config.hidden_dim = 64;
    config.num_heads = 4;
    config.num_layers = 4;
    config.ffn_dim = 256;
    config.max_seq_len = 128;
    return config;
}

/**
 * 打印配置信息
 */
static inline void config_print(ModelConfig* config) {
    printf("ModelConfig:\n");
    printf("  vocab_size: %d\n", config->vocab_size);
    printf("  hidden_dim: %d\n", config->hidden_dim);
    printf("  num_heads: %d\n", config->num_heads);
    printf("  num_layers: %d\n", config->num_layers);
    printf("  ffn_dim: %d\n", config->ffn_dim);
    printf("  max_seq_len: %d\n", config->max_seq_len);
}

/**
 * 验证配置有效性
 * @return 1 有效, 0 无效
 */
static inline int config_validate(ModelConfig* config) {
    if (config->vocab_size <= 0) return 0;
    if (config->hidden_dim <= 0) return 0;
    if (config->num_heads <= 0) return 0;
    if (config->num_layers <= 0) return 0;
    if (config->ffn_dim <= 0) return 0;
    if (config->max_seq_len <= 0) return 0;
    if (config->hidden_dim % config->num_heads != 0) return 0;
    return 1;
}

/**
 * 保存配置到文件
 */
static inline int config_save(ModelConfig* config, const char* path) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) return -1;
    fwrite(config, sizeof(ModelConfig), 1, f);
    fclose(f);
    return 0;
}

/**
 * 从文件加载配置
 */
static inline int config_load(ModelConfig* config, const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    size_t read = fread(config, sizeof(ModelConfig), 1, f);
    fclose(f);
    return (read == 1) ? 0 : -1;
}

#endif // CONFIG_H
