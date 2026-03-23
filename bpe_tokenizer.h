#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdio.h>

/**
 * BPE Tokenizer - Byte Pair Encoding 分词器
 *
 * BPE 通过迭代合并最频繁的字符对来构建子词词汇表
 * 这比字节级分词器效率高得多
 */

#define BPE_MAX_VOCAB_SIZE 8192
#define BPE_MAX_TOKEN_LEN 64
#define BPE_MAX_MERGES 8000

/**
 * BPE 合并规则
 */
typedef struct {
    int first;      // 第一个 token ID
    int second;     // 第二个 token ID
    int result;     // 合并后的 token ID
} BPEMerge;

/**
 * BPE Tokenizer 结构
 */
typedef struct {
    // 词汇表
    char** vocab;           // token ID -> token 字符串
    int vocab_size;         // 当前词汇表大小

    // 合并规则
    BPEMerge* merges;       // 合并规则数组
    int num_merges;         // 合并规则数量

    // 特殊 token
    int pad_id;             // <pad> token ID
    int unk_id;             // <unk> token ID
    int bos_id;             // <bos> token ID
    int eos_id;             // <eos> token ID

    // 字符到 token ID 映射 (用于基础字符)
    int char_to_id[256];    // ASCII 字符 -> token ID
} BPETokenizer;

// ============ 创建和销毁 ============

/**
 * 创建 BPE tokenizer (空的，需要训练或加载词汇表)
 */
BPETokenizer* bpe_tokenizer_create(void);

/**
 * 释放 BPE tokenizer
 */
void bpe_tokenizer_free(BPETokenizer* tok);

// ============ 词汇表训练 ============

/**
 * 从文本训练 BPE 词汇表
 * @param tok tokenizer
 * @param text 训练文本
 * @param num_merges 合并次数 (决定词汇表大小)
 * @return 0 成功, -1 失败
 */
int bpe_train(BPETokenizer* tok, const char* text, int num_merges);

/**
 * 从文件训练 BPE 词汇表
 */
int bpe_train_from_file(BPETokenizer* tok, const char* filepath, int num_merges);

// ============ 词汇表保存/加载 ============

/**
 * 保存词汇表到文件
 */
int bpe_save_vocab(BPETokenizer* tok, const char* filepath);

/**
 * 从文件加载词汇表
 */
int bpe_load_vocab(BPETokenizer* tok, const char* filepath);

// ============ 编码和解码 ============

/**
 * 将文本编码为 token IDs
 * @param tok tokenizer
 * @param text 输入文本
 * @param out_len 输出 token 数量
 * @param add_bos 是否添加 BOS token
 * @param add_eos 是否添加 EOS token
 * @return token ID 数组 (需要 free)
 */
int* bpe_encode(BPETokenizer* tok, const char* text, int* out_len,
                int add_bos, int add_eos);

/**
 * 将 token IDs 解码为文本
 * @param tok tokenizer
 * @param ids token ID 数组
 * @param len 数组长度
 * @return 解码后的文本 (需要 free)
 */
char* bpe_decode(BPETokenizer* tok, int* ids, int len);

/**
 * 解码单个 token
 */
const char* bpe_decode_token(BPETokenizer* tok, int id);

// ============ 工具函数 ============

/**
 * 获取词汇表大小
 */
int bpe_vocab_size(BPETokenizer* tok);

/**
 * 打印词汇表统计
 */
void bpe_print_stats(BPETokenizer* tok);

/**
 * 打印 token 序列
 */
void bpe_print_tokens(BPETokenizer* tok, int* ids, int len);

#endif // BPE_TOKENIZER_H
