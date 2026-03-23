#define _POSIX_C_SOURCE 200809L
#include "bpe_tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// ============ 内部辅助结构 ============

typedef struct {
    int first;
    int second;
    int count;
} PairCount;

// ============ 创建和销毁 ============

BPETokenizer* bpe_tokenizer_create(void) {
    BPETokenizer* tok = (BPETokenizer*)calloc(1, sizeof(BPETokenizer));
    if (tok == NULL) return NULL;

    tok->vocab = (char**)calloc(BPE_MAX_VOCAB_SIZE, sizeof(char*));
    tok->merges = (BPEMerge*)calloc(BPE_MAX_MERGES, sizeof(BPEMerge));

    if (tok->vocab == NULL || tok->merges == NULL) {
        bpe_tokenizer_free(tok);
        return NULL;
    }

    // 初始化特殊 token
    tok->vocab[0] = strdup("<pad>");
    tok->vocab[1] = strdup("<unk>");
    tok->vocab[2] = strdup("<bos>");
    tok->vocab[3] = strdup("<eos>");
    tok->pad_id = 0;
    tok->unk_id = 1;
    tok->bos_id = 2;
    tok->eos_id = 3;
    tok->vocab_size = 4;

    // 初始化字符映射
    for (int i = 0; i < 256; i++) {
        tok->char_to_id[i] = -1;
    }

    // 添加基础 ASCII 字符 (32-126 可打印字符 + 换行)
    for (int c = 32; c <= 126; c++) {
        char* s = (char*)malloc(2);
        s[0] = (char)c;
        s[1] = '\0';
        tok->vocab[tok->vocab_size] = s;
        tok->char_to_id[c] = tok->vocab_size;
        tok->vocab_size++;
    }

    // 添加换行符
    tok->vocab[tok->vocab_size] = strdup("\n");
    tok->char_to_id['\n'] = tok->vocab_size;
    tok->vocab_size++;

    // 添加制表符
    tok->vocab[tok->vocab_size] = strdup("\t");
    tok->char_to_id['\t'] = tok->vocab_size;
    tok->vocab_size++;

    tok->num_merges = 0;

    return tok;
}

void bpe_tokenizer_free(BPETokenizer* tok) {
    if (tok == NULL) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        free(tok->vocab);
    }

    if (tok->merges) free(tok->merges);

    free(tok);
}

// ============ 内部辅助函数 ============

// 将文本转换为初始 token 序列
static int* text_to_initial_tokens(BPETokenizer* tok, const char* text,
                                   int* out_len) {
    int len = strlen(text);
    int* tokens = (int*)malloc(len * sizeof(int));
    int count = 0;

    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        int id = tok->char_to_id[c];
        if (id >= 0) {
            tokens[count++] = id;
        } else {
            tokens[count++] = tok->unk_id;
        }
    }

    *out_len = count;
    return tokens;
}

// 统计相邻 token 对的频率
static PairCount* count_pairs(int* tokens, int len, int* num_pairs) {
    if (len < 2) {
        *num_pairs = 0;
        return NULL;
    }

    // 使用简单的数组计数 (对于小词汇表足够)
    int capacity = 10000;
    PairCount* pairs = (PairCount*)calloc(capacity, sizeof(PairCount));
    int count = 0;

    for (int i = 0; i < len - 1; i++) {
        int first = tokens[i];
        int second = tokens[i + 1];

        // 查找是否已存在
        int found = 0;
        for (int j = 0; j < count; j++) {
            if (pairs[j].first == first && pairs[j].second == second) {
                pairs[j].count++;
                found = 1;
                break;
            }
        }

        if (!found && count < capacity) {
            pairs[count].first = first;
            pairs[count].second = second;
            pairs[count].count = 1;
            count++;
        }
    }

    *num_pairs = count;
    return pairs;
}

// 找到最频繁的 pair
static int find_most_frequent_pair(PairCount* pairs, int num_pairs,
                                   int* first, int* second) {
    if (num_pairs == 0) return 0;

    int max_count = 0;
    int max_idx = -1;

    for (int i = 0; i < num_pairs; i++) {
        if (pairs[i].count > max_count) {
            max_count = pairs[i].count;
            max_idx = i;
        }
    }

    if (max_idx >= 0 && max_count >= 2) {
        *first = pairs[max_idx].first;
        *second = pairs[max_idx].second;
        return max_count;
    }

    return 0;
}

// 合并 tokens 中的指定 pair
static int* merge_pair(int* tokens, int len, int first, int second,
                       int new_id, int* new_len) {
    int* result = (int*)malloc(len * sizeof(int));
    int j = 0;

    for (int i = 0; i < len; i++) {
        if (i < len - 1 && tokens[i] == first && tokens[i + 1] == second) {
            result[j++] = new_id;
            i++;  // 跳过下一个 token
        } else {
            result[j++] = tokens[i];
        }
    }

    *new_len = j;
    return result;
}

// ============ 词汇表训练 ============

int bpe_train(BPETokenizer* tok, const char* text, int num_merges) {
    if (tok == NULL || text == NULL || num_merges <= 0) return -1;

    printf("BPE 训练开始...\n");
    printf("  初始词汇表大小: %d\n", tok->vocab_size);

    // 将文本转换为初始 token 序列
    int len;
    int* tokens = text_to_initial_tokens(tok, text, &len);

    printf("  文本长度: %d 字符\n", (int)strlen(text));
    printf("  初始 tokens: %d\n", len);

    // 迭代合并
    for (int i = 0; i < num_merges; i++) {
        // 统计 pair 频率
        int num_pairs;
        PairCount* pairs = count_pairs(tokens, len, &num_pairs);

        // 找最频繁的 pair
        int first, second;
        int count = find_most_frequent_pair(pairs, num_pairs, &first, &second);

        free(pairs);

        if (count < 2) {
            printf("  在第 %d 次合并后停止 (无更多可合并的 pair)\n", i);
            break;
        }

        // 创建新 token
        int new_id = tok->vocab_size;
        if (new_id >= BPE_MAX_VOCAB_SIZE) {
            printf("  达到最大词汇表大小\n");
            break;
        }

        // 合并 token 字符串
        const char* s1 = tok->vocab[first];
        const char* s2 = tok->vocab[second];
        int new_len = strlen(s1) + strlen(s2) + 1;
        char* new_token = (char*)malloc(new_len);
        strcpy(new_token, s1);
        strcat(new_token, s2);

        tok->vocab[new_id] = new_token;
        tok->vocab_size++;

        // 记录合并规则
        tok->merges[tok->num_merges].first = first;
        tok->merges[tok->num_merges].second = second;
        tok->merges[tok->num_merges].result = new_id;
        tok->num_merges++;

        // 在 tokens 中应用合并
        int merged_len;
        int* merged = merge_pair(tokens, len, first, second, new_id, &merged_len);
        free(tokens);
        tokens = merged;
        len = merged_len;

        if ((i + 1) % 100 == 0) {
            printf("  合并 %d: '%s' + '%s' -> '%s' (出现 %d 次, tokens: %d)\n",
                   i + 1, s1, s2, new_token, count, len);
        }
    }

    free(tokens);

    printf("BPE 训练完成!\n");
    printf("  最终词汇表大小: %d\n", tok->vocab_size);
    printf("  合并规则数量: %d\n", tok->num_merges);

    return 0;
}

int bpe_train_from_file(BPETokenizer* tok, const char* filepath, int num_merges) {
    FILE* f = fopen(filepath, "r");
    if (f == NULL) return -1;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* text = (char*)malloc(size + 1);
    size_t read = fread(text, 1, size, f);
    text[read] = '\0';
    fclose(f);

    int result = bpe_train(tok, text, num_merges);
    free(text);

    return result;
}

// ============ 词汇表保存/加载 ============

int bpe_save_vocab(BPETokenizer* tok, const char* filepath) {
    FILE* f = fopen(filepath, "w");
    if (f == NULL) return -1;

    // 写入词汇表大小
    fprintf(f, "%d %d\n", tok->vocab_size, tok->num_merges);

    // 写入词汇表
    for (int i = 0; i < tok->vocab_size; i++) {
        // 转义换行和制表符
        const char* s = tok->vocab[i];
        fprintf(f, "%d ", i);
        for (int j = 0; s[j]; j++) {
            if (s[j] == '\n') {
                fprintf(f, "\\n");
            } else if (s[j] == '\t') {
                fprintf(f, "\\t");
            } else if (s[j] == '\\') {
                fprintf(f, "\\\\");
            } else {
                fputc(s[j], f);
            }
        }
        fprintf(f, "\n");
    }

    // 写入合并规则
    for (int i = 0; i < tok->num_merges; i++) {
        fprintf(f, "%d %d %d\n",
                tok->merges[i].first,
                tok->merges[i].second,
                tok->merges[i].result);
    }

    fclose(f);
    return 0;
}

int bpe_load_vocab(BPETokenizer* tok, const char* filepath) {
    FILE* f = fopen(filepath, "r");
    if (f == NULL) return -1;

    // 清空现有词汇表
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i]) {
            free(tok->vocab[i]);
            tok->vocab[i] = NULL;
        }
    }

    // 重置字符映射
    for (int i = 0; i < 256; i++) {
        tok->char_to_id[i] = -1;
    }

    // 读取词汇表大小
    int vocab_size, num_merges;
    if (fscanf(f, "%d %d\n", &vocab_size, &num_merges) != 2) {
        fclose(f);
        return -1;
    }

    // 读取词汇表
    char line[1024];
    for (int i = 0; i < vocab_size; i++) {
        int id;
        // 使用 %d 读取 id，然后手动读取一个空格分隔符
        // 注意：不能使用 "%d " 因为它会消耗所有空白，破坏以空格开头的 token
        if (fscanf(f, "%d", &id) != 1) break;
        fgetc(f);  // 消耗单个空格分隔符

        if (fgets(line, sizeof(line), f) == NULL) break;

        // 移除换行符
        int len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

        // 反转义
        char* decoded = (char*)malloc(len + 1);
        int j = 0, k = 0;
        while (line[j]) {
            if (line[j] == '\\' && line[j + 1]) {
                if (line[j + 1] == 'n') {
                    decoded[k++] = '\n';
                    j += 2;
                } else if (line[j + 1] == 't') {
                    decoded[k++] = '\t';
                    j += 2;
                } else if (line[j + 1] == '\\') {
                    decoded[k++] = '\\';
                    j += 2;
                } else {
                    decoded[k++] = line[j++];
                }
            } else {
                decoded[k++] = line[j++];
            }
        }
        decoded[k] = '\0';

        tok->vocab[id] = decoded;

        // 更新字符映射
        if (strlen(decoded) == 1) {
            unsigned char c = (unsigned char)decoded[0];
            tok->char_to_id[c] = id;
        }
    }

    tok->vocab_size = vocab_size;

    // 读取合并规则
    for (int i = 0; i < num_merges; i++) {
        if (fscanf(f, "%d %d %d\n",
                   &tok->merges[i].first,
                   &tok->merges[i].second,
                   &tok->merges[i].result) != 3) break;
    }
    tok->num_merges = num_merges;

    fclose(f);
    return 0;
}

// ============ 编码和解码 ============

int* bpe_encode(BPETokenizer* tok, const char* text, int* out_len,
                int add_bos, int add_eos) {
    if (tok == NULL || text == NULL) {
        *out_len = 0;
        return NULL;
    }

    // 转换为初始 tokens
    int len;
    int* tokens = text_to_initial_tokens(tok, text, &len);

    // 应用所有合并规则
    for (int i = 0; i < tok->num_merges; i++) {
        int first = tok->merges[i].first;
        int second = tok->merges[i].second;
        int result = tok->merges[i].result;

        int new_len;
        int* merged = merge_pair(tokens, len, first, second, result, &new_len);
        free(tokens);
        tokens = merged;
        len = new_len;
    }

    // 添加特殊 tokens
    int extra = (add_bos ? 1 : 0) + (add_eos ? 1 : 0);
    if (extra > 0) {
        int* result = (int*)malloc((len + extra) * sizeof(int));
        int j = 0;
        if (add_bos) result[j++] = tok->bos_id;
        memcpy(result + j, tokens, len * sizeof(int));
        j += len;
        if (add_eos) result[j++] = tok->eos_id;
        free(tokens);
        tokens = result;
        len += extra;
    }

    *out_len = len;
    return tokens;
}

char* bpe_decode(BPETokenizer* tok, int* ids, int len) {
    if (tok == NULL || ids == NULL || len == 0) return NULL;

    // 计算总长度
    int total_len = 0;
    for (int i = 0; i < len; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            total_len += strlen(tok->vocab[id]);
        }
    }

    char* result = (char*)malloc(total_len + 1);
    result[0] = '\0';

    for (int i = 0; i < len; i++) {
        int id = ids[i];
        // 跳过特殊 token
        if (id == tok->pad_id || id == tok->bos_id || id == tok->eos_id) {
            continue;
        }
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            strcat(result, tok->vocab[id]);
        }
    }

    return result;
}

const char* bpe_decode_token(BPETokenizer* tok, int id) {
    if (tok == NULL || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

// ============ 工具函数 ============

int bpe_vocab_size(BPETokenizer* tok) {
    if (tok == NULL) return 0;
    return tok->vocab_size;
}

void bpe_print_stats(BPETokenizer* tok) {
    if (tok == NULL) {
        printf("BPE Tokenizer: NULL\n");
        return;
    }

    printf("BPE Tokenizer 统计:\n");
    printf("  词汇表大小: %d\n", tok->vocab_size);
    printf("  合并规则数: %d\n", tok->num_merges);
    printf("  特殊 tokens:\n");
    printf("    <pad>: %d\n", tok->pad_id);
    printf("    <unk>: %d\n", tok->unk_id);
    printf("    <bos>: %d\n", tok->bos_id);
    printf("    <eos>: %d\n", tok->eos_id);

    // 打印一些示例 tokens
    printf("  前 20 个 tokens:\n");
    for (int i = 0; i < 20 && i < tok->vocab_size; i++) {
        printf("    %d: '%s'\n", i, tok->vocab[i]);
    }

    if (tok->num_merges > 0) {
        printf("  最后 10 个合并的 tokens:\n");
        int start = tok->vocab_size - 10;
        if (start < 0) start = 4;  // 跳过特殊 token
        for (int i = start; i < tok->vocab_size; i++) {
            printf("    %d: '%s'\n", i, tok->vocab[i]);
        }
    }
}

void bpe_print_tokens(BPETokenizer* tok, int* ids, int len) {
    printf("[");
    for (int i = 0; i < len; i++) {
        if (i > 0) printf(", ");
        printf("'%s'(%d)", bpe_decode_token(tok, ids[i]), ids[i]);
    }
    printf("]\n");
}
